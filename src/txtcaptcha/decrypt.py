"""decrypt: predict labels for captcha images using a trained CRNN model."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from .captcha import Captcha
from .dataset import BLANK_INDEX, decode_indices, transform_image
from .model import CRNN

PathLike = Union[str, Path]
MaskType = Optional[Union[Sequence[str], str, re.Pattern]]


def _resolve_mask(
    mask: MaskType,
    vocab: Sequence[str],
    case_sensitive: bool,
) -> Optional[List[bool]]:
    """Returns a boolean list of length len(vocab): True for allowed chars.

    None means "no masking".
    """
    if mask is None:
        return None

    if isinstance(mask, (list, tuple, set)):
        allowed = set(mask)
        if not case_sensitive:
            allowed = {c.lower() for c in allowed} | {c.upper() for c in allowed}
        return [c in allowed for c in vocab]

    if isinstance(mask, re.Pattern):
        pat = mask
    elif isinstance(mask, str):
        # Treat as a regex; if not a character class, wrap it.
        pat_src = mask if mask.startswith("[") else f"[{mask}]"
        flags = 0 if case_sensitive else re.IGNORECASE
        pat = re.compile(pat_src, flags)
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)!r}")

    return [bool(pat.fullmatch(c)) for c in vocab]


def _apply_mask(logits: torch.Tensor, allowed: Optional[List[bool]]) -> torch.Tensor:
    """Set logits of forbidden classes to -inf. Blank stays untouched."""
    if allowed is None:
        return logits
    mask_t = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask_t[BLANK_INDEX] = True  # always allow blank
    for i, ok in enumerate(allowed):
        if ok:
            mask_t[i + 1] = True
    forbidden = ~mask_t
    return logits.masked_fill(forbidden, float("-inf"))


def _ctc_decode_fixed_length(log_probs: np.ndarray, length: int) -> List[int]:
    """Find the CTC path whose collapse has exactly ``length`` non-blank chars
    and the highest total log-probability.

    Exact dynamic programming over states ``(n, last)`` where ``n`` is the
    number of collapsed chars emitted so far and ``last`` is the index chosen
    at the previous timestep (needed to detect repeat-collapses). Complexity
    is ``O(T * (L+1) * C)`` thanks to the top-2 trick in case C. Returns the
    per-timestep argmax-like path; the caller still CTC-collapses it.
    """
    T, C = log_probs.shape
    L = length
    if T < L:
        raise ValueError(
            f"Cannot decode length={L}: model produced only T={T} timesteps. "
            "Use a wider input image or a smaller length."
        )
    NEG = -np.inf
    dp = np.full((L + 1, C), NEG)
    dp[0, BLANK_INDEX] = 0.0  # virtual start: 0 emitted, last=blank, logp=0
    bps: List[np.ndarray] = []
    ns = np.arange(L + 1)
    i_row = np.arange(C)[None, :]
    for t in range(T):
        lp_t = log_probs[t]
        new_dp = np.full((L + 1, C), NEG)
        bp = np.full((L + 1, C), -1, dtype=np.int64)

        # Case A: emit blank. Target state (n, blank). Pick best previous `last`.
        best_val = dp.max(axis=1)
        best_last = dp.argmax(axis=1)
        cand_a = best_val + lp_t[BLANK_INDEX]
        better_a = cand_a > new_dp[:, BLANK_INDEX]
        new_dp[better_a, BLANK_INDEX] = cand_a[better_a]
        bp[better_a, BLANK_INDEX] = (ns * C + best_last)[better_a]

        # Case B: emit i != blank, i == last (repeat collapse). Target (n, i).
        cand_b = dp + lp_t[None, :]
        cand_b[:, BLANK_INDEX] = NEG
        better_b = cand_b > new_dp
        new_dp = np.where(better_b, cand_b, new_dp)
        bp = np.where(better_b, ns[:, None] * C + i_row, bp)

        # Case C: emit i != blank, i != last. Target (n+1, i).
        if L > 0:
            order = np.argsort(-dp, axis=1)
            top1_idx = order[:, 0]
            top2_idx = order[:, 1]
            top1_val = dp[ns, top1_idx]
            top2_val = dp[ns, top2_idx]
            is_top1 = top1_idx[:, None] == i_row
            best_excl_val = np.where(is_top1, top2_val[:, None], top1_val[:, None])
            best_excl_idx = np.where(is_top1, top2_idx[:, None], top1_idx[:, None])
            cand_c = best_excl_val + lp_t[None, :]
            cand_c[:, BLANK_INDEX] = NEG
            src_cand = cand_c[:-1]
            src_excl_idx = best_excl_idx[:-1]
            src_n = ns[:-1]
            better_c = src_cand > new_dp[1:]
            new_dp[1:] = np.where(better_c, src_cand, new_dp[1:])
            bp[1:] = np.where(better_c, src_n[:, None] * C + src_excl_idx, bp[1:])

        dp = new_dp
        bps.append(bp)

    final = dp[L]
    if not np.isfinite(final).any():
        raise ValueError(f"No CTC path of length {L} is feasible for these logits.")
    best_last = int(np.argmax(final))
    path_rev = [best_last]
    n, last = L, best_last
    for t in range(T - 1, 0, -1):
        prev = int(bps[t][n, last])
        pn, pl = divmod(prev, C)
        path_rev.append(pl)
        n, last = pn, pl
    path_rev.reverse()
    return path_rev


@torch.no_grad()
def decrypt(
    files: Union[PathLike, Sequence[PathLike], Captcha],
    model: CRNN,
    mask: MaskType = None,
    case_sensitive: bool = True,
    length: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> List[str]:
    """Predict labels for one or more captcha images.

    Parameters
    ----------
    files: file path, list of paths, or a Captcha object.
    model: a trained `CRNN` instance.
    mask: optional restriction on output characters. Either a list of allowed
        characters (e.g. ``['a','b','c']``), a regex character class
        (e.g. ``'[0-9a-z]'`` or ``'0-9'``), or a compiled `re.Pattern`.
    case_sensitive: when False and a list/string mask is given, both cases
        are accepted; output is also lowercased.
    length: if set, force the decoded output to contain exactly this many
        characters. Uses an exact CTC DP over constrained paths — strictly
        better than greedy when you know the captcha length up-front, since
        it never emits a wrong-length prediction. Leave as ``None`` to use
        the (variable-length) greedy CTC decoder.
    device: torch device for inference. Defaults to model's current device.
    """
    if isinstance(files, Captcha):
        path_list: List[Union[str, Path]] = list(files.paths)
    elif isinstance(files, (str, Path)):
        path_list = [files]
    else:
        path_list = list(files)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    images = transform_image(path_list).to(device)
    logits = model(images)  # (T, B, C)
    allowed = _resolve_mask(mask, model.vocab, case_sensitive)
    logits = _apply_mask(logits, allowed)

    results: List[str] = []
    if length is None:
        indices = logits.argmax(dim=-1).transpose(0, 1)  # (B, T)
        for row in indices:
            text = decode_indices(row.tolist(), model.vocab, collapse=True)
            if not case_sensitive:
                text = text.lower()
            results.append(text)
    else:
        log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()  # (T, B, C)
        for b in range(log_probs.shape[1]):
            path = _ctc_decode_fixed_length(log_probs[:, b], length)
            text = decode_indices(path, model.vocab, collapse=True)
            if not case_sensitive:
                text = text.lower()
            results.append(text)
    return results
