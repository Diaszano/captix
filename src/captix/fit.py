"""Training loop for the CRNN captcha model."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .dataset import (
    BLANK_INDEX,
    DEFAULT_VOCAB,
    CaptchaDataset,
    decode_indices,
    pad_collate,
)
from .model import CRNN

PathLike = str | Path


@dataclass
class FitHistory:
    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    valid_acc: list[float] = field(default_factory=list)


def _split_dataset(ds: CaptchaDataset, prop_valid: float, seed: int = 0):
    n = len(ds)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_valid = max(1, round(prop_valid * n))
    return Subset(ds, idx[n_valid:]), Subset(ds, idx[:n_valid])


def _greedy_decode(logits: torch.Tensor, vocab: Sequence[str]) -> list[str]:
    """logits: (T, B, C). Returns list of B decoded strings."""
    indices = logits.argmax(dim=-1).transpose(0, 1)  # (B, T)
    return [decode_indices(row.tolist(), vocab, collapse=True) for row in indices]


def _evaluate(
    model: CRNN,
    loader: DataLoader,
    loss_fn: torch.nn.CTCLoss,
    device: torch.device,
    vocab: Sequence[str],
):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            logits = model(images)  # (T, B, C)
            T_dim, B = logits.shape[0], logits.shape[1]
            input_lengths = torch.full((B,), T_dim, dtype=torch.long, device=device)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            n_batches += 1

            preds = _greedy_decode(logits, vocab)
            # Reconstruct gold strings from concatenated targets
            offset = 0
            for i, ln in enumerate(target_lengths.tolist()):
                gold_idx = targets[offset : offset + ln].tolist()
                offset += ln
                gold = "".join(vocab[j - 1] for j in gold_idx if j != BLANK_INDEX)
                if preds[i] == gold:
                    correct += 1
                total += 1

    return total_loss / max(1, n_batches), correct / max(1, total)


def fit_model(
    dir: PathLike,
    dir_valid: PathLike | None = None,
    prop_valid: float = 0.2,
    vocab: Sequence[str] = DEFAULT_VOCAB,
    height: int = 32,
    case_sensitive: bool = True,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    dropout: float = 0.25,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.01,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    seed: int = 0,
    verbose: bool = True,
):
    """Train a CRNN captcha model on a directory of labeled images.

    Returns a tuple ``(model, history)``.
    """
    train_ds_full = CaptchaDataset(
        dir, vocab=vocab, height=height, case_sensitive=case_sensitive
    )

    if dir_valid is None:
        train_subset, valid_subset = _split_dataset(
            train_ds_full, prop_valid, seed=seed
        )
    else:
        train_subset = train_ds_full
        valid_subset = CaptchaDataset(
            dir_valid, vocab=vocab, height=height, case_sensitive=case_sensitive
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=num_workers,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = CRNN(vocab=vocab, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    history = FitHistory()

    best_acc = -math.inf
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for images, targets, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)  # (T, B, C)
            T_dim, B = logits.shape[0], logits.shape[1]
            input_lengths = torch.full((B,), T_dim, dtype=torch.long, device=device)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += loss.item()
            n_batches += 1

        train_loss = running / max(1, n_batches)
        valid_loss, valid_acc = _evaluate(model, valid_loader, loss_fn, device, vocab)

        history.train_loss.append(train_loss)
        history.valid_loss.append(valid_loss)
        history.valid_acc.append(valid_acc)

        if verbose:
            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
            )

        if valid_acc > best_acc + early_stopping_min_delta:
            best_acc = valid_acc
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    return model, history
