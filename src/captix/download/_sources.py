"""Fetcher functions for each supported captcha source.

Ported from the R package ``jtrecenti/captchaDownload``. Each fetcher takes a
:class:`FetchContext`, issues whatever HTTP calls the source requires, and
returns a :class:`CaptchaPayload`. Parse failures should raise
:class:`DownloadError` so the batch loop can skip them.
"""
from __future__ import annotations

import base64
import re
import time

from ._base import CaptchaPayload, DownloadError, FetchContext, register


def _ensure_ok(resp, label: str) -> None:
    if resp.status_code != 200:
        raise DownloadError(f"{label}: HTTP {resp.status_code}")


def _decode_b64(b64: str) -> bytes:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64)
    except Exception as e:  # noqa: BLE001 — surface any decode error uniformly
        raise DownloadError(f"base64 decode failed: {e}") from None


# --------------------------------------------------------------------- cadesp

_CADESP_START = (
    "https://www.cadesp.fazenda.sp.gov.br/Pages/Cadastro/Consultas/"
    "ConsultaPublica/ConsultaPublica.aspx"
)
_CADESP_SESS_RE = re.compile(r"\(S\([^)]+\)\)")


@register("cadesp")
def fetch_cadesp(ctx: FetchContext) -> CaptchaPayload:
    # CADESP uses an ASP.NET "cookieless session": the session id is embedded
    # in the URL path as (S(...)). Hit the landing page, let it redirect,
    # then pull the segment out of the final URL.
    r = ctx.get(_CADESP_START, allow_redirects=True)
    _ensure_ok(r, "cadesp start")
    m = _CADESP_SESS_RE.search(r.url) or _CADESP_SESS_RE.search(r.text)
    if not m:
        raise DownloadError("cadesp: could not find (S(...)) session id")
    sess = m.group(0)
    captcha_url = (
        f"https://www.cadesp.fazenda.sp.gov.br/{sess}/imagemDinamica.aspx"
    )
    img = ctx.get(captcha_url)
    _ensure_ok(img, "cadesp image")
    return CaptchaPayload(content=img.content, ext="jpeg")


# ----------------------------------------------------------------------- esaj

_ESAJ_URL = "http://esaj.tjba.jus.br/cpopg/imagemCaptcha.do"


@register("esaj", verify_ssl=False)
def fetch_esaj(ctx: FetchContext) -> CaptchaPayload:
    body = {
        "timestamp": str(int(time.time() * 1000)),
        "uuidCaptcha": "",
        "conversationId": "",
    }
    r = ctx.post(_ESAJ_URL, data=body)
    _ensure_ok(r, "esaj captcha")
    try:
        data = r.json()
    except ValueError:
        raise DownloadError("esaj: non-JSON response")
    b64 = data.get("image") or data.get("imagem")
    if not b64:
        raise DownloadError("esaj: missing 'image' field")
    return CaptchaPayload(content=_decode_b64(b64), ext="png")


# --------------------------------------------------------------------- jucesp

_JUCESP_START = (
    "https://www.jucesponline.sp.gov.br/"
    "Pre_Visualiza.aspx?idproduto=&nire=35222827792"
)
_JUCESP_GUID_RE = re.compile(r"guid=([^\"'&\s]+)")


@register("jucesp")
def fetch_jucesp(ctx: FetchContext) -> CaptchaPayload:
    r = ctx.get(_JUCESP_START)
    _ensure_ok(r, "jucesp start")
    m = _JUCESP_GUID_RE.search(r.text)
    if not m:
        raise DownloadError("jucesp: could not find captcha guid")
    guid = m.group(1)
    img = ctx.get(
        f"https://www.jucesponline.sp.gov.br/CaptchaImage.aspx?guid={guid}"
    )
    _ensure_ok(img, "jucesp image")
    return CaptchaPayload(content=img.content, ext="jpeg")


# ------------------------------------------------------------------------ rfb

_RFB_SESSION = (
    "http://servicos.receita.fazenda.gov.br/Servicos/cnpjreva/"
    "Cnpjreva_Solicitacao_CS.asp"
)
_RFB_CAPTCHA = (
    "http://servicos.receita.fazenda.gov.br/Servicos/cnpjreva/"
    "captcha/gerarCaptcha.asp"
)


@register("rfb")
def fetch_rfb(ctx: FetchContext) -> CaptchaPayload:
    # RFB binds the captcha to an ASPSESSIONID cookie set by the landing
    # page; hit it once to warm the jar, then request the captcha.
    r = ctx.get(_RFB_SESSION)
    _ensure_ok(r, "rfb session")
    img = ctx.get(_RFB_CAPTCHA)
    _ensure_ok(img, "rfb captcha")
    return CaptchaPayload(content=img.content, ext="png")


# ------------------------------------------------------------------------ sei

_SEI_START = (
    "https://sei.economia.gov.br/sei/modulos/pesquisa/"
    "md_pesq_processo_pesquisar.php"
    "?acao_externa=protocolo_pesquisar&acao_origem_externa=protocolo_pesquisar"
    "&id_orgao_acesso_externo=0"
)
_SEI_IMG_RE = re.compile(
    r"""<img[^>]*src=["']([^"']*infra_gerar_captcha[^"']*)["']""",
    re.IGNORECASE,
)


@register("sei", verify_ssl=False)
def fetch_sei(ctx: FetchContext) -> CaptchaPayload:
    r = ctx.get(_SEI_START)
    _ensure_ok(r, "sei start")
    m = _SEI_IMG_RE.search(r.text)
    if not m:
        raise DownloadError("sei: could not find captcha img tag")
    src = m.group(1)
    if src.startswith("http"):
        url = src
    elif src.startswith("/"):
        url = "https://sei.economia.gov.br" + src
    else:
        url = "https://sei.economia.gov.br/" + src
    img = ctx.get(url)
    _ensure_ok(img, "sei image")
    return CaptchaPayload(content=img.content, ext="png")


# ----------------------------------------------------------------------- tjmg

_TJMG_URL = "http://www4.tjmg.jus.br/juridico/sf/captcha.svl"


@register("tjmg")
def fetch_tjmg(ctx: FetchContext) -> CaptchaPayload:
    # The simple .svl endpoint returns a fresh captcha without needing a
    # parent JSP session. Good enough for download-only.
    r = ctx.get(_TJMG_URL)
    _ensure_ok(r, "tjmg captcha")
    return CaptchaPayload(content=r.content, ext="jpeg")


# ----------------------------------------------------------------------- tjpe

_TJPE_URL = (
    "https://srv01.tjpe.jus.br/consultaprocessualunificadaservico/api/captcha"
)


@register("tjpe", verify_ssl=False)
def fetch_tjpe(ctx: FetchContext) -> CaptchaPayload:
    r = ctx.get(_TJPE_URL)
    _ensure_ok(r, "tjpe captcha")
    try:
        data = r.json()
    except ValueError:
        raise DownloadError("tjpe: non-JSON response")
    b64 = data.get("image") or data.get("imagem")
    if not b64:
        raise DownloadError("tjpe: missing 'image' field")
    return CaptchaPayload(content=_decode_b64(b64), ext="png")


# ----------------------------------------------------------------------- tjrs

_TJRS_START = "http://www.tjrs.jus.br/busca/?tb=proc"
_TJRS_CAPTCHA_TPL = (
    "https://www.tjrs.jus.br/site_php/consulta/human_check/"
    "humancheck_showcode.php?{cache}"
)


@register("tjrs", verify_ssl=False)
def fetch_tjrs(ctx: FetchContext) -> CaptchaPayload:
    r = ctx.get(_TJRS_START)
    _ensure_ok(r, "tjrs start")
    url = _TJRS_CAPTCHA_TPL.format(cache=int(time.time() * 1000))
    img = ctx.get(url)
    _ensure_ok(img, "tjrs captcha")
    return CaptchaPayload(content=img.content, ext="jpeg")


# ----------------------------------------------------------------------- trf5

_TRF5_URL = "https://pje.trf5.jus.br/pjeconsulta/seam/resource/captcha"


@register("trf5")
def fetch_trf5(ctx: FetchContext) -> CaptchaPayload:
    r = ctx.get(_TRF5_URL)
    _ensure_ok(r, "trf5 captcha")
    return CaptchaPayload(content=r.content, ext="jpeg")


# ------------------------------------------------------------------------ trt

_TRT_URL = (
    "https://pje-consulta.trt3.jus.br/pje-consulta-api/api/captcha"
    "?idProcesso=2104879"
)


@register("trt")
def fetch_trt(ctx: FetchContext) -> CaptchaPayload:
    r = ctx.get(_TRT_URL)
    _ensure_ok(r, "trt captcha")
    try:
        data = r.json()
    except ValueError:
        raise DownloadError("trt: non-JSON response")
    b64 = data.get("imagem") or data.get("image")
    if not b64:
        raise DownloadError("trt: missing 'imagem' field")
    return CaptchaPayload(content=_decode_b64(b64), ext="jpeg")
