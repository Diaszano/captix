"""Tests for captix.download. All network calls are mocked — these
tests must not touch the internet."""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from captix.download import _base, _sources
from captix.download._base import (
    CaptchaPayload,
    DownloadError,
    FetchContext,
    Source,
    _REGISTRY,
    available_sources,
    download_captchas,
    get_source,
)
from captix.download.cli import main as cli_main, _resolve_sources

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fakepngdata"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"fakejpegdata"


def _response(content=b"", text=None, json_data=None, url="", status=200):
    r = MagicMock()
    r.status_code = status
    r.content = content
    r.text = text if text is not None else content.decode("latin-1", errors="replace")
    r.url = url
    if json_data is not None:
        r.json.return_value = json_data
    else:
        r.json.side_effect = ValueError("no json")
    return r


class FakeSession:
    """Stand-in for requests.Session. Routes are keyed by (method, url_prefix)
    and return a zero-arg factory so each call gets a fresh response mock.
    """

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.closed = False

    def _lookup(self, method, url):
        for key, factory in self.routes.items():
            m, prefix = key
            if m == method and url.startswith(prefix):
                return factory()
        raise AssertionError(f"Unexpected {method} {url}")

    def get(self, url, **kwargs):
        return self._lookup("GET", url)

    def post(self, url, **kwargs):
        return self._lookup("POST", url)

    def close(self):
        self.closed = True


def _ctx(session):
    return FetchContext(session=session, timeout=1.0)


# --------------------------------------------------------------- registry API


def test_available_sources_contains_all_ten():
    assert set(available_sources()) == {
        "cadesp", "esaj", "jucesp", "rfb", "sei",
        "tjmg", "tjpe", "tjrs", "trf5", "trt",
    }


def test_get_source_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_source("not-a-source")


def test_get_source_returns_source_object():
    src = get_source("esaj")
    assert src.name == "esaj"
    assert src.verify_ssl is False


# -------------------------------------------------------------------- cadesp


def test_fetch_cadesp_happy_path():
    session = FakeSession({
        ("GET", "https://www.cadesp.fazenda.sp.gov.br/Pages/"):
            lambda: _response(
                url=(
                    "https://www.cadesp.fazenda.sp.gov.br/"
                    "(S(abc123def456))/Pages/Cadastro/Consultas/"
                    "ConsultaPublica/ConsultaPublica.aspx"
                ),
                text="<html/>",
            ),
        ("GET", "https://www.cadesp.fazenda.sp.gov.br/(S(abc123def456))/"):
            lambda: _response(content=JPEG_BYTES),
    })
    payload = _sources.fetch_cadesp(_ctx(session))
    assert payload.content == JPEG_BYTES
    assert payload.ext == "jpeg"


def test_fetch_cadesp_missing_session_id_raises():
    session = FakeSession({
        ("GET", "https://www.cadesp.fazenda.sp.gov.br/Pages/"):
            lambda: _response(url="https://www.cadesp.fazenda.sp.gov.br/err", text="boom"),
    })
    with pytest.raises(DownloadError):
        _sources.fetch_cadesp(_ctx(session))


# ---------------------------------------------------------------------- esaj


def test_fetch_esaj_decodes_data_uri():
    b64 = base64.b64encode(PNG_BYTES).decode()
    session = FakeSession({
        ("POST", "http://esaj.tjba.jus.br/cpopg/imagemCaptcha.do"):
            lambda: _response(json_data={
                "image": f"data:image/png;base64,{b64}",
                "uuidCaptcha": "u",
            }),
    })
    payload = _sources.fetch_esaj(_ctx(session))
    assert payload.content == PNG_BYTES
    assert payload.ext == "png"


def test_fetch_esaj_non_json_raises():
    session = FakeSession({
        ("POST", "http://esaj.tjba.jus.br/cpopg/imagemCaptcha.do"):
            lambda: _response(content=b"<html>down</html>"),
    })
    with pytest.raises(DownloadError):
        _sources.fetch_esaj(_ctx(session))


# -------------------------------------------------------------------- jucesp


def test_fetch_jucesp_scrapes_guid():
    html = '<html><img src="CaptchaImage.aspx?guid=XYZ-789"/></html>'
    session = FakeSession({
        ("GET", "https://www.jucesponline.sp.gov.br/Pre_Visualiza.aspx"):
            lambda: _response(text=html),
        ("GET", "https://www.jucesponline.sp.gov.br/CaptchaImage.aspx?guid=XYZ-789"):
            lambda: _response(content=JPEG_BYTES),
    })
    payload = _sources.fetch_jucesp(_ctx(session))
    assert payload.ext == "jpeg"
    assert payload.content == JPEG_BYTES


def test_fetch_jucesp_missing_guid_raises():
    session = FakeSession({
        ("GET", "https://www.jucesponline.sp.gov.br/Pre_Visualiza.aspx"):
            lambda: _response(text="<html>nope</html>"),
    })
    with pytest.raises(DownloadError):
        _sources.fetch_jucesp(_ctx(session))


# ----------------------------------------------------------------------- rfb


def test_fetch_rfb_uses_session_then_captcha():
    session = FakeSession({
        ("GET", "http://servicos.receita.fazenda.gov.br/Servicos/cnpjreva/Cnpjreva_Solicitacao_CS.asp"):
            lambda: _response(text="warm"),
        ("GET", "http://servicos.receita.fazenda.gov.br/Servicos/cnpjreva/captcha/"):
            lambda: _response(content=PNG_BYTES),
    })
    payload = _sources.fetch_rfb(_ctx(session))
    assert payload.content == PNG_BYTES
    assert payload.ext == "png"


# ----------------------------------------------------------------------- sei


def test_fetch_sei_scrapes_img_src():
    html = '<img src="/infra_js/infra_gerar_captcha.php?codetorandom=65-90">'
    session = FakeSession({
        ("GET", "https://sei.economia.gov.br/sei/modulos/pesquisa/"):
            lambda: _response(text=html),
        ("GET", "https://sei.economia.gov.br/infra_js/infra_gerar_captcha.php"):
            lambda: _response(content=PNG_BYTES),
    })
    payload = _sources.fetch_sei(_ctx(session))
    assert payload.content == PNG_BYTES


def test_fetch_sei_missing_img_raises():
    session = FakeSession({
        ("GET", "https://sei.economia.gov.br/sei/modulos/pesquisa/"):
            lambda: _response(text="<html>gone</html>"),
    })
    with pytest.raises(DownloadError):
        _sources.fetch_sei(_ctx(session))


# ---------------------------------------------------------------------- tjmg


def test_fetch_tjmg_happy_path():
    session = FakeSession({
        ("GET", "http://www4.tjmg.jus.br/juridico/sf/captcha.svl"):
            lambda: _response(content=JPEG_BYTES),
    })
    payload = _sources.fetch_tjmg(_ctx(session))
    assert payload.ext == "jpeg"


def test_fetch_tjmg_http_error_raises():
    session = FakeSession({
        ("GET", "http://www4.tjmg.jus.br/juridico/sf/captcha.svl"):
            lambda: _response(content=b"", status=500),
    })
    with pytest.raises(DownloadError):
        _sources.fetch_tjmg(_ctx(session))


# ---------------------------------------------------------------------- tjpe


def test_fetch_tjpe_decodes_base64():
    b64 = base64.b64encode(PNG_BYTES).decode()
    session = FakeSession({
        ("GET", "https://srv01.tjpe.jus.br/"):
            lambda: _response(json_data={"image": f"data:image/png;base64,{b64}"}),
    })
    payload = _sources.fetch_tjpe(_ctx(session))
    assert payload.content == PNG_BYTES


# ---------------------------------------------------------------------- tjrs


def test_fetch_tjrs_warms_session_then_fetches():
    session = FakeSession({
        ("GET", "http://www.tjrs.jus.br/busca/"):
            lambda: _response(text="ok"),
        ("GET", "https://www.tjrs.jus.br/site_php/consulta/human_check/"):
            lambda: _response(content=JPEG_BYTES),
    })
    payload = _sources.fetch_tjrs(_ctx(session))
    assert payload.content == JPEG_BYTES


# ---------------------------------------------------------------------- trf5


def test_fetch_trf5_returns_bytes():
    session = FakeSession({
        ("GET", "https://pje.trf5.jus.br/pjeconsulta/seam/resource/captcha"):
            lambda: _response(content=JPEG_BYTES),
    })
    payload = _sources.fetch_trf5(_ctx(session))
    assert payload.ext == "jpeg"
    assert payload.content == JPEG_BYTES


# ----------------------------------------------------------------------- trt


def test_fetch_trt_decodes_plain_base64():
    b64 = base64.b64encode(JPEG_BYTES).decode()
    session = FakeSession({
        ("GET", "https://pje-consulta.trt3.jus.br/"):
            lambda: _response(json_data={"imagem": b64, "tokenDesafio": "tok"}),
    })
    payload = _sources.fetch_trt(_ctx(session))
    assert payload.content == JPEG_BYTES
    assert payload.ext == "jpeg"


def test_fetch_trt_missing_field_raises():
    session = FakeSession({
        ("GET", "https://pje-consulta.trt3.jus.br/"):
            lambda: _response(json_data={"other": 1}),
    })
    with pytest.raises(DownloadError):
        _sources.fetch_trt(_ctx(session))


# ------------------------------------------------------------- batch loop


@pytest.fixture
def fake_source_factory(monkeypatch):
    """Register a dummy source and patch make_session to return an empty
    FakeSession. Cleans up the registry after the test."""
    added: list[str] = []

    def _add(name, fetcher):
        _REGISTRY[name] = Source(name=name, fetcher=fetcher)
        added.append(name)

    monkeypatch.setattr(_base, "make_session", lambda verify=True: FakeSession())
    yield _add
    for name in added:
        _REGISTRY.pop(name, None)


def test_download_captchas_skips_failures(tmp_path, fake_source_factory):
    calls = {"i": 0}

    def flaky(ctx):
        calls["i"] += 1
        if calls["i"] == 1:
            raise DownloadError("transient")
        return CaptchaPayload(content=PNG_BYTES, ext="png")

    fake_source_factory("_flaky", flaky)
    with pytest.warns(UserWarning):
        out = download_captchas("_flaky", n=2, dest_dir=tmp_path)
    assert len(out) == 2
    assert all(p.exists() and p.read_bytes() == PNG_BYTES for p in out)
    assert calls["i"] == 3


def test_download_captchas_gives_up_on_persistent_failure(tmp_path, fake_source_factory):
    def always_fail(ctx):
        raise DownloadError("nope")

    fake_source_factory("_dead", always_fail)
    with pytest.warns(UserWarning):
        out = download_captchas("_dead", n=5, dest_dir=tmp_path)
    assert out == []
    # dest dir may or may not exist — no files were saved
    assert not any(tmp_path.iterdir()) or all(p.is_dir() for p in tmp_path.iterdir())


def test_download_captchas_on_error_raise_propagates(tmp_path, fake_source_factory):
    def always_fail(ctx):
        raise DownloadError("boom")

    fake_source_factory("_boom", always_fail)
    with pytest.raises(DownloadError):
        download_captchas("_boom", n=3, dest_dir=tmp_path, on_error="raise")


def test_download_captchas_requests_exception_is_skipped(tmp_path, fake_source_factory):
    import requests

    calls = {"i": 0}

    def flaky(ctx):
        calls["i"] += 1
        if calls["i"] < 2:
            raise requests.ConnectionError("net down")
        return CaptchaPayload(content=JPEG_BYTES, ext="jpeg")

    fake_source_factory("_net", flaky)
    with pytest.warns(UserWarning):
        out = download_captchas("_net", n=1, dest_dir=tmp_path)
    assert len(out) == 1


def test_download_captchas_writes_correct_extension(tmp_path, fake_source_factory):
    def ok(ctx):
        return CaptchaPayload(content=PNG_BYTES, ext="png")

    fake_source_factory("_ok", ok)
    out = download_captchas("_ok", n=2, dest_dir=tmp_path)
    assert all(p.suffix == ".png" for p in out)
    assert all(p.stem.startswith("_ok_") for p in out)


# ------------------------------------------------------------------------- CLI


def test_resolve_sources_all_expands():
    assert _resolve_sources(["all"]) == available_sources()
    assert _resolve_sources(None) == available_sources()


def test_resolve_sources_unknown_exits():
    with pytest.raises(SystemExit):
        _resolve_sources(["not-a-source"])


def test_resolve_sources_preserves_order_and_dedup():
    assert _resolve_sources(["trt", "esaj", "trt"]) == ["trt", "esaj"]


def test_cli_main_invokes_download(monkeypatch, tmp_path):
    seen = []

    def fake(source, n, dest_dir, **kw):
        seen.append((source, n, Path(dest_dir)))
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        out = Path(dest_dir) / "x.png"
        out.write_bytes(PNG_BYTES)
        return [out]

    monkeypatch.setattr("captix.download.cli.download_captchas", fake)
    ret = cli_main([
        "--source", "esaj", "--n", "3", "--dest", str(tmp_path),
    ])
    assert ret == 0
    assert seen == [("esaj", 3, tmp_path / "esaj")]


def test_cli_main_all_sources(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "captix.download.cli.download_captchas",
        lambda source, n, dest_dir, **kw: seen.append(source) or [],
    )
    cli_main(["--source", "all", "--n", "1", "--dest", str(tmp_path)])
    assert set(seen) == set(available_sources())


def test_cli_main_returns_1_when_nothing_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "captix.download.cli.download_captchas",
        lambda source, n, dest_dir, **kw: [],
    )
    ret = cli_main(["--source", "esaj", "--n", "1", "--dest", str(tmp_path)])
    assert ret == 1
