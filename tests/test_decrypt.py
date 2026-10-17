import re

from captix import CRNN, decrypt
from captix.decrypt import _resolve_mask


def test_resolve_mask_list():
    vocab = ["a", "b", "c", "1"]
    mask = _resolve_mask(["a", "1"], vocab, case_sensitive=True)
    assert mask == [True, False, False, True]


def test_resolve_mask_list_case_insensitive():
    vocab = ["a", "B", "c"]
    mask = _resolve_mask(["a"], vocab, case_sensitive=False)
    assert mask == [True, False, False]
    mask = _resolve_mask(["B"], vocab, case_sensitive=False)
    assert mask[1] is True


def test_resolve_mask_regex_string():
    vocab = ["0", "1", "a", "B"]
    mask = _resolve_mask("[0-9a-z]", vocab, case_sensitive=True)
    assert mask == [True, True, True, False]


def test_resolve_mask_compiled_regex():
    vocab = ["0", "1", "a"]
    mask = _resolve_mask(re.compile(r"\d"), vocab, case_sensitive=True)
    assert mask == [True, True, False]


def test_decrypt_returns_strings(tmp_captcha_file):
    model = CRNN()
    out = decrypt(tmp_captcha_file, model)
    assert isinstance(out, list)
    assert isinstance(out[0], str)


def test_decrypt_with_mask_restricts_chars(tmp_captcha_file):
    model = CRNN()
    out = decrypt(tmp_captcha_file, model, mask="[0-9]")
    assert all(ch.isdigit() or ch == "" for ch in (out[0],))
    for ch in out[0]:
        assert ch.isdigit()


def test_decrypt_case_insensitive_lowercases(tmp_captcha_file):
    model = CRNN()
    out = decrypt(tmp_captcha_file, model, mask="[a-zA-Z]", case_sensitive=False)
    assert out[0] == out[0].lower()


def test_decrypt_fixed_length_enforced(tmp_captcha_file):
    model = CRNN()
    for L in (3, 5):
        out = decrypt(tmp_captcha_file, model, length=L)
        assert len(out[0]) == L


def test_decrypt_fixed_length_honors_mask(tmp_captcha_file):
    model = CRNN()
    out = decrypt(tmp_captcha_file, model, mask="[0-9]", length=4)
    assert len(out[0]) == 4
    assert out[0].isdigit()
