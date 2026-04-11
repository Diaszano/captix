import torch

from txtcaptcha import CRNN, DEFAULT_VOCAB


def test_model_forward_shapes():
    model = CRNN()
    x = torch.randn(2, 3, 32, 160)
    out = model(x)
    # (T, B, C)
    assert out.ndim == 3
    assert out.shape[1] == 2
    assert out.shape[2] == len(DEFAULT_VOCAB) + 1


def test_model_variable_widths():
    model = CRNN()
    out_a = model(torch.randn(1, 3, 32, 80))
    out_b = model(torch.randn(1, 3, 32, 240))
    assert out_a.shape[2] == out_b.shape[2]
    assert out_a.shape[0] != out_b.shape[0]  # T scales with W


def test_model_variable_height():
    model = CRNN()
    # height >= 16 needed because of three /2 down-samples (16/8=2)
    out = model(torch.randn(1, 3, 48, 100))
    assert out.shape[2] == len(DEFAULT_VOCAB) + 1


def test_custom_vocab():
    vocab = list("abc")
    model = CRNN(vocab=vocab)
    out = model(torch.randn(1, 3, 32, 100))
    assert out.shape[2] == 4  # 3 + blank
