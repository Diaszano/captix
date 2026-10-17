"""CRNN model for text captcha recognition.

A small ResNet-style CNN extracts features, an adaptive pool fixes the height
to 1 so the feature map becomes a width-indexed sequence, then a bidirectional
LSTM produces per-step logits over the vocabulary (plus a CTC blank).

This architecture works with arbitrary input image sizes (height is normalized
during preprocessing; width is preserved). The output sequence length scales
with the input width, which is what CTC needs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .dataset import DEFAULT_VOCAB


class _BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for variable-length text recognition.

    Parameters
    ----------
    vocab: list of characters. Index 0 is reserved for the CTC blank.
    cnn_channels: tuple of channel counts for the four CNN stages.
    rnn_hidden: hidden size of the BiLSTM.
    rnn_layers: number of stacked BiLSTM layers.
    dropout: dropout applied before the RNN and the final linear layer.
    """

    def __init__(
        self,
        vocab: Sequence[str] = DEFAULT_VOCAB,
        cnn_channels: Sequence[int] = (64, 128, 256, 256),
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.vocab: list[str] = list(vocab)
        self.num_classes = len(self.vocab) + 1  # +1 for CTC blank

        c1, c2, c3, c4 = cnn_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        # Downsample height/width by 2 each stage. After 3 pools: H/8, W/8.
        # We keep width-only pooling on the final stage to preserve resolution.
        self.layer1 = nn.Sequential(_BasicBlock(c1, c1), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(_BasicBlock(c1, c2), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(_BasicBlock(c2, c3), nn.MaxPool2d((2, 1), (2, 1)))
        self.layer4 = _BasicBlock(c3, c4)
        # Collapse remaining height to 1 -> sequence over width
        self.collapse = nn.AdaptiveAvgPool2d((1, None))

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=c4,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(rnn_hidden * 2, self.num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sequence features (B, T, C) ready for the RNN."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.collapse(x)  # (B, C, 1, T)
        x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, T, C)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (T, B, num_classes) suitable for CTC loss."""
        feats = self.features(x)
        feats = self.dropout(feats)
        rnn_out, _ = self.rnn(feats)
        logits = self.fc(self.dropout(rnn_out))  # (B, T, C)
        return logits.transpose(0, 1).contiguous()  # (T, B, C)
