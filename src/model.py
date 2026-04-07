from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net for binary segmentation of synthetic quantum-dot images."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, features: list[int] | None = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(_DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        self.bottleneck = _DoubleConv(ch, ch * 2)
        ch = ch * 2

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(f * 2, f))
            ch = f

        self.out_conv = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)


def build_model(cfg: dict) -> UNet:
    features = list(cfg.get("encoder_channels", [32, 64, 128]))
    return UNet(in_channels=1, out_channels=2, features=features)
