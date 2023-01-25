from typing import Tuple

import torch
import torch.nn as nn
from moduleZoo.attention import MultiHeadAttention2d
from moduleZoo.convolution import ConvNormActivation2d
from moduleZoo.dense import LinearNormActivation
from moduleZoo.resblocks import (
    ConvBottleNeckResidualBlock2d,
    ConvInvertedResidualBlock2d,
    ConvResidualBlock2d,
)


class FeatureExtractorBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 128,
        in_size: Tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = in_size

        # Layer1: Conv2d
        self.conv1 = ConvNormActivation2d(
            self.in_channels,
            8,
            kernel_size=5,
            stride=2,
            padding="stride_effective",
            activation_layer=nn.ReLU6,
        )

        # Layer2: ConvResidualBlock2d
        self.res2 = ConvBottleNeckResidualBlock2d(
            8, 2, 16, kernel_size=3, stride=2, activation_layer=nn.ReLU6
        )

        # Layer3: ConvInvertedResidualBlock2d
        self.res3 = ConvInvertedResidualBlock2d(
            16, 2, 32, kernel_size=3, stride=1, activation_layer=nn.ReLU6
        )

        # Layer4: ConvResidualBlock2d
        self.res4 = ConvResidualBlock2d(
            32,
            64,
            kernel_size=3,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6,
        )

        # Layer5: ConvBottleNeckResidualBlock2d
        self.res5 = ConvBottleNeckResidualBlock2d(
            64, 2, 128, kernel_size=3, stride=1, activation_layer=nn.ReLU6
        )

        # Layer6: ConvInvertedResidualBlock2d
        self.res6 = ConvInvertedResidualBlock2d(
            128, 2, 256, kernel_size=3, stride=1, activation_layer=nn.ReLU6
        )

        # Layer7: ConvResidual2d
        self.res7 = ConvResidualBlock2d(
            256,
            512,
            kernel_size=3,
            stride=2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SELU,
        )

        # Compression layer: ConvInvertedResidualBlock2d
        self.res_comp = ConvInvertedResidualBlock2d(
            512, self.out_channels, kernel_size=3, stride=2, activation_layer=nn.SELU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res_comp(x)

        return x


class MultiViewCrossAttention(nn.Module):
    def __init__(
        self, in_channels: int = 64, n_heads: int = 1, out_channels: int = 16
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.out_channels = out_channels

        self.cross_attention1 = MultiHeadAttention2d(
            self.in_channels, self.out_channels, n_heads=self.n_heads, kernel_size=3
        )

        self.cross_attention2 = MultiHeadAttention2d(
            self.in_channels, self.out_channels, n_heads=self.n_heads, kernel_size=3
        )

        self.cross_attention3 = MultiHeadAttention2d(
            self.in_channels, self.out_channels, n_heads=self.n_heads, kernel_size=3
        )

    def forward(
        self, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, vc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        yv1 = self.cross_attention1(vc, v1)
        yv2 = self.cross_attention1(vc, v2)
        yv3 = self.cross_attention1(vc, v3)

        return yv1, yv2, yv3


class DARClassifier(nn.Module):
    def __init__(
        self, in_channels: int = 128, hidden_channels: int = 256, n_class: int = 18
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_class = n_class

        self.conv_compressor = ConvNormActivation2d(
            self.in_channels * 3,
            self.hidden_channels,
            kernel_size=1,
            padding="stride_effective",
            activation_layer=nn.ReLU6,
        )

        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)

        self.linear1 = LinearNormActivation(
            self.hidden_channels, 128, activation_layer=nn.ReLU6
        )
        self.linear2 = LinearNormActivation(
            128, 64, norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU6
        )
        self.linear3 = LinearNormActivation(64, 32, activation_layer=nn.ReLU6)
        self.linear4 = LinearNormActivation(
            32, 18, bias=False, activation_layer=nn.ReLU6
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, yv1: torch.Tensor, yv2: torch.Tensor, yv3: torch.Tensor
    ) -> torch.Tensor:

        y_c = torch.cat((yv1, yv2, yv3), dim=1)

        features = self.conv_compressor(y_c)
        features = self.adaptive_pool(features).flatten(start_dim=1)

        features = self.linear1(features)
        features = self.linear2(features)
        features = self.linear3(features)
        features = self.linear4(features)

        cls_score = self.softmax(features)

        return cls_score
