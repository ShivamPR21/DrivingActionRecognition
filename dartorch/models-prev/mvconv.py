from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from moduleZoo.attention import SelfAttention1d, SelfAttention2d
from moduleZoo.convolution import Conv2DNormActivation, ConvNormActivation1d
from moduleZoo.resblocks import Conv2DResidualBlock, ConvResidualBlock1d


class MultiViewConvBackbone(nn.Module):
    def __init__(
        self,
        n_views: int = 3,
        n_channels: int = 1,
        n_frames: int = 5,
        in_size: Tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.in_size = in_size

        self.out_size = np.array(list(in_size), dtype=int)
        self.conv1 = Conv2DNormActivation(
            self.n_views * self.n_channels,
            64,
            kernel_size=5,
            stride=2,
            padding="stride_effective",
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6,
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        self.res2 = Conv2DResidualBlock(
            64, 64, kernel_size=3, stride=1, activation_layer=nn.ReLU6
        )

        self.res3 = Conv2DResidualBlock(
            64,
            64,
            kernel_size=3,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6,
        )

        # self.norm_3 = nn.BatchNorm2d(64)
        self.attn_3 = SelfAttention2d(64, nn.ReLU6)

        self.res4 = Conv2DResidualBlock(
            64, 128, kernel_size=3, stride=2, activation_layer=nn.ReLU6
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        self.res5 = Conv2DResidualBlock(
            128, 128, kernel_size=3, stride=1, activation_layer=nn.ReLU6
        )

        self.res6 = Conv2DResidualBlock(
            128,
            128,
            kernel_size=3,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6,
        )

        # self.norm_6 = nn.BatchNorm2d(128)
        self.attn_6 = SelfAttention2d(128, nn.ReLU6)

        self.res7 = Conv2DResidualBlock(
            128, 256, kernel_size=3, stride=2, activation_layer=nn.ReLU6
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        self.res8 = Conv2DResidualBlock(
            256, 256, kernel_size=3, stride=1, activation_layer=nn.ReLU6
        )

        self.res9 = Conv2DResidualBlock(
            256,
            256,
            kernel_size=3,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6,
        )

        # self.norm_9 = nn.BatchNorm2d(256)
        self.attn_9 = SelfAttention2d(256, nn.ReLU6)

        self.res10 = Conv2DResidualBlock(
            256, 512, kernel_size=3, stride=2, activation_layer=nn.ReLU6
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        self.res11 = Conv2DResidualBlock(
            512, 512, kernel_size=3, stride=1, activation_layer=nn.ReLU6
        )

        self.res12 = Conv2DResidualBlock(
            512,
            512,
            kernel_size=3,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6,
        )

        # self.norm_12 = nn.BatchNorm2d(512)
        self.attn_12 = SelfAttention2d(512, nn.ReLU6)

        pool_kernel_size = tuple(self.out_size)
        self.pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()

        x = x.split(
            split_size=self.n_views * self.n_channels, dim=1
        )  # Split the data in sequences ([B, n_views*n_channels, H, W])*n_seq
        x = torch.cat(
            x, dim=0
        )  # Concatenate the sequences to batch dimension [B*n_seq, n_views*n_channels, H, W]

        x = self.conv1(x)
        x = self.res2(x)

        x = self.res3(x)
        x, _ = self.attn_3(x)

        x = self.res4(x)
        x = self.res5(x)

        x = self.res6(x)
        x, _ = self.attn_6(x)

        x = self.res7(x)
        x = self.res8(x)

        x = self.res9(x)
        x, _ = self.attn_9(x)

        x = self.res10(x)
        x = self.res11(x)

        x = self.res12(x)
        x, _ = self.attn_12(x)

        x = self.pool(x)

        x = x.flatten(start_dim=1)

        x = x.unsqueeze(dim=1)
        x = x.split(split_size=B, dim=0)
        x = torch.cat(x, dim=1)

        return x


class MultiFrameFeatureMixer(nn.Module):
    def __init__(self, n_frames: int = 5, n_features: int = 512) -> None:
        super().__init__()

        self.n_frames = n_frames
        self.n_features = n_features

        self.out_size = n_features

        self.conv1 = ConvNormActivation1d(
            self.n_frames,
            16,
            kernel_size=5,
            stride=1,
            padding="stride_effective",
            bias=True,
            activation_layer=nn.ReLU6,
        )

        self.conv2 = ConvNormActivation1d(
            16,
            64,
            kernel_size=3,
            stride=1,
            padding="stride_effective",
            bias=False,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU6,
        )

        self.attn_2 = SelfAttention1d(64, nn.ReLU6)

        self.res3 = ConvResidualBlock1d(
            64, 128, kernel_size=3, stride=2, activation_layer=nn.ReLU6
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        self.res4 = ConvResidualBlock1d(
            128,
            128,
            kernel_size=3,
            stride=2,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU6,
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        # self.norm_4 = nn.BatchNorm1d(128)
        self.attn_4 = SelfAttention1d(128, nn.ReLU6)

        self.res5 = ConvResidualBlock1d(
            128, 128, kernel_size=3, stride=2, activation_layer=nn.ReLU6
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        self.res6 = ConvResidualBlock1d(
            128,
            128,
            kernel_size=3,
            stride=2,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU6,
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        # self.norm_6 = nn.BatchNorm1d(128)
        self.attn_6 = SelfAttention1d(128, nn.ReLU6)

        self.res7 = ConvResidualBlock1d(
            128, 128, kernel_size=3, stride=2, activation_layer=nn.ReLU6
        )
        self.out_size = (self.out_size - 1) // 2 + 1

        self.pool = nn.AvgPool1d(self.out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x, _ = self.attn_2(self.conv2(x))

        x = self.res3(x)
        x = self.res4(x)
        x, _ = self.attn_4(x)

        x = self.res5(x)
        x = self.res6(x)
        x, _ = self.attn_6(x)

        x = self.res7(x)
        x = self.pool(x).flatten(start_dim=1)

        return x


class Classifier(nn.Module):
    def __init__(self, in_dim: int = 128, n_class: int = 18) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.n_class = n_class

        self.activation = nn.ReLU6()
        self.classifier = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(self.in_dim, 256)
        self.linear2 = nn.Linear(256, 512, bias=False)
        self.norm_2 = nn.BatchNorm1d(512)

        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 64, bias=False)
        self.norm_4 = nn.BatchNorm1d(64)

        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.norm_2(x)

        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.norm_4(x)

        x = self.activation(self.linear5(x))
        x = self.linear6(x)

        x = self.classifier(x)

        return x
