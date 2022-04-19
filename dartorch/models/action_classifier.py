from typing import Tuple

import torch
import torch.nn as nn

from .mvconv import Classifier, MultiFrameFeatureMixer, MultiViewConvBackbone


class DrivingActionClassifier(nn.Module):

    def __init__(self,
                 n_views: int = 3,
                 n_channels: int = 1,
                 n_frames: int = 5,
                 in_size: Tuple[int, int] = (128, 128),
                 n_class: int = 18) -> None:
        super().__init__()

        self.multi_view_encoder = MultiViewConvBackbone(n_views, n_channels, n_frames, in_size)
        self.multi_frame_feature_mixer = MultiFrameFeatureMixer(n_frames, 512)
        self.classifier = Classifier(128, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.multi_view_encoder(x)
        x = self.multi_frame_feature_mixer(x)
        cls = self.classifier(x)

        return cls
