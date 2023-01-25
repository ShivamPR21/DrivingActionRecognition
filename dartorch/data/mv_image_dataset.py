import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiViewImageClsDataset(Dataset):
    def __init__(self, root: str, resize: Tuple[int.int] = (128, 128)) -> None:
        super().__init__()

        self.root = root
        self.resize = resize
        self.views = [0, 1, 2]
        self.cls = np.arange(0, 18, dtype=np.int32)

        self.view_dirs = [os.path.join(root, str(view)) for view in self.views]

        data = {cls: self.extract_cls_paths(cls, self.view_dirs) for cls in self.cls}

        self.data = []
        for cls, paths in data.items():
            self.data += [[cls, path] for path in paths]

    def extract_cls_paths(cls: int, view_dirs: List[str]) -> List[str]:

        cls_view_dirs = [os.path.join(view_dir, str(cls)) for view_dir in view_dirs]

        view1_files, view2_files, view3_files = [
            os.listdir(cls_view_dir) for cls_view_dir in cls_view_dirs
        ]

        common_list: List[str] = []

        for file in view1_files:
            if file in view2_files and file in view3_files:
                common_list.append(file)

        return common_list

    def read_img(self, path: str) -> np.ndarray:
        img = np.array(Image.open(path).resize(self.resize), dtype=np.float32) / 255.0
        return img

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cls, file_name = self.data[index]

        view_paths = [
            os.path.join(view_dir, str(cls), file_name) for view_dir in self.view_dirs
        ]
        images = tuple(
            torch.from_numpy(self.read_img(view_path)).unsqueeze(dim=0)
            for view_path in view_paths
        )

        return *images, cls

    def __len__(self) -> int:
        return len(self.data)
