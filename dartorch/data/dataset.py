'''
Copyright (C) 2022  Shivam Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class DARDataset(Dataset):

    def __init__(self, root: str,
                 user_id: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 img_size: Tuple[int, int] = (200, 200),
                 seq_length: int = 5) -> None:
        super().__init__()
        self.root = root
        self.user_id = user_id
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        self.seq_length = seq_length
        self.label_df : pd.DataFrame = None
        self.data : List[torch.Tensor] = []
        self.labels : List[int] = []

    def datainit(self):
        self.label_df = pd.read_csv()
