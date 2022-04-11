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

import argparse
import os
from statistics import mode
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DARDatasetOnImages(Dataset):

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 img_size : Tuple[int, int] = (200, 200),
                 seq_length: int = 5,
                 seq_type: str = 'trailing') -> None:
        super().__init__()
        assert(seq_type in ['trailing', 'forward', 'symmetric'])
        self.root = root # Training root directory
        self.transform = transform # Image transforms
        self.target_transform = target_transform # Target transforms
        self.img_size = img_size # Image size
        self.seq_length = seq_length # sequence length
        self.seq_type = seq_type # sequence type to sample
        self.label_dfs : List[pd.DataFrame] = None # List of Label dataset
        self.data_dir_paths : List[str] = None
        self.user_ids : List[str] = None
        self._n_frames : List[int] = None
        self._n_frames_up_to : List[int] = None
        self.n : int = None
        self.views : List[str] = ['Dashboard', 'Rearview', 'Rightside window']

    def datainit(self):
        self.user_ids = os.listdir(self.root) # Populate user IDs
        self.data_dir_paths = []
        self.label_dfs = []
        self._n_frames = []
        self.n = 0
        for user in self.user_ids:
            id_dir = os.path.join(self.root, user)
            session_ids = os.listdir(id_dir)
            for session_id_ in session_ids:
                self.data_dir_paths += [os.path.join(id_dir, session_id_)]
                dashboard_view_files = os.listdir(os.path.join(self.data_dir_paths[-1], 'Dashboard'))
                rearview_view_files = os.listdir(os.path.join(self.data_dir_paths[-1], 'Rearview'))
                rightside_view_files = os.listdir(os.path.join(self.data_dir_paths[-1], 'Rightside window'))
                label_df = pd.read_csv(os.path.join(self.data_dir_paths[-1], 'labels.csv'), index_col=0)
                drop_idx = []
                for i, frame_id in enumerate(label_df.loc[:, 'frame_idx'].values):
                    frame_name = str(frame_id)+'.jpg'
                    if not (frame_name in dashboard_view_files and \
                        frame_name in rearview_view_files and \
                        frame_name in rightside_view_files):
                            drop_idx += [i]
                label_df.drop(drop_idx, inplace=True)
                self._n_frames += [label_df.shape[0]]
                self.n += label_df.shape[0]
                self.label_dfs += [label_df]
        self._n_frames_up_to = np.cumsum(self._n_frames)

    def reduce_idx(self, index: int) -> Tuple[str, pd.DataFrame, int]:
        n_elements = index+1
        data_dir_path : str = None
        df : pd.DataFrame = None
        idx : int = None

        for i, n in enumerate(self._n_frames_up_to):
            if n > n_elements:
                data_dir_path = self.data_dir_paths[i]
                df = self.label_dfs[i]
                idx = n_elements-1 if i == 0 else n_elements-self._n_frames_up_to[i-1]-1
                break

        return data_dir_path, df, idx

    def get_multi_vuew_frame(self, file_paths:List[str]) -> torch.Tensor:
        views = []
        for file in file_paths:
            assert(file.endswith('.jpg'))
            img = Image.open(file, mode='L')
            img = img.resize(self.img_size)
            img = np.asarray(img, dtype=np.uint8).astype(np.float32)
            img /= 255.
            views += [img]
        views = torch.from_numpy(np.array(views, dtype=np.float32))
        return views

    def __getitem__(self, index: int) -> Any:
        data_dir_path, df, idx = self.reduce_idx(index) # Find out relevant data directory, dataframe, and reduced index

        # Get a patch of required seq length on both side of index
        left_idx, right_idx = None, None
        if self.seq_type == 'trailing':
            left_idx, right_idx = idx - self.seq_length, idx+1
        elif self.seq_type == 'forward':
            left_idx, right_idx = idx, idx + self.seq_length + 1
        elif self.seq_type == 'symmetric':
            left_idx, right_idx = idx - self.seq_length//2, idx + self.seq_length//2+1
        else:
            raise NotImplementedError

        data_sample = []
        left_deficit_frames, right_deficit_frames = 0, 0
        if left_idx < 0:
            left_deficit_frames = abs(left_idx)
            left_idx = 0

        if right_idx >= df.shape[0]:
            right_deficit_frames = right_idx - df.shape[0] - 1
            right_idx = df.shape[0]

        frame_labels = np.asarray(df.loc[left_idx:right_idx, 3].values, dtype=np.int8)
        frame_idxs = np.asarray(df.loc[left_idx:right_idx, 2].values, dtype=np.int32)

        for frame_id in frame_idxs:
            raise

    def __len__(self) -> int:
        assert(self.n != None)
        return self.n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    __HOME_DIR__ = os.getenv('HOME', '~')
    parser.add_argument('--root_dir', default=os.path.join(__HOME_DIR__, 'AI-City/A1'), help='Root directory path, for AI-City track3 dataset with folder A1, A2, and B')
    args = parser.parse_args()

    dataset = DARDatasetOnImages(args.root_dir)
    dataset.datainit()

    print(dataset.data_dir_paths)

    for df in dataset.label_dfs:
        print(df.shape)
