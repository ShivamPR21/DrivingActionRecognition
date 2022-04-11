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
from cProfile import label
from csv import reader
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tomlkit import value
from torch.nn.functional import interpolate
from torch.utils.data import Dataset


def _load_video_frames(reader:imageio.plugins.ffmpeg.FfmpegFormat.Reader, frame_no:int, n_frames:int = 1):
        reader.set_image_index(frame_no - n_frames)
        frame = [reader.get_next_data()[:, :, ::-1] for i in range(n_frames)]
        return frame

def _get_multi_view_frames(readers:List[imageio.plugins.ffmpeg.FfmpegFormat.Reader], frame_no:int, n_frames:int) -> List[List[np.ndarray]]:
        views = []
        for reader in readers:
            imgs:List[np.ndarray] = _load_video_frames(reader, frame_no, n_frames)
            for i in range(n_frames):
                imgs[i] = imgs[i].mean(axis=-1).astype(np.float32)
                imgs[i] /= 255.

            views += [imgs]
        return views
class DARDatasetOnVideos(Dataset):

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 img_size: Tuple[int, int] = (200, 200),
                 seq_length: int = 5,
                 remove_unlabeled: bool = True,
                 test_user_ids: List[str] = ["user_id_49381"],
                 load_reader_instances:bool = False) -> None:
        super().__init__()
        self.root = root # Training root directory
        self.transform = transform # Image transforms
        self.target_transform = target_transform # Target transforms
        self.img_size = img_size # Image size
        self.seq_length = seq_length # sequence length
        self.remove_unlabeled = remove_unlabeled # Whether to remvoe unlabeled data(-1 class_id)
        self.test_user_ids = test_user_ids # Removed from train set
        self.load_reader_instances = load_reader_instances # If true stores video reader instances, instead of path
        self.labels : Dict[str, Tuple[Dict[str, Any], np.ndarray]] = None # user_id/session_id : view : (video_path, labels)
        self.user_ids : List[str] = None
        self._n_labels : Dict[str, int] = None
        self.n : int = None
        self.views : List[str] = ['Dashboard', 'Rearview', 'Rightside window']

        self._datainit()

    def _datainit(self):
        self.user_ids = os.listdir(self.root) # Populate user IDs
        train_user_ids = []
        for user_id_ in self.user_ids:
            if user_id_ not in self.test_user_ids:
                train_user_ids += [user_id_]
        self.user_ids = train_user_ids
        self.labels = {}
        self._n_labels = {}
        self.n = 0
        for user in self.user_ids:
            id_dir = os.path.join(self.root, user)
            session_ids = []
            for dir_elem in os.listdir(id_dir):
                if dir_elem.startswith("session"):
                    session_ids += [dir_elem]
            for session_id_ in session_ids:
                label_df = pd.read_csv(os.path.join(os.path.join(id_dir, session_id_), 'labels.csv'), index_col=0)
                videos_df = pd.read_csv(os.path.join(os.path.join(id_dir, session_id_), 'vid_files.csv'), index_col=1)

                session_key = user + '/' + session_id_
                print(session_key)
                view_file_paths: Dict[str, str] = {}
                for view in self.views:
                    view_file_paths[view] = os.path.join(id_dir, videos_df.loc[view].values[0])
                    if self.load_reader_instances:
                        view_file_paths[view] = imageio.get_reader(view_file_paths[view], 'ffmpeg')

                if self.remove_unlabeled:
                    label_df = label_df[label_df.iloc[:, 2] != -1]

                self.labels[session_key] = (view_file_paths, np.array(label_df.values, dtype=np.int32))
                self._n_labels[session_key] = label_df.shape[0]
                self.n += label_df.shape[0]

    def reduce_idx(self, index: int) -> Tuple[str, int]:
        n_elements = index+1
        session_key, idx = None, None

        for key, value in self._n_labels.items():
            if n_elements <= value:
                idx = n_elements - 1
                session_key = key
                break
            n_elements -= value

        return session_key, idx

    def __getitem__(self, index: int) -> Any:
        session_key, idx = self.reduce_idx(index) # Find out relevant data directory, dataframe, and reduced index

        reader_dict, label = self.labels[session_key]
        readers = [val if self.load_reader_instances else imageio.get_reader(val, 'ffmpeg') for _, val in reader_dict.items()]

        frame_start, frame_end, target = label[idx]
        frame_no = np.random.randint(frame_start, frame_end+1, 1)[0]

        data = _get_multi_view_frames(readers, frame_no, self.seq_length)

        tensor_data = []
        for i in range(self.seq_length):
            view = []
            for j in range(3):
                view += [torch.from_numpy(data[j][i]).unsqueeze(dim=0)]
            tensor_data += [torch.cat(view, dim=0)]

        tensor_data = torch.cat(tensor_data, dim=0).unsqueeze(dim=0)
        tensor_data = interpolate(tensor_data, self.img_size).squeeze(dim=0)

        return tensor_data, target

    def __len__(self) -> int:
        assert(self.n != None)
        return self.n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    __HOME_DIR__ = os.getenv('HOME', '~')
    parser.add_argument('--root_dir', default=os.path.join(__HOME_DIR__, 'AI-City/A1'), help='Root directory path, for AI-City track3 dataset with folder A1, A2, and B')
    parser.add_argument('--load_reader_instances', action='store_true', help='If given, then video reader instances will be created andstored in advance.')
    args = parser.parse_args()

    dataset = DARDatasetOnVideos(args.root_dir, load_reader_instances=args.load_reader_instances)
    data = dataset.__getitem__(0)
    print(f'Dataset element size : {data.shape}')
