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
from typing import List

import ffmpeg
import imageio
import numpy as np
import pandas as pd


def read_frame_as_raw(in_filename, frame_num):
    out, err = (
        ffmpeg
        .input(in_filename)
        .filter('select', 'gte(n,{})'.format(frame_num))
        .output('pipe:', vframes=1, format='rawvideo', pix_fmt='gray')
        .run(capture_stdout=True)
    )
    return out

def preprocess_csv(x):
    x[1] = str(x[1]).strip()
    x[1] = '' if x[1] == 'nan' else x[1]

    st_split, et_split = str(x[4]).split(':'), str(x[5]).split(':')
    x[4] = 0
    for s in range(len(st_split)):
        idx = len(st_split) - 1 - s
        x[4] += int(st_split[idx])*(60**(s))
        if x[4] > 5000:
            print(st_split)
            raise

    x[5] = 0
    for s in range(len(et_split)):
        idx = len(et_split) - 1 - s
        x[5] += int(et_split[idx])*(60**(s))
        if x[5] > 5000:
            print(et_split)
            raise

    try:
        x[6] = int(x[6])
    except:
        x[6] = np.nan

    return x

def ffill(df : pd.Series, dtype = str):
    vals:List[str] = df.values
    last_valid = np.nan

    for i, val in enumerate(vals):
        if val == '':
            vals[i] = last_valid
        else:
            last_valid = val

    new_df = pd.Series(vals, dtype=dtype)
    # print(new_df)
    return new_df

def get_actual_file_name(files : List[str], id_dir : str):
    real_file_names = []
    for file in files:
        real_file_name = file
        for f in os.listdir(id_dir):
            f = f.split('.')[0]
            if f.lower()[:-1].startswith(real_file_name.lower()[:4]) and f.lower()[-1] == real_file_name.lower()[-1]:
                real_file_name = f
                break
        real_file_names += [real_file_name+'.MP4']

    return real_file_names

def get_cam_views_from_files(files : List[str]) -> List[str]:
    cam_views = []
    for file in files:
        cam_view = None
        if (file.startswith('Dashboard')):
            cam_view = 'Dashboard'
        elif (file.startswith('Rear')):
            cam_view = 'Rearview'
        elif (file.startswith('Right')):
            cam_view = 'Rightside window'
        else:
            raise

        cam_views += [cam_view]

    return cam_views

def dump_labels_to_session_id_format(root:str,
                              target:str):

    labeled_dir = os.path.join(root, 'A1')
    target = os.path.join(target, 'A1')
    os.makedirs(target, exist_ok=True)

    # Dumping labeled data
    user_ids_a1 = os.listdir(labeled_dir) # Get the user_id dirs list

    for id_ in user_ids_a1:
        id_dir = os.path.join(labeled_dir, id_) # Path to user_id directory

        csv_label_id_ = None
        for d in os.listdir(id_dir):
            d = str(d)
            if d.lower().endswith('.csv') and d.lower().startswith('user'):
                csv_label_id_ = str(d)

        label_file = os.path.join(id_dir, csv_label_id_) # labels file path
        # print(label_file)
        label = pd.read_csv(label_file, sep=',', index_col=None, dtype=str) # Read the labels file
        label.drop(label.filter(regex="Unnamed"),axis=1, inplace=True)
        # print(label)
        labels = label.apply(preprocess_csv, axis=1) # Preprocess csv for NaN removal, and time processing
        labels.iloc[:, 1] = ffill(labels.iloc[:, 1], dtype = str)
        # print(labels)
        labels.to_csv(os.path.join(id_dir, 'labels.csv'))
        labels.fillna(-1, inplace=True) # Drop non labeled data

        vid_file_names = np.array(labels.iloc[:, 1].values, dtype=str) # Get the video file names
        vid = [x.split('_')[-1] for x in vid_file_names] # Get session ids
        uvid = np.unique(vid) # Get unique session ids

        # get file_names corresponding to each session_id
        # row repressents .MP4 file_names for each session_id
        vid_file_names = [vid_file_names[s_match_idx] for s_match_idx in np.array([uvid]).T == np.array([vid])]

        # Iterate through all the session ids and
        # Dump that session id into session_id/view/frame_id.jpg format
        for session_id, files in zip(uvid, vid_file_names):
            session_id = 'session_'+session_id

            session_id_dir = os.path.join(os.path.join(target, id_), session_id) # Session id directory path
            os.makedirs(session_id_dir, exist_ok=True) # create the directory

            session_new_labels:pd.DataFrame = None

            files = [str(f) for f in np.unique(files)]

            file = files[0]
            loc_labels = labels[labels.iloc[:, 1] == file] # Labels corresponding to the view and session_id
            loc_new_labels = []
            class_ids = np.array(loc_labels.iloc[:, 6], dtype=int)
            time_patches = np.array(loc_labels.iloc[:, [4, 5]], dtype = int)

            real_file_names = get_actual_file_name(files, id_dir)
            cam_views = get_cam_views_from_files(real_file_names)
            real_file_names_df = pd.DataFrame(np.array([real_file_names, cam_views]).T, columns=['Filename', 'Camview'])
            real_file_names_df.to_csv(os.path.join(session_id_dir, 'vid_files.csv'), index=False)

            view_video_path = os.path.join(id_dir, real_file_names[0])

            # Get video information
            reader = imageio.get_reader(view_video_path, 'ffmpeg') # probe the video

            # Get video information
            num_frames = reader.get_meta_data()['fps']*reader.get_meta_data()['duration']
            frame_rate = float(reader.get_meta_data()['fps'])

            prev_frame = 0
            for time_patch, label in zip(time_patches, class_ids):
                init_frame, end_frame = int(time_patch[0]*frame_rate), int(time_patch[1]*frame_rate)
                init_frame = max(init_frame, 1)
                if init_frame != prev_frame+1:
                    loc_new_labels += [[min(prev_frame+1, num_frames), min(init_frame-1, num_frames), -1]]

                loc_new_labels += [[min(init_frame, num_frames), min(end_frame, num_frames), label]]
                prev_frame = end_frame

            session_new_labels = pd.DataFrame(loc_new_labels, columns=['frame_idx_start', 'frame_idx_end', 'class_id'])
            session_labels_file_path = os.path.join(session_id_dir, 'labels.csv')
            session_new_labels.to_csv(session_labels_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    __HOME_DIR__ = os.getenv('HOME', '~')
    parser.add_argument('root_dir', default='', type=str, help='Root directory path, for AI-City track3 dataset with folder A1, A2, and B')
    parser.add_argument('--target_dir', default=os.path.join(__HOME_DIR__, 'AI-City'), type=str, help='Target directory path to dump the session id format data.')
    args = parser.parse_args()

    dump_labels_to_session_id_format(args.root_dir, args.target_dir)

    print(f'Data successfully converted to session id format and stored in the folder {args.target_dir}')
