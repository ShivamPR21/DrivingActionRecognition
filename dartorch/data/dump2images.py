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
from fractions import Fraction
from time import time
from typing import List

import ffmpeg
import imageio
import numpy as np
import pandas as pd
from PIL import Image


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

def dump_labels_to_session_id_format(root:str = "../../datasets/2022/",
                              target:str = "../../datasets/aicity/session_id_format"):

    labeled_dir = os.path.join(root, 'A1')
    target = os.path.join(target, 'A1')
    os.makedirs(target, exist_ok=True)

    # Dumping labeled data
    user_ids_a1 = os.listdir(labeled_dir) # Get the user_id dirs list
    # print(user_ids_a1)

    for id_ in user_ids_a1:
        # print(id_)
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
            session_id_dir = os.path.join(os.path.join(target, id_), session_id) # Session id directory path
            os.makedirs(session_id_dir, exist_ok=True) # create the directory

            session_new_labels:pd.DataFrame = None

            file = np.unique(files)[0]
            loc_labels = labels[labels.iloc[:, 1] == file] # Labels corresponding to the view and session_id
            loc_new_labels = []
            class_ids = np.array(loc_labels.iloc[:, 6], dtype=int)
            time_patches = np.array(loc_labels.iloc[:, [4, 5]], dtype = int)

            cam_views = np.unique(loc_labels.iloc[:, 2].values)
            assert(len(cam_views) == 1)


            view_target_folder_path = os.path.join(session_id_dir, str(cam_views[0]))
            os.makedirs(view_target_folder_path, exist_ok=True)

            # print(file)
            real_file_name = str(file)
            for f in os.listdir(id_dir):
                f = str(f).split('.')[0]
                if str(f).lower()[:-1].startswith(real_file_name.lower()[:-1]) and str(f).lower()[-1] == real_file_name.lower()[-1]:
                    real_file_name = str(f)

            # print(real_file_name)
            view_video_path = os.path.join(id_dir, real_file_name+'.MP4')
            # print(view_video_path)
            # Get video information
            probe = ffmpeg.probe(view_video_path) # probe the video
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video') # get video information

            # Get video information
            num_frames = int(video_info['nb_frames'])
            frame_rate = float(Fraction(video_info['avg_frame_rate']))

            frame_ids = []
            # print(time_patches)
            for time_patch, label in zip(time_patches, labels):
                frame_ids += [np.arange(int(time_patch[0]*frame_rate), int(time_patch[1]*frame_rate)+1)]

            # print(frame_ids)
            # Can be reduced to a single step
            for frame_ids_, class_id in zip(frame_ids, class_ids):
                # print(session_id, frame_ids_)
                for frame_id in frame_ids_:
                    if (frame_id >= num_frames or class_id == -1):
                        continue
                    loc_new_labels += [[session_id, frame_id, class_id]]

            session_new_labels = pd.DataFrame(loc_new_labels, columns=['session id', 'frame_idx', 'class_id'])
            session_labels_file_path = os.path.join(session_id_dir, 'labels.csv')
            # print(session_labels_file_path)
            session_new_labels.to_csv(session_labels_file_path)

def dump_unlabeled_data_to_session_id_format(root:str = "../../datasets/2022/",
                              target:str = "../../datasets/aicity/session_id_format",
                              sub_dir:str = 'A1'):
    assert(sub_dir in ['A1', 'A2', 'B'])

    unlabeled_dir = os.path.join(root, sub_dir) # Source unlabeled data path

    os.makedirs(target, exist_ok=True) # Create the target directory, escape if already exists

    # Dumping labeled data
    user_ids = os.listdir(unlabeled_dir) # Get the user_id dirs list

    # Process each user id
    for id_ in user_ids:
        id_dir = os.path.join(unlabeled_dir, id_) # Path to user_id directory

        vid_file_names:List[str] = os.listdir(id_dir) # Get the video file names
        vid_file_names_:List[str] = []
        for vid in vid_file_names:
            if not str(vid).endswith('.MP4'):
                continue

            vid_file_names_ += [str(vid).split('.')[0]]

        vid_file_names = np.array(vid_file_names_, dtype=str)

        vid = [x.split('_')[-1] for x in vid_file_names] # Get session ids
        uvid = np.unique(vid) # Get unique session ids

        # get file_names corresponding to each session_id
        # row repressents .MP4 file_names for each session_id
        vid_file_names = [vid_file_names[s_match_idx] for s_match_idx in np.array([uvid]).T == np.array([vid])]

        # Iterate through all the session ids and
        # Dump that session id into session_id/view/frame_id.jpg format
        for session_id, files in zip(uvid, vid_file_names):
            session_id_dir = os.path.join(os.path.join(os.path.join(target, sub_dir), id_), session_id) # Session id directory path
            os.makedirs(session_id_dir, exist_ok=True) # create the directory
            loc_new_labels = []

            session_new_labels:pd.DataFrame = None

            generate_session_label = True

            # For each view
            for file in np.unique(files):

                cam_view = None
                if (file.startswith('Dashboard')):
                    cam_view = 'Dashboard'
                elif (file.startswith('Rear')):
                    cam_view = 'Rearview'
                elif (file.startswith('Right')):
                    cam_view = 'Rightside window'
                else:
                    print(f'Camview can\'t be defined for session id : {session_id}, and file_name : {file}')

                view_target_folder_path = os.path.join(session_id_dir, cam_view)
                os.makedirs(view_target_folder_path, exist_ok=True)

                view_video_path = os.path.join(id_dir, str(file)+'.MP4')

                reader = imageio.get_reader(view_video_path)

                print(f'Writing video file : {view_video_path}, to target folder : {view_target_folder_path}')
                for frame_id, img in enumerate(reader):
                    frame_path = os.path.join(view_target_folder_path, str(frame_id)+'.jpg')

                    imageio.imwrite(frame_path, img)

                    if generate_session_label:
                        loc_new_labels += [[session_id, frame_id, -1]]

                if generate_session_label:
                    session_new_labels = pd.DataFrame(loc_new_labels, columns=['session id', 'frame_idx', 'class_id'])
                    session_labels_file_path = os.path.join(session_id_dir, 'submission.csv')
                    session_new_labels.to_csv(session_labels_file_path)
                    generate_session_label = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', default='', help='Root directory path, for AI-City track3 dataset with folder A1, A2, and B')
    parser.add_argument('--target_dir', default='/home/shivam/AI-City', help='Target directory path to dump the session id format data.')
    args = parser.parse_args()

    dump_labels_to_session_id_format(args.root_dir, args.target_dir)
    dump_unlabeled_data_to_session_id_format(args.root_dir, args.target_dir)

    print(f'Data successfully converted to session id format and stored in the folder {args.target_dir}')
