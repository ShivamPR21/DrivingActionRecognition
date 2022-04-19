import argparse
import os
from random import sample

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from dartorch.data import DARDatasetOnVideos
from dartorch.models import DrivingActionClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Data root directory.')
    parser.add_argument('--batch_size', type=int, default=1, help='Model batch size.')
    parser.add_argument('--im_size', type=int, default=128, help='Model image size(assumed square in size).')
    parser.add_argument('--nproc_dl', type=int, default=5, help='Number of workers to load data.')
    parser.add_argument('--cuda', action='store_true', help='If given the GPU accelerator will be used if available.')

    args = parser.parse_args()

    dataset = DARDatasetOnVideos(args.data_dir, test_user_ids = ["user_id_49381"],
                                 img_size=(args.im_size, args.im_size),
                                 load_reader_instances = True,
                                 mode='train', shuffle_views=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nproc_dl)

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    print(f'Target decvice : {device}')

    model = DrivingActionClassifier(in_size=(args.im_size, args.im_size))
    print(summary(model, (args.batch_size, 15, args.im_size, args.im_size)))

    model.to(device)
    model.eval()

    sample_data, target = iter(dataloader).next()
    sample_data = sample_data.to(device)

    out = model(sample_data)

    print(f'Input shape : {sample_data.shape}, Output shape : {out.shape}')
