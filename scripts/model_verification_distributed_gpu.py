import argparse
import builtins
import os

import torch
import torch.distributed as dist
from dartorch.data import DARDatasetOnVideos
from dartorch.models import DrivingActionClassifier
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchinfo import summary


def train_one_epoch(train_loader, model):
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        cls = model(data)
        print(f'Input shape : {data.shape}, Output shape : {cls.shape}')
        break
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Data root directory.')
    parser.add_argument('--batch_size', type=int, default=1, help='Model batch size.')
    parser.add_argument('--im_size', type=int, default=128, help='Model image size(assumed square in size).')
    parser.add_argument('--nproc_dl', type=int, default=5, help='Number of workers to load data.')

    # DDP configuration
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')

    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    print(f'Target decvice : {device}')

    # Define model
    model = DrivingActionClassifier(in_size=(args.im_size, args.im_size))
    # model_without_ddp = None
    # if args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         model = DistributedDataParallel(model, device_ids=[args.gpu])
    #         model_without_ddp = model.module
    #     else:
    #         model.cuda()
    #         model = DistributedDataParallel(model)
    #         model_without_ddp = model.module
    # else:
    #     raise NotImplementedError("Only DistributedDataParallel is supported.")

    # train_dataset = DARDatasetOnVideos(args.data_dir, test_user_ids = ["user_id_49381"],
    #                              img_size=(args.im_size, args.im_size),
    #                              load_reader_instances = True,
    #                              mode='train', shuffle_views=True)
    # train_sampler = DistributedSampler(train_dataset, shuffle=True)

    # train_loader = DataLoader(
    #         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #         num_workers=args.nproc_dl, pin_memory=True, sampler=train_sampler, drop_last=False)

    for epoch in range(2):
        if args.distributed:
            print(f'Training on epoch : {epoch}')
            # train_loader.sampler.set_epoch(epoch)
            # train_one_epoch(train_loader, model)

    print(f'Test completed.')
