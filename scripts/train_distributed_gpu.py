import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torchinfo import summary

from dartorch.models import DrivingActionClassifier
from dartorch.old_data_model import DARDatasetOnVideos


def compute_loss(dl, model, crt):
    total_loss = 0.0
    cnt = 0
    total = 0
    correct = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data, target in dl:
            data = data.to(device)
            target = target.to(device)

            # calculate outputs by running images through the network
            cls = model(data)

            loss = crt(cls, target)

            total_loss += loss.item()

            predictions = cls.max(dim=1)[1]
            total += predictions.size(0)
            correct += (predictions == target).sum().item()
            cnt += 1

    return [total_loss / cnt, correct / total]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Data root directory.")
    parser.add_argument("--batch_size", type=int, default=1, help="Model batch size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs the complete dataset to be iterated.",
    )
    parser.add_argument(
        "--n_itr_logs",
        type=int,
        default=1,
        help="Number of times logging the information in an epoch.",
    )
    parser.add_argument(
        "--n_epoch_logs",
        type=int,
        default=1,
        help="Number of times logging the information in all epochs.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        default=128,
        help="Model image size(assumed square in size).",
    )
    parser.add_argument(
        "--nproc_dl", type=int, default=5, help="Number of workers to load data."
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="If given the GPU accelerator will be used if available.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=os.path.join(os.getenv("HOME"), "AI_CITY"),
        help="Results directory.",
    )

    args = parser.parse_args()

    # Create the results directory
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    dataset = DARDatasetOnVideos(
        args.data_dir,
        test_user_ids=["user_id_49381"],
        img_size=(args.im_size, args.im_size),
        load_reader_instances=True,
        mode="train",
        shuffle_views=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nproc_dl
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model = DrivingActionClassifier(in_size=(args.im_size, args.im_size))
    model = DistributedDataParallel(model)
    print(summary(model, (args.batch_size, 15, args.im_size, args.im_size)))

    model.to(device)

    n_itrs = len(dataloader)
    n_logs = n_itrs // args.n_itr_logs
    n_epochs = args.epochs
    n_epoch_logs = n_epochs // args.n_epoch_logs

    # Define Loss function, Optimizer
    criterion = nn.CrossEntropyLoss()
    param_list = model.parameters()
    optimizer = optim.AdamW(param_list, lr=0.0007, weight_decay=0.0001)

    # Init Running and Total loss list
    rll, tll = [], []

    # Set model as trainable
    model.train()

    # Training loop
    for epoch in range(n_epochs):
        running_loss = 0.0
        total = 0.0
        correct = 0.0
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Zero the parameter gradient
            optimizer.zero_grad()

            cls = model(data)

            loss = criterion(cls, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = cls.max(dim=1)[1]
            total += predictions.size(0)
            correct += (predictions == target).sum().item()

            if i % n_logs == n_logs - 1:
                rll += [[epoch + 1, i + 1, running_loss / n_logs, correct / total]]
                print(
                    f"Epoch : {epoch}/{n_epochs}, Iteration : {i}/{n_itrs},Running loss : {running_loss/n_logs}, Train Accuracy : {correct/total}"
                )
                running_loss, total, correct = 0, 0, 0
        if epoch % n_epoch_logs == n_epoch_logs - 1:
            tll += [compute_loss(dataloader, model, criterion)]
            print(f"Total loss, and accuracy : {tll[-1]}")

        # Save model state
        model_state_path = os.path.join(result_dir, f"model_state_at_epoch_{epoch}")
        torch.save(model.state_dict(), model_state_path)

        # Save rll and tll
        rll_df_path = os.path.join(result_dir, f"rll_at_epoch_{epoch}.csv")
        rll_df = pd.DataFrame(
            rll, columns=["Epoch", "Iteration", "Running Loss", "Running Accuracy"]
        )
        rll_df.to_csv(rll_df_path, index=False)

        tll_df_path = os.path.join(result_dir, f"tll_at_epoch_{epoch}.csv")
        tll_df = pd.DataFrame(tll, columns=["Xnt Loss", "Accuracy"])
        tll_df.to_csv(tll_df_path, index=False)
