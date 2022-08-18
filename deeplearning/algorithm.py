"""
In this file, the basic function for training and evaluating `Classification Network` and `Siamese Network`
"""
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable as V

import nets as nets


def classification_train(
        net,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        device,
        epoch=1,
        batch_size=16,
        num_workers=1,
        saving_path=None,
        saving_prefix="checkpoint_",
        saving_frequency=1,
        gpu=False,
        lr_scheduler=None,
        # milestones=None
):
    """
    Training Classification network
    --------------------------------------------------
    Parameters:
        - net (nets.ClassificationNetwork)
            * Classification Netwrok

        - train_dataloader (dataloaders.ClassifierDataLoader)
            * Data loader for train set

        - val_dataloader (dataloaders.ClassifierDataLoader)
            * Data loader for validation set

        - optimizer (torch.optim)
            * Optimizer Algorithm

        - device (torch.device)
            * Device for training network

        - epoch (int)
            * Number of training epochs

        - batch_size (int)
            * Data loading batch size

        - num_workers (int)

        - lr_scheduler
            * Learning Rate Scheduler
    """

    train_dataloader = DataLoader(dataset=train_dataloader,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=gpu and torch.cuda.is_available(),
                                  num_workers=num_workers
                                  )

    report = pd.DataFrame(
        columns=["epoch", "train/eval", "batch_size", "loss", "acc", "correct", "lr"])

    net = net.float()

    for e in range(epoch):
        net.train()
        # if (milestones is not None) and (e in milestones):
        #     optimizer = nets.schedule_lr(optimizer)
        running_loss = 0
        running_count = 0
        running_correct = 0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for (X, Y) in tepoch:

                tepoch.set_description(f"Training @ Epoch {e}")

                X, Y = V(X), V(Y)

                if device != 'cpu' and gpu and torch.cuda.is_available():
                    if device.type == 'cuda':
                        X, Y = X.cuda(device=device), Y.cuda(device=device)
                    elif device == 'multi':
                        X, Y = nn.DataParallel(X), nn.DataParallel(Y)

                optimizer.zero_grad()

                out = net(X)

                loss = criterion(out, Y)
                loss.backward()
                optimizer.step()

                # print("**** X.device: {}".format(X.device))
                # print("**** Y.device: {}".format(Y.device))
                # print("**** out.device: {}".format(out.device))
                # print("**** net.device: {}".format(next(net.parameters()).device))

                pred = F.softmax(out, dim=-1).argmax(dim=-1)
                correct = (Y == pred).sum().item()
                # accuracy = (correct / Y.shape[0])*100

                # import pdb
                # pdb.set_trace()
                running_loss += loss.item()
                running_count += Y.shape[0]
                running_correct += correct
                current_lr = optimizer.param_groups[0]['lr']

                current_report = pd.DataFrame({
                    "epoch": [e],
                    "train/eval": ["train"],
                    "batch_size": [Y.shape[0]],
                    "loss": [loss.item()],
                    # "acc": [accuracy],
                    "correct": [correct],
                    "lr": [current_lr]})

                report = pd.concat([report, current_report])

                # accuracy = int(accuracy)

                tepoch.set_postfix(
                    loss="{:.3f}".format(running_loss/running_count),
                    accuracy="{:.3f}".format(running_correct/running_count),
                    lr="{}".format(current_lr))

        val_report, val_acc = classification_eval(
            net=net,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu=gpu,
            tqbar_description=f"Validation @ Epoch  {e}",
            epoch=e,
            mode="val"
        )

        report = pd.concat([report, val_report])

        if lr_scheduler is not None:
            lr_scheduler.step(val_acc)

        if e % saving_frequency == 0:
            nets.save(
                file_path=saving_path,
                file_name="{}_epoch_{}".format(saving_prefix, e),
                model=net,
                optimizer=optimizer
            )

    return net, report


def classification_eval(
    net,
    dataloader,
    criterion,
    device,
    batch_size=16,
    num_workers=1,
    gpu=False,
    tqbar_description="Test",
    epoch=None,
    mode="Test"
):
    """
    Evaluation Function
    """
    dataloader = DataLoader(dataset=dataloader,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=gpu and torch.cuda.is_available(),
                            num_workers=num_workers
                            )

    report = pd.DataFrame(
        columns=["epoch", "train/eval", "batch_size", "loss", "acc", "correct", "lr"])

    net = net.float()
    net.train(mode=False)

    running_loss = 0
    running_count = 0
    running_correct = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        with torch.no_grad():
            for ix, (X, Y) in enumerate(tepoch):
                tepoch.set_description(tqbar_description)

                X, Y = V(X), V(Y)

                if device != 'cpu' and gpu and torch.cuda.is_available():
                    if device.type == 'cuda':
                        X, Y = X.cuda(device=device), Y.cuda(device=device)
                    elif device == 'multi':
                        X, Y = nn.DataParallel(X), nn.DataParallel(Y)

                out = net(X)

                pred = F.softmax(out, dim=-1).argmax(dim=-1)

                correct = (Y == pred).sum().item()
                # accuracy = (correct / batch_size)*100

                loss = criterion(out, Y)

                running_loss += loss.item()
                running_count += Y.shape[0]
                running_correct += correct

                current_report = pd.DataFrame({
                    "epoch": [epoch],
                    "train/eval": [mode],
                    "batch_size": [Y.shape[0]],
                    "loss": [loss.item() * Y.shape[0]],
                    # "acc": [accuracy],
                    "correct": [correct],
                    "lr": [None]})

                report = pd.concat([report, current_report])

                # accuracy = int(accuracy)

                tepoch.set_postfix(
                    loss="{:.3f}".format(running_loss/running_count),
                    accuracy="{:.3f}".format(running_correct/running_count))

    test_acc = 100 * running_correct / len(dataloader)
    return report, test_acc
