"""***************************************************************************************
* Code taken and ajusted (removed nvidia apex amp dependencies + custom load model/data, logging)
 from
*    Title: fast_adversarial
*    Date: 27/02/2023
*    Availability: https://github.com/locuslab/fast_adversarial
*
* Example:
* python train_mnist.py --fname models/pgd_madry.pth --attack pgd --alpha 0.01 --lr-type flat --lr-max 0.0001 --epochs 100 --batch-size 50
***************************************************************************************"""


import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pytorch.utils import get_model, load_data
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_set", default="MNIST", type=str, choices=["MNIST", "FashionMNIST"]
    )
    parser.add_argument("--model", default="cnn", type=str, choices=["cnn", "fcn"])
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--data-dir", default="../mnist-data", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument(
        "--attack", default="none", type=str, choices=["none", "pgd", "fgsm"]
    )
    parser.add_argument(
        "--dataset", default="MNIST", type=str, choices=["MNIST", "FashionMNIST"]
    )
    parser.add_argument("--epsilon", default=0.3, type=float)
    parser.add_argument("--alpha", default=0.375, type=float)
    parser.add_argument("--attack-iters", default=40, type=int)
    parser.add_argument("--lr-max", default=5e-3, type=float)
    parser.add_argument("--lr-type", default="cyclic")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    logger.info("# " + str(args))
    model_name = "{}_{}_{}_{}".format(
        args.dataset,
        args.model,
        args.epsilon,
        args.attack,
    )
    if not os.path.exists(os.getcwd() + "/src/pytorch/log/fast/MNIST/"):
        os.mkdir(os.getcwd() + "/src/pytorch/log/fast/MNIST/")
    logfile = os.getcwd() + "/src/pytorch/log/fast/MNIST/" + model_name + ".csv"

    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format=" %(message)s",
        level=logging.INFO,
        filename=logfile,
        force=True,
    )
    logger.info("Epoch, Train loss,Train acc, Learning Rate, Time, Time elapsed")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data = load_data(args.data_set, args.batch_size)
    model = get_model(args.model)
    model = model(1, 10, (28, 28)).cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == "cyclic":
        lr_schedule = lambda t: np.interp(
            [t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0]
        )[0]
    elif args.lr_type == "flat":
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError("Unknown lr_type")

    criterion = nn.CrossEntropyLoss()
    total_time = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(data.train, leave=False)):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i + 1) / len(data.train))
            opt.param_groups[0].update(lr=lr)

            if args.attack == "fgsm":
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda()
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(
                    delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon
                )
                delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                delta = delta.detach()
            elif args.attack == "none":
                delta = torch.zeros_like(X)
            elif args.attack == "pgd":
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                for _ in range(args.attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(
                        delta + args.alpha * torch.sign(grad),
                        -args.epsilon,
                        args.epsilon,
                    )[I]
                    delta.data[I] = torch.max(torch.min(1 - X, delta.data), 0 - X)[I]
                delta = delta.detach()
            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()
        time_epoch = train_time - start_time
        total_time += time_epoch
        logger.info(
            "{}, {:.4f},{:.4f},{:.4f},{:.2f},{:.2f}".format(
                epoch,
                train_loss / train_n,
                train_acc / train_n,
                lr,
                time_epoch,
                total_time,
            )
        )
    torch.save(
        model,
        "src/pytorch/saves/fast/MNIST/{}_epoch={}.pt".format(
            model_name, args.epochs + 1
        ),
    )


if __name__ == "__main__":
    main()
