import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    upper_limit,
    lower_limit,
    std,
    clamp,
    get_loaders,
    evaluate_pgd,
    evaluate_standard,
)
from src.pytorch.utils import get_model, load_data

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", type=str, choices=["cnn", "fcn"])
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--dataset", default="CIFAR10", type=str)
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Total number of epochs will be this argument * number of minibatch replays.",
    )
    parser.add_argument(
        "--lr-schedule", default="cyclic", type=str, choices=["cyclic", "multistep"]
    )
    parser.add_argument("--lr-min", default=0.0, type=float)
    parser.add_argument("--lr-max", default=0.04, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--minibatch-replays", default=8, type=int)
    parser.add_argument(
        "--out-dir", default="train_free_output", type=str, help="Output directory"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--opt-level",
        default="O2",
        type=str,
        choices=["O0", "O1", "O2"],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision',
    )
    parser.add_argument(
        "--loss-scale",
        default="1.0",
        type=str,
        choices=["1.0", "dynamic"],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time',
    )
    parser.add_argument(
        "--master-weights",
        action="store_true",
        help="Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level",
    )
    return parser.parse_args()


def main():
    args = get_args()

    logger.info("# " + str(args))
    model_name = "{}_{}_{}_{}".format(args.dataset, args.model, args.epsilon, "free")
    if not os.path.exists(os.getcwd() + "/src/pytorch/log/fast/CIFAR10/"):
        os.mkdir(os.getcwd() + "/src/pytorch/log/fast/CIFAR10/")
    logfile = os.getcwd() + "/src/pytorch/log/fast/CIFAR10/" + model_name + ".csv"

    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format=" %(message)s",
        level=logging.INFO,
        filename=logfile,
        force=True,
    )
    logger.info("# " + str(args))
    logger.info("Epoch, Train loss,Train acc, Learning Rate, Time, Time elapsed")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    data = load_data(args.dataset, args.batch_size)

    epsilon = (args.epsilon / 255.0) / std

    model = get_model(args.model)
    model = model(3, 10, (32, 32)).cuda()
    model.train()

    opt = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    delta.requires_grad = True

    lr_steps = args.epochs * len(data.train) * args.minibatch_replays
    if args.lr_schedule == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=args.lr_min,
            max_lr=args.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif args.lr_schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1
        )

    # Training
    total_time = 0
    for epoch in range(1, args.epochs + 1):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(data.train, leave=False)):
            X, y = X.cuda(), y.cuda()
            for _ in range(args.minibatch_replays):

                output = model(X + delta[: X.size(0)])
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(
                    delta + epsilon * torch.sign(grad), -epsilon, epsilon
                )
                delta.data[: X.size(0)] = clamp(
                    delta[: X.size(0)], lower_limit - X, upper_limit - X
                )

                opt.step()
                delta.grad.zero_()
                scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        time_epoch = epoch_time - start_epoch_time
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
        "src/pytorch/saves/fast/CIFAR10/{}_epoch={}.pt".format(
            model_name, args.epochs + 1
        ),
    )


if __name__ == "__main__":
    main()
