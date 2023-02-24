import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)
from src.pytorch.utils import get_model, load_data

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'fcn'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--epochs', default=10, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.04, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='train_free_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()
    half = True
    if half:
        scaler = torch.cuda.amp.GradScaler()
    if not os.path.exists( os.getcwd()+"/src/pytorch/log/fast/" + args.out_dir):
        os.mkdir( os.getcwd()+"/src/pytorch/log/fast/" + args.out_dir)
    logfile = os.getcwd()+"/src/pytorch/log/fast/" + args.out_dir + "/output.log"
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    data = load_data(args.dataset, args.batch_size)

    epsilon = (args.epsilon / 255.) / std

    model = get_model(args.model)
    model = model(3,10,(32,32)).cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    delta.requires_grad = True

    lr_steps = args.epochs * len(data.train) * args.minibatch_replays
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(data.train,leave=False)):
            X, y = X.cuda(), y.cuda()
            for _ in range(args.minibatch_replays):
                if half:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(X + delta[:X.size(0)])
                    loss = criterion(output, y)
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                else:
                    output = model(X + delta[:X.size(0)])
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                
                if half:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                delta.grad.zero_()
                scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = get_model(args.model)
    model_test = model_test(3,10,(32,32)).cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(data.test, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(data.test, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
