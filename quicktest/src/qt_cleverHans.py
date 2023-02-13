from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.carlini_wagner_l2 import (
    carlini_wagner_l2,
)
from cleverhans.torch.attacks.spsa import (
    spsa,
)

FLAGS = flags.FLAGS


import models.cnn as cnn


def ld_cifar10():
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, transform=train_transforms, download=True
    )
    data = torchvision.datasets.MNIST
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


def main(_):
    print("+" + "-" * 40 + "+")
    print("nb_epochs", FLAGS.nb_epochs)
    print("eps", FLAGS.eps)
    print("adv_train", FLAGS.adv_train)
    print("+" + "-" * 40 + "+")
    # Load training and test data
    data = ld_cifar10()

    # Instantiate model, loss, and optimizer for training
    net = cnn.cnn(in_channels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    print("+" + "-" * 40 + "+")
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in tqdm(data.train, leave=False):
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                # x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
                x = fast_gradient_method(net, x, FLAGS.eps, np.inf)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )
        torch.save(net, "./saves/qt_cnn.pt")

    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
        x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
        _, y_pred = net(x).max(1)  # model prediction on clean examples
        _, y_pred_fgm = net(x_fgm).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd = net(x_pgd).max(
            1
        )  # model prediction on PGD adversarial examples
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report.correct_fgm / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report.correct_pgd / report.nb_test * 100.0
        )
    )
    # Save the last batch and their adversarial examples
    # with their labels
    torch.save(x, "data/x.pt")
    torch.save(x_fgm, "data/x_fgm.pt")
    torch.save(x_pgd, "data/x_pgd.pt")
    torch.save(y_pred, "data/y_pred.pt")
    torch.save(y_pred_fgm, "data/y_pred_fgm.pt")
    torch.save(y_pred_pgd, "data/y_pred_pgd.pt")


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 0, "Number of epochs.")
    flags.DEFINE_float("eps", 0, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", None, "Use adversarial training (on PGD adversarial examples)."
    )
    print("+" + "-" * 40 + "+")
    # Check device availability: GPU if available, else CPU
    print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Deterministic
    print("Seed set to 0")
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(0)
    app.run(main)