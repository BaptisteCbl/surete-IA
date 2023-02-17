import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils import *
import torch.nn.functional as F
from numpy import linalg as la

CIFAR10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def transform(x):
    # C*H*W to H*W*C
    x = x.cpu().detach().numpy().swapaxes(0, 1).swapaxes(1, 2)
    return np.squeeze(np.clip(x, 0, 1))


def visualize(
    x, x_adv, y, y_prob, y_adv, y_adv_prob, rgb: bool = False, cifar10: bool = False
):

    x = transform(x)
    x_adv = transform(x_adv)
    pert = x_adv - x

    vmax = max(np.max(x), np.max(x_adv))
    vmin = max(np.min(x), np.min(x_adv))

    figure, ax = plt.subplots(1, 3, figsize=(12, 7))
    if rgb:
        ax[0].imshow(x, interpolation="lanczos")
    else:
        ax[0].imshow(x, cmap="Greys", vmin=vmin, vmax=vmax)
    ax[0].set_title("Clean Example", fontsize=20)

    if rgb:
        ax[1].imshow(pert, cmap="Greys", vmin=np.min(pert), vmax=np.max(pert))
    else:
        ax[1].imshow(pert, cmap="Greys", vmin=vmin, vmax=vmax)
    ax[1].set_title("Perturbation", fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    if rgb:
        ax[2].imshow(x_adv, vmin=vmin, vmax=vmax, interpolation="lanczos")
    else:
        ax[2].imshow(x_adv, cmap="Greys", vmin=vmin, vmax=vmax)
    ax[2].set_title("Adversarial Example", fontsize=20)
    ax[0].axis("off")
    ax[2].axis("off")
    ax[0].text(1.1, 0.5, "+", size=15, ha="center", transform=ax[0].transAxes)

    if cifar10:
        text = "Prediction: {}\n Proba: {:.4f} \n Name {}".format(
            y, y_prob, CIFAR10_labels[y]
        )
    else:
        text = "Prediction: {}\n Proba: {:.4f}".format(y, y_prob)
    ax[0].text(
        0.5,
        -0.43,
        text,
        size=15,
        ha="center",
        transform=ax[0].transAxes,
    )
    pert_flat = pert.flatten()
    ax[1].text(
        0.5,
        -0.53,
        "Relatives Norms: \n $l_0$: {:.4f}  \n $l_1$ : {:.3f}\n $l_\infty$: {:.3f}".format(
            la.norm(pert_flat, 0) / pert_flat.shape[0],  # np.count_nonzero(pert_flat)
            la.norm(pert_flat, 1) / int(la.norm(pert_flat, 0)),
            la.norm(pert_flat, np.inf),
        ),
        size=15,
        ha="center",
        transform=ax[1].transAxes,
    )

    ax[1].text(1.1, 0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    if cifar10:
        text = "Prediction: {}\n Proba: {:.4f} \n Name {}".format(
            y_adv, y_adv_prob, CIFAR10_labels[y_adv]
        )
    else:
        text = "Prediction: {}\n Proba: {:.4f}".format(y_adv, y_adv_prob)
    ax[2].text(
        0.5,
        -0.43,
        text,
        size=15,
        ha="center",
        transform=ax[2].transAxes,
    )

    plt.show()


if __name__ == "__main__":
    data = load_data("CIFAR10")
    first_batch = next(iter(data.test))
    batch = first_batch[0]
    label = first_batch[1]

    model_name = "cnn"
    net = torch.load("./saves/" + model_name + ".pt")
    # Load the model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    net.eval()
    i = 1
    xi = batch[i, :, :, :].cuda()

    y_prob, y = F.softmax(net(xi), dim=1).max(1)
    y_prob = y_prob.cpu().detach().numpy()[0] * 100
    y = y.cpu().detach().numpy()[0]
    attack_name = "fast_gradient_method"
    eps = 0.09
    attack = get_attack(attack_name)

    # param = (eps, 0.005, 1000, np.inf)
    param = (10, torch.tensor(y).cuda())
    # param = (eps, np.inf)
    x_adv = attack(net, batch.cuda(), eps, np.inf)[i, :, :, :]
    y_adv_prob, y_adv = F.softmax(net(x_adv), dim=1).max(1)
    y_adv_prob = y_adv_prob.cpu().detach().numpy()[0] * 100
    y_adv = y_adv.cpu().detach().numpy()[0]

    visualize(xi, x_adv, y, y_prob, y_adv, y_adv_prob, rgb=True, cifar10=True)
