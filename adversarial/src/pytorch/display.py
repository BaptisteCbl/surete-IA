"""
Script to display a clean image, the perturbation and the adversarial example.
The model to load, data to use, and attacks to perform can be modified in config_files/display.cfg

Example:
python src/pytorch/display.py --flagfile=config_files/display.cfg

@author: GuillaumeCld
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.pytorch.utils import *
import torch.nn.functional as F
from numpy import linalg as la
from src.pytorch.evaluation import parse_attacks, parse_parameters
from absl import app, flags
import os

FLAGS = flags.FLAGS

# The labels for the CIFAR10 dataset 10 (0_9) classes
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


def perform_attack(
    net,
    x,
    attacks: dict,
    parameters: dict,
) -> tuple[list, list, list]:
    """
    Perform attacks given attacks and there parameters on a specific image and returns
    the adversarial example, the output label and probability of the model on this example
    """
    x_advs = []
    y_pred_advs = []
    y_prob_advs = []

    for attack_name in FLAGS.attacks:
        # Perform the attack
        attack = attacks[attack_name]  # get the attack function
        param = parameters[attack_name]  # get the attack's parameters
        x_adv = attack(net, x, *param)  # generate the adversarial image (tensor)
        y_adv_prob, y_adv = F.softmax(net(x_adv), dim=1).max(
            1
        )  # get the label and its probability predicted by the model
        # Store the outputs
        x_advs.append(x_adv)
        y_pred_advs.append(y_adv)
        y_prob_advs.append(y_adv_prob)
    return x_advs, y_pred_advs, y_prob_advs


def toNumpy(x):
    """Transform a tensor object on gpu to a numpy object on cpu:

    x.cpu().detach().numpy()
    """
    return x.cpu().detach().numpy()


def transform(x):
    """Transform a tensor object on gpu to a numpy object on cpu + swap the axes in
    order to correctly use the pyplot.imshow method. Removes the first dimension of
    the tensor.

    B * C * H * W -> H * W * C
    """
    x = toNumpy(x)
    x = np.squeeze(x)
    # x = x.swapaxes(0, 1).swapaxes(1, 2)
    return np.clip(x, 0, 1)


def visualize(
    x,
    x_adv,
    y,
    y_prob,
    y_adv,
    y_adv_prob,
    attack: str,
    rgb: bool = False,
    cifar10: bool = False,
):
    x = transform(x)
    x_adv = transform(x_adv)
    pert = x_adv - x

    figure, ax = plt.subplots(1, 3, figsize=(12, 7))
    if rgb:
        ax[0].imshow(x, interpolation="lanczos")
    else:
        ax[0].imshow(x, cmap="Greys")
    ax[0].set_title("Clean Example", fontsize=20)

    if rgb:
        ax[1].imshow(np.clip(pert / la.norm(pert.flatten(), np.inf), 0, 1))
    else:
        ax[1].imshow(
            np.clip(pert / la.norm(pert.flatten(), np.inf), 0, 1), cmap="Greys"
        )

    ax[1].set_title("Perturbation", fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    if rgb:
        ax[2].imshow(x_adv, interpolation="lanczos")
    else:
        ax[2].imshow(x_adv, cmap="Greys")
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
        "Statistiques: \n Pixels modified: {:.4f} % \n Average perturbation : {:.3f}\n Maximum perturbation: {:.3f}".format(
            la.norm(pert_flat, 0)
            / pert_flat.shape[0]
            * 100,  # np.count_nonzero(pert_flat)
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
    figure.suptitle("Adversarial example for the {} attack".format(attack))
    plt.show()


def main(_):
    # Load the data and retrieve the first batch
    data = load_data(FLAGS.data, FLAGS.batchsize)
    first_batch = next(iter(data.test))
    batch = first_batch[0]
    label = first_batch[1]
    # Load the model
    net = torch.load(os.getcwd() + "/src/pytorch/saves/" + FLAGS.save + ".pt")
    # Load the model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    net.eval()
    # Select the image on which to perform the attacks
    i = 1
    dim = batch.shape
    xi = torch.zeros(1, dim[1], dim[2], dim[3]).cuda()
    xi[0, :, :, :] = batch[i, :, :, :]

    # Get the label and its probability of the
    # network output for xi
    y_prob, y = F.softmax(net(xi), dim=1).max(1)
    y_prob = toNumpy(y_prob)[0] * 100
    y = toNumpy(y)[0]

    # Perform the attacks
    for eps in list(map(float, FLAGS.eps)):
        # get all the parameters needed for the attacks
        num_classes = FLAGS.num_classes
        norm = np.inf if FLAGS.norm == "inf" else int(FLAGS.norm)
        clip_min = FLAGS.clip_min
        clip_max = FLAGS.clip_max
        batch_size = 1
        param = parse_parameters(
            num_classes, eps, norm, clip_min, clip_max, batch_size, device
        )

        x_advs, y_pred_advs, y_prob_advs = perform_attack(
            net,
            xi,
            parse_attacks(),
            param,
        )

        for i, attack in enumerate(FLAGS.attacks):
            x_adv = x_advs[i]
            y_adv = toNumpy(y_pred_advs[i])[0]
            y_adv_prob = toNumpy(y_prob_advs[i])[0] * 100
            visualize(
                xi,
                x_adv,
                y,
                y_prob,
                y_adv,
                y_adv_prob,
                attack,
                rgb=FLAGS.data != "MNIST",
                cifar10=FLAGS.data == "CIFAR10",
            )


if __name__ == "__main__":
    flags.DEFINE_string("data", "", "The dataset to load.")
    flags.DEFINE_integer("batchsize", 0, "The batch size for the loader.")

    flags.DEFINE_string("save", "", "The path to save the model.")
    flags.DEFINE_list("attacks", [], "List of all attacks to perform")
    flags.DEFINE_integer("num_classes", 10, "Number of classes in the dataset")

    # Attacks parameters flags
    flags.DEFINE_list("eps", [], "Total epsilon for the attacks.")
    flags.DEFINE_string("norm", "", "Norm for the attacks")
    flags.DEFINE_integer("clip_min", 0, "Clip min.")
    flags.DEFINE_integer("clip_max", 1, "Clip max.")

    flags.DEFINE_list(
        "FGSM_params", [], "Parameters for the fast_gradient_method attack"
    )
    flags.DEFINE_list(
        "PGD_params", [], "Parameters for the projected_gradient_descent attack"
    )
    flags.DEFINE_list("CW_params", [], "Parameters for the carlini_wagner_l2 attack")
    flags.DEFINE_list("SPSA_params", [], "Parameters for the spsa attack")
    flags.DEFINE_list("L1d_params", [], "Parameters for the sparse_l1_descent attack")
    flags.DEFINE_list("HSJA_params", [], "Parameters for the  hop_skip_jump attack")
    flags.DEFINE_list("deepfool_params", [], "Parameters for the deepfool attack")

    app.run(main)
