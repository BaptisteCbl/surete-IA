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

FashionMNIST_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
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
    grid = torch.zeros(10, dim[1], dim[2], dim[3], device="cuda")
    # Perform the attacks
    # all_eps = list(map(float, FLAGS.eps))
    all_eps = [i / 255 for i in range(0, 40)]
    for j in range(0, 10):
        # get all the parameters needed for the attacks
        num_classes = FLAGS.num_classes
        norm = np.inf if FLAGS.norm == "inf" else int(FLAGS.norm)
        clip_min = FLAGS.clip_min
        clip_max = FLAGS.clip_max
        batch_size = 1
        param = parse_parameters(
            num_classes, all_eps[j], norm, clip_min, clip_max, batch_size, device
        )

        x_advs, _, _ = perform_attack(
            net,
            xi,
            parse_attacks(),
            param,
        )
        grid[j, :, :, :] = x_advs[0]
    img = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(grid, nrow=2))
    img.show()


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
