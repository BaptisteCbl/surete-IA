from absl import app, flags
import numpy as np
import torch
from tqdm import tqdm

from src.utils import *

FLAGS = flags.FLAGS


def perform_attack(net, x, y, eps, device):
    y_pred_advs = []

    for attack_name in FLAGS.attacks:
        attack = get_attack(attack_name)
        batch = True
        match attack_name:
            case "fast_gradient_method":
                param = (eps, np.inf)
            case "projected_gradient_descent":
                param = (eps, 0.01, 40, np.inf)
            case "carlini_wagner_l2":
                param = (10, y)
            case "spsa":
                param = (eps, 40, np.inf)
            case "sparse_l1_descent":
                param = (10, 1, 20)
            case "hop_skip_jump_attack":
                param = (np.inf, None)
            case "deepfool":
                batch = False
                param = (10, 0.02, 5, device)
            case _:
                assert False, "Unsupported attack"
        if batch:
            x_adv = attack(net, x, *param)
        else:
            batch_size = x.shape[0]
            x_adv = torch.zeros((batch_size, 3, 32, 32)).to(device)
            for i in range(batch_size):
                x_adv[i, :, :, :] = attack(net, x[i, :, :, :], *param)
        _, y_pred_adv = net(x_adv).max(1)
        y_pred_advs.append(y_pred_adv.eq(y).sum().item())

    return y_pred_advs


def evaluation(net, data, device):
    # Evaluate on clean and adversarial data
    net.eval()
    for eps in list(map(float, FLAGS.eps)):
        print("For epsilon = ", eps)
        report = dict(nb_test=0, base=0)
        for attack in FLAGS.attacks:
            report[attack] = 0
        for x, y in tqdm(data.test, leave=False):
            x, y = x.to(device), y.to(device)

            _, y_pred = net(x).max(1)  # model prediction on clean examples
            y_preds_adv = perform_attack(net, x, y, eps, device)

            report["nb_test"] += y.size(0)
            report["base"] += y_pred.eq(y).sum().item()
            for attack, y_pred_adv in zip(FLAGS.attacks, y_preds_adv):
                report[attack] += y_pred_adv
        print(
            "test acc on clean examples (%): {:.3f}".format(
                report["base"] / report["nb_test"] * 100.0
            )
        )
        for attack in FLAGS.attacks:
            print(
                "test acc on {} adversarial examples (%): {:.3f}".format(
                    attack, report[attack] / report["nb_test"] * 100.0
                )
            )


def display_flag():
    print("+" + "-" * 40 + "+")
    print("Model:")
    print("Save", FLAGS.save)
    print("-" * 20)
    print("Data:")
    print("Data", FLAGS.data)
    print("-" * 20)
    print("Adversarial")
    print("Epsilon ", FLAGS.eps)
    print("+" + "-" * 40 + "+")


def main(_):
    display_flag()
    # Load training and test data
    data = load_data(FLAGS.data)
    # Load the saved model from the save path flag
    net = torch.load("./saves/" + FLAGS.save)
    # Load the model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()

    print("+" + "-" * 40 + "+")
    evaluation(net, data, device)


if __name__ == "__main__":
    flags.DEFINE_list("eps", [0], "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_string("data", "", "The dataset to load.")
    flags.DEFINE_string("save", "", "The path to save the model.")
    flags.DEFINE_list("attacks", [], "List of all attacks to perform")
    print("+" + "-" * 40 + "+")
    # Check device availability: GPU if available, else CPU
    print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Deterministic
    print("Seed set to 0")
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(0)
    app.run(main)
