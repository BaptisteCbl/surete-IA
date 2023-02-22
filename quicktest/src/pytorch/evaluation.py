from absl import app, flags
import numpy as np
import torch
from tqdm import tqdm

from src.pytorch.utils import *

# from src.pytorch.attacks import *

FLAGS = flags.FLAGS


## All attacks are test in an untargeted setup (thus some None in parameters)
def parse_parameters(
    num_classes: int = 10,
    eps: float = 0.05,
    norm: int | float = np.inf,
    clip_min: int = 0,
    clip_max: int = 1,
    batch_size: int = 128,
    device: str = "cuda",
):
    """
    Given some parameters in arguments and the parameters in
    the config file (stored in FLAGS) create for all attacks a tuple of
    the attack parameters. The store them in a dictionnary.

    Args:
        num_classes: number of classes in the data set.
        eps: eps parameter for the attacks (FGSM, PGD).
        norm: norm used in the attacks, either 0,1,2,np.inf (not all are implemented
            for all attacks).
        clip_min: minimal clip value for the image.
        clip_max: max clip value for the image either 1/255 for float/int values.
        batch_size: size of the batches in the data loader.
        device: device to use for the attacks, either cpu or cuda.

    """
    dict_param = {}
    # For all attacks specified in the confile file
    for attack_name in FLAGS.attacks:
        match attack_name:
            case "fast_gradient_method":
                targeted, sanity_checks = FLAGS.FGSM_params
                dict_param[attack_name] = (
                    eps,
                    norm,
                    clip_min,
                    clip_max,
                    None,
                    eval(targeted),
                    eval(sanity_checks),
                )
            case "projected_gradient_descent":
                (
                    eps_iter,
                    nb_iter,
                    targeted,
                    rand_init,
                    rand_minmax,
                    sanity_checks,
                ) = FLAGS.PGD_params
                dict_param[attack_name] = (
                    eps,
                    eval(eps_iter),
                    eval(nb_iter),
                    norm,
                    clip_min,
                    clip_max,
                    None,
                    eval(targeted),
                    eval(rand_init),
                    eval(rand_minmax),
                    eval(sanity_checks),
                )
            case "carlini_wagner_l2":
                (
                    targeted,
                    lr,
                    confidence,
                    initial_const,
                    binary_search_steps,
                    max_iter,
                ) = FLAGS.CW_params
                dict_param[attack_name] = (
                    num_classes,
                    None,
                    eval(targeted),
                    eval(lr),
                    eval(confidence),
                    clip_min,
                    clip_max,
                    eval(initial_const),
                    eval(binary_search_steps),
                    eval(max_iter),
                )
            case "spsa":
                (
                    nb_iter,
                    targeted,
                    early_stop,
                    lr,
                    delta,
                    spsa_iters,
                    debug,
                    sanity_checks,
                ) = FLAGS.SPSA_params
                dict_param[attack_name] = (
                    eps,
                    eval(nb_iter),
                    norm,
                    clip_min,
                    clip_max,
                    None,
                    eval(targeted),
                    eval(early_stop),
                    eval(lr),
                    eval(delta),
                    batch_size,
                    eval(spsa_iters),
                    eval(debug),
                    eval(sanity_checks),
                )
            case "sparse_l1_descent":
                (
                    eps_iter,
                    iter,
                    targeted,
                    rand_init,
                    clip_grad,
                    grad_sparsity,
                    sanity_checks,
                ) = FLAGS.L1d_params
                dict_param[attack_name] = (
                    eps,
                    eval(eps_iter),
                    eval(iter),
                    None,
                    eval(targeted),
                    clip_min,
                    clip_max,
                    eval(rand_init),
                    eval(clip_grad),
                    eval(grad_sparsity),
                    eval(sanity_checks),
                )
            case "hop_skip_jump_attack":
                (
                    initial_num_evals,
                    max_num_evals,
                    stepsize_search,
                    iter,
                    gamma,
                    constraint,
                    verbose,
                ) = FLAGS.HSJA_params
                dict_param[attack_name] = (
                    None,
                    None,
                    eval(initial_num_evals),
                    eval(max_num_evals),
                    stepsize_search,
                    eval(iter),
                    eval(gamma),
                    eval(constraint),
                    batch_size,
                    eval(verbose),
                    clip_min,
                    clip_max,
                )
            case "deepfool":
                overshoot, iter = FLAGS.deepfool_params
                dict_param[attack_name] = [
                    num_classes,
                    eval(overshoot),
                    eval(iter),
                    device,
                ]
            case _:
                assert False, "Unsupported attack: " + attack_name
    return dict_param


def parse_attacks():
    """
    Create a dictionnary contening all attacks function using attack names
    in the config file
    """
    dict_attack = {}
    for attack_name in FLAGS.attacks:
        dict_attack[attack_name] = get_attack(attack_name)
    return dict_attack


def perform_attack(net, x, y: int, attacks: dict, parameters: dict) -> list:
    """
    Perform all adversarial attacks, from the attacks dictionnary using the parameters
    dictionnary, on the input sample x.

    Args:
        net: the model.
        x: the input sample (Batch_size * C * H * W)
        y: the true label of the sample
        attacks: dictionnary containing all the attacks' functions
        parameters: dictionnary containing all the attacks' parameters
    Return:
        y_pred_advs: list of the labels predicted by the model
            on the adversarial examples.
    """
    y_pred_advs = []

    for attack_name in FLAGS.attacks:
        attack = attacks[attack_name] # get the attack
        param = parameters[attack_name] # get the parameters
        x_adv = attack(net, x, *param)  # perform the attack
        _, y_pred_adv = net(x_adv).max(1) # compute the labels
        y_pred_advs.append(y_pred_adv.eq(y).sum().item())  # store the number of correct labels

    return y_pred_advs


def evaluation(net, data, device):
    """
    Compute and display the accuracy of the network on the clean samples
    and all adversarial examples

    Args:
        net: the model.
        data: easydict containing the training and test dataset.
        device: device to use, either cpu or cuda.
    """
    # Evaluate on clean and adversarial data
    net.eval()
    # Gather all attacks function in a dictionnary
    attacks = parse_attacks()
    # Define some basic parameters
    num_classes = FLAGS.num_classes
    norm = np.inf if FLAGS.norm == "inf" else int(FLAGS.norm)
    clip_min = FLAGS.clip_min
    clip_max = FLAGS.clip_max
    batch_size = 128
    for eps in list(map(float, FLAGS.eps)):
        print("For epsilon = ", eps)
        # Parse the paremters for each attack
        parameters = parse_parameters(
            num_classes, eps, norm, clip_min, clip_max, batch_size, device
        )
        # Define the dictionnary to log the accuracy
        report = dict(nb_test=0, base=0)
        for attack in FLAGS.attacks:
            report[attack] = 0

        # Evaluation loop
        for x, y in tqdm(data.test, leave=False):
            # Clean sample
            x, y = x.to(device), y.to(device)
            _, y_pred = net(x).max(1)  # model prediction on clean examples
            # Adversarial examples
            y_preds_adv = perform_attack(net, x, y, attacks, parameters)
            # Accuracy log
            report["nb_test"] += y.size(0)
            report["base"] += y_pred.eq(y).sum().item()
            for attack, y_pred_adv in zip(FLAGS.attacks, y_preds_adv):
                report[attack] += y_pred_adv

        # Display the accuracy
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
    """
    Display some flags value.
    """
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
    data = load_data(FLAGS.data, FLAGS.batchsize)
    # Load the saved model from the save path flag
    net = torch.load("./saves/" + FLAGS.save + ".pt")
    # Load the model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()

    print("+" + "-" * 40 + "+")
    evaluation(net, data, device)


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

    print("+" + "-" * 40 + "+")
    # Check device availability: GPU if available, else CPU
    print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Deterministic
    print("Seed set to 0")
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(0)
    app.run(main)
