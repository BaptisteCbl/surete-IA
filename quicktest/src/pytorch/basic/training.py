from absl import app, flags
import numpy as np
import torch
import logging
import os
from tqdm import tqdm
import time

from src.pytorch.attacks.fast_gradient_method import fast_gradient_method
from src.pytorch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from src.pytorch.utils import *


FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)


def training(net, data, device, optimizer, loss_fn) -> None:
    half = False
    if half:
        scaler = torch.cuda.amp.GradScaler()

    # optimize = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    # optimizer, base_lr=0.01, max_lr=0.1, cycle_momentum=False
    # )
    # Train vanilla model
    net.train()
    # Training loop
    start_time = time.time()
    total_time = 0
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss, train_acc, tot = 0.0, 0.0, 0
        for x, y in tqdm(data.train, leave=False):
            x, y = x.to(device), y.to(device)
            # TODO flag to choose the attack
            # Replace clean example with adversarial example for adversarial training
            if FLAGS.adv_train:
                # x = fast_gradient_method(net, x, FLAGS.eps, np.inf)
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)

            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            if half:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = net(x)
                    _, predicted = torch.max(output.data, 1)
                    loss = loss_fn(output, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = net(x)
                _, predicted = torch.max(output.data, 1)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
            # endif
            # scheduler.step()
            tot += y.shape[0]
            train_acc += predicted.eq(y).sum().item()
            train_loss += loss.item()
            time_epoch = time.time() - start_time

        total_time += time_epoch
        ### Epoch end
        # Store the loss and accuracy
        logger.info(
            "{},{:.4f},{:.4f},{:.2f},{:.2f}".format(
                epoch,
                train_loss,
                train_acc / tot * 100,
                time_epoch ,
                total_time ,
            )
        )
        # Display loss/epoch
        print(
            "epoch: {}/{},\t train loss: {:.3f}, \t accuracy: {:.4f}%".format(
                epoch, FLAGS.nb_epochs, train_loss, train_acc / tot * 100
            )
        )
    # Save the model (only for the last epoch)
    torch.save(net, "src/pytorch/saves/basic/" + FLAGS.save + ".pt")


def display_flag():
    """
    Display some flags value.
    """
    print("+" + "-" * 40 + "+")
    print("Model:")
    print("Model ", FLAGS.model)
    print("Save", FLAGS.save)
    print("-" * 20)
    print("Data:")
    print("Data", FLAGS.data)
    print("-" * 20)
    print("Hyperparameters:")
    print("Number epochs ", FLAGS.nb_epochs)
    print("In channels ", FLAGS.in_channels)
    print("Adversarial")
    print("Epsilon ", FLAGS.eps)
    print("Adv_train ", FLAGS.adv_train)
    print("+" + "-" * 40 + "+")


def main(_):
    attack = "fgsm"
    model_name = "{}_{}_{}_{}".format(
        FLAGS.data,
        FLAGS.model,
        FLAGS.eps,
        attack,
    )
    if not os.path.exists(os.getcwd() + "/src/pytorch/log/basic/"):
        os.mkdir(os.getcwd() + "/src/pytorch/log/basic/")
    logfile = os.getcwd() + "/src/pytorch/log/basic/" + model_name + ".csv"

    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format=" %(message)s",
        level=logging.INFO,
        filename=logfile,
        force=True,
    )
    logger.info(
        "# model:{} save: {} dataset:{} epoch:{} attack:{} eps:{}".format(
            FLAGS.model,
            FLAGS.save,
            FLAGS.data,
            FLAGS.nb_epochs,
            attack,
            FLAGS.eps,
        )
    )
    logger.info("Epoch, Train loss,Train acc, Time, Time elapsed")
    display_flag()
    # Load training and test data
    data = load_data(FLAGS.data, FLAGS.batchsize)
    # Load the model from the string flag
    model = get_model(FLAGS.model)
    # Instantiate the model
    net = model(
        in_channels=FLAGS.in_channels,
        out_channels=FLAGS.out_channels,
        dim=(int(FLAGS.dim[0]), int(FLAGS.dim[0])),
    )
    # Load the model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    # Instantiate the loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    # Instantiate the optimizer
    optimizer = torch.optim.AdamW(net.parameters())
    print("+" + "-" * 40 + "+")
    st = time.time()
    training(net, data, device, optimizer, loss_fn)
    et = time.time()
    print("Training time: ", et - st)


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 10, "Number of epochs.")
    flags.DEFINE_integer("in_channels", 1, "Number of input channels of the model.")
    flags.DEFINE_integer("out_channels", 10, "Number of output channels of the model.")
    flags.DEFINE_list("dim", [28, 28], "Dimension of the input H,W")
    flags.DEFINE_float("eps", 0.05, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    flags.DEFINE_string("data", "MNIST", "The dataset to load.")
    flags.DEFINE_integer("batchsize", 512, "The batch size for the loader.")

    flags.DEFINE_string("model", "cnn", "The model to load.")
    flags.DEFINE_string("save", "tmp", "The path to save the model.")

    print("+" + "-" * 40 + "+")
    # Check device availability: GPU if available, else CPU
    print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Deterministic
    print("Seed set to 0")
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    app.run(main)
