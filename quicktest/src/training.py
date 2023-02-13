from absl import app, flags
import numpy as np
import torch

from tqdm import tqdm

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from utils import *

FLAGS = flags.FLAGS


def training(net, data, device, optimizer, loss_fn):
    # Train vanilla model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in tqdm(data.train, leave=False):
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                # x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
                x = fast_gradient_method(net, x, FLAGS.eps, np.inf)
            # Optimization step
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Display loss/epoch
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )
        # Save the model
        torch.save(net, "./saves/" + FLAGS.save)


def display_flag():
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
    display_flag()
    # Load training and test data
    data = load_data(FLAGS.data)
    # Load the model from the string flag
    model = get_model(FLAGS.model)
    # Instantiate the model
    net = model(in_channels=FLAGS.in_channels)
    # Load the model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    # Instantiate the loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    # Instantiate the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    print("+" + "-" * 40 + "+")
    training(net, data, device, optimizer, loss_fn)


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 0, "Number of epochs.")
    flags.DEFINE_integer("in_channels", 0, "Number of input channels of the model.")
    flags.DEFINE_float("eps", 0, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", None, "Use adversarial training (on PGD adversarial examples)."
    )
    flags.DEFINE_string("data", "", "The dataset to load.")
    flags.DEFINE_string("model", "", "The model to load.")
    flags.DEFINE_string("save", "", "The path to save the model.")

    print("+" + "-" * 40 + "+")
    # Check device availability: GPU if available, else CPU
    print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Deterministic
    print("Seed set to 0")
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(0)
    app.run(main)