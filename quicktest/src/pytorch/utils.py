import torch
import torchvision
from easydict import EasyDict
from typing import Callable


def get_model(model_name: str) -> torch.nn.Module:
    """Load the model from the models module by its name

    Args:
        model_name: The name of the model to fetch.
    Exemple:
        model = get_model("cnn") -> model()
        is the equivalent to
        import models.cnn as cnn -> cnn.cnn()
    """
    # __import__ method used
    # to fetch module
    module_models = __import__("models." + model_name)
    module_model = getattr(module_models, model_name)
    my_model = getattr(module_model, model_name)
    return my_model


def get_attack(attack_name: str) -> Callable:
    """Load the attack from cleverhans.torch.attacks by its name

    Args:
        attack_name: name of the attack to fetch.
    Exemple:
        get_atttack(fast_gradient_method)
        is the equivalent to
        from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
    """
    module = __import__("attacks." + attack_name)
    moduleAtk = getattr(module, attack_name)
    attack = getattr(moduleAtk, attack_name)
    return attack


def load_data(data_name: str, batch_size: int) -> EasyDict:
    """Load data from torchvision.datasets by its name.

    Args
        data_name: name of the dataset to fetch
    Exemple:
        data = load_data("MNIST)
        is the equivalent to
        import torchvision.datasets.MNIST as data
    """
    data = getattr(torchvision.datasets, data_name)
    root = "./data/" + data_name

    # Define the transformations to the data
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    # Load training and test data, from root (downloaded if needed)
    train_dataset = data(
        root=root, train=True, transform=train_transforms, download=True
    )
    test_dataset = data(
        root=root, train=False, transform=test_transforms, download=True
    )
    # Define the loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)
