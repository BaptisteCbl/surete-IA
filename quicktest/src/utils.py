import torch
import torchvision
from easydict import EasyDict


def get_model(model_name):
    # __import__ method used
    # to fetch module
    module_models = __import__("models." + model_name)
    module_model = getattr(module_models, model_name)
    my_model = getattr(module_model, model_name)
    return my_model


def get_attack(attack_name):
    module = __import__("cleverhans.torch.attacks." + attack_name)
    module1 = getattr(module, "torch")
    module2 = getattr(module1, "attacks")
    module3 = getattr(module2, attack_name)
    attack = getattr(module3, attack_name)
    return attack


def load_data(data_name):
    data = getattr(torchvision.datasets, data_name)
    root = "./data/" + data_name

    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = data(
        root=root, train=True, transform=train_transforms, download=True
    )
    test_dataset = data(
        root=root, train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)
