"""***************************************************************************************
* Code taken from
*    Title: understanding-fast-adv-training 
*    Date: 27/02/2023
*    Availability: https://github.com/tml-epfl/understanding-fast-adv-training/data.py
*
***************************************************************************************"""


import torch
from torchvision import datasets, transforms


def get_loaders(dataset, n_ex, batch_size, train_set, shuffle, data_augm):
    dir_ = "./data/" + dataset
    dataset_f = datasets_dict[dataset]
    num_workers = 6
    data_augm_transforms = [transforms.RandomCrop(32, padding=4)]
    if dataset not in ["MNIST", "FashionMNIST", "svhn"]:
        data_augm_transforms.append(transforms.RandomHorizontalFlip())
    transform_list = data_augm_transforms if data_augm else []
    transform = transforms.Compose(transform_list + [transforms.ToTensor()])

    if train_set:
        data = dataset_f(dir_, train=True, transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex

        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]

        loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        data = dataset_f(dir_, train=False, transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]
        loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=False,
            num_workers=2,
            drop_last=False,
        )
    return loader


datasets_dict = {
    "MNIST": datasets.MNIST,
    "FashionMNIST": datasets.FashionMNIST,
    "CIFAR10": datasets.CIFAR10,
}
shapes_dict = {
    "MNIST": (60000, 1, 28, 28),
    "FashionMNIST": (60000, 1, 28, 28),
    "svhn": (73257, 3, 32, 32),
    "CIFAR10": (50000, 3, 32, 32),
    "cifar10_binary": (10000, 3, 32, 32),
    "cifar10_binary_gs": (10000, 1, 32, 32),
    "uniform_noise": (1000, 1, 28, 28),
}
classes_dict = {
    "cifar10": {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
}
