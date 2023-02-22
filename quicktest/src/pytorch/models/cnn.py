import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def output_dims(
    Hin: int = 28,
    Win: int = 28,
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    kernel_size: tuple[int, int] = (1, 1),
    stride: tuple[int, int] = (1, 1),
):
    Hout = math.floor(
        ((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0])
        + 1
    )
    Wout = math.floor(
        ((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1])
        + 1
    )

    return (Hout, Wout)


class cnn(nn.Module):
    """Basic CNN architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 10,
        dim: tuple[int, int] = (28, 28),
    ):
        super(cnn, self).__init__()
        self.dim = dim

        kernel_size = (8, 8)
        stride = (1, 1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=stride)
        self.dim = output_dims(
            self.dim[0], self.dim[1], kernel_size=kernel_size, stride=stride
        )

        kernel_size = (6, 6)
        stride = (2, 2)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.dim = output_dims(
            self.dim[0], self.dim[1], kernel_size=kernel_size, stride=stride
        )
        self.in_channels = in_channels

        kernel_size = (5, 5)
        stride = (2, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)

        self.dim = output_dims(
            self.dim[0], self.dim[1], kernel_size=kernel_size, stride=stride
        )
        self.fc = nn.Linear(128 * self.dim[0] * self.dim[1], out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * self.dim[0] * self.dim[1])
        x = self.fc(x)
        return x
