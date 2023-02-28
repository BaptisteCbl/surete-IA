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


class cnn_small(nn.Module):
    """Basic CNN architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 10,
        dim: tuple[int, int] = (28, 28),
    ):
        super(cnn_small, self).__init__()
        self.dim = dim
        classes = out_channels

        #### First layer
        ## Paremeters
        kernel_size = (4, 4)
        stride = (2, 2)
        padding = (1, 1)
        out_channels = 16
        ## Convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Output dimension
        self.dim = output_dims(
            self.dim[0],
            self.dim[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        #### Second layer
        ## Parameters
        kernel_size = (4, 4)
        stride = (2, 2)
        in_channels = out_channels
        out_channels = 32
        ## Convolution
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        ## Output dimension
        self.dim = output_dims(
            self.dim[0],
            self.dim[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        #### Third and fourth layers
        ## Fully connected
        self.fc1 = nn.Linear(32 * self.dim[0] * self.dim[1], 100)
        self.fc2 = nn.Linear(100, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
