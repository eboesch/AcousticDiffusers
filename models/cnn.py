import math
from tqdm import tqdm
from logging import Logger
from typing import Tuple, Union, Dict, Any, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def expand(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Expands single number inputs to a tuple with the same entry twice. E.g. for an input x=3, returns (3,3).
    If the input is already a tuple, leave it as is.

    Args:
        x (int or Tuple[int, int]): Input to be expanded to a tuple.

    Returns:
        (Tuple): Tuple containing the input value twice, or the original input (if it was already a tuple).
    """
    return (x, x) if isinstance(x, int) else x


def calc_output_size(
    in_shape: Tuple[int, int],
    kernel: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> List[int]:
    """
    Calculates output height and width after input was been feed through a 2D conv or pool layer.

    Args:
        in_shape (Tuple[int, int]): 2D tuple containing the height and width of input.
        kernel (int or Tuple[int, int]]): Size of the convolving kernel. 
        stride (int or Tuple[int, int]]): Stride of the convolution. 
        padding (int or Tuple[int, int]]): Padding added to all four sides of the input. 
        dilation (int or Tuple[int, int]]): Spacing between kernel elements. 

        For kernel, stride, padding, if a single value is provided, then the same value is used in both dimensions.

    Returns:
        (List[int]): List containing the height and width of the output
    """
    kernel = expand(kernel)
    stride = expand(stride)
    padding = expand(padding)
    dilation = expand(dilation)

    h_out = math.floor(((in_shape[0] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0]) + 1)
    w_out = math.floor(((in_shape[1] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1]) + 1)
    return [h_out, w_out]


class CNN(nn.Module):
    """
    Custom CNN class with 2 convolutional layers, each followed by a MaxPooling layer, and 2 linear layers.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        output_size: int,
        cnn_out_channels_1: int = 16,
        cnn_out_channels_2: int = 32,
        cnn_kernel_size: int = 3,
        cnn_stride: int = 1,
        cnn_padding: int = 0,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        linear_size: int = 64,
    ):
        """
        Constructs a CNN instance:

        Args:
            input_shape (Tuple[int, int, int]): Shape of input, i.e. (C, H, W)
            output_size (int): Size of the output, i.e. the number of values the model should output
            cnn_out_channels_1 (int): Number of channels in the first convolutional layer
            cnn_out_channels_2 (int): Number of channels in the second convolutional layer
            cnn_kernel_size (int): Size of the convolutional kernel.
            cnn_stride (int): Stride of the convolutional kernel
            cnn_padding (int): Padding added to all four sides of the input.
            pool_kernel_size: Size of the kernel of the pooling layers.
            pool_stride: Stride of the kernel of the pooling layers
            linear_size: Hidden size of the linear layer
        """
        super(CNN, self).__init__()
        # set up layers
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=cnn_out_channels_1,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_padding,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.conv2 = nn.Conv2d(
            in_channels=cnn_out_channels_1,
            out_channels=cnn_out_channels_2,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_padding,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # compute size of output after convolutional layers
        conv1_shape = calc_output_size(input_shape[1:], cnn_kernel_size, cnn_stride)
        pool1_shape = calc_output_size(conv1_shape, pool_kernel_size, pool_stride)
        conv2_shape = calc_output_size(pool1_shape, cnn_kernel_size, cnn_stride)
        pool2_shape = calc_output_size(conv2_shape, pool_kernel_size, pool_stride)
        flattened_size = int(cnn_out_channels_2 * pool2_shape[0] * pool2_shape[1])

        self.fc1 = nn.Linear(flattened_size, linear_size)
        self.fc2 = nn.Linear(linear_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class PhysicsCNN(nn.Module):
    """
    Custom CNN class with 2 convolutional layers, each followed by a MaxPooling layer, and 2 linear layers.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        out_channels: int = 17,
        cnn_out_channels_1: int = 16,
        cnn_kernel_size: int = 3,
        cnn_stride: int = 1,
        cnn_padding: int = 0,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
    ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=cnn_out_channels_1,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_padding,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.conv2 = nn.Conv2d(
            in_channels=cnn_out_channels_1,
            out_channels=out_channels,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_padding,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        return x


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    log_interval: int,
    logger: Logger,
    device: str = "cpu",
) -> None:
    """
    Trains the provided model for a single epoch.

    Args:
        model (nn.Module): The model being trained
        train_loader (DataLoader): DataLoader containing the training data.
        loss_fct (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function being minimized during training.
        optimizer (optim.Optimizer): The optimized used for training.
        log_interval (int): Interval between batches at which training progress should be logged.
        logger (Logger): Logger to write training progress to.
        device (str): Device (i.e. 'cuda' or 'cpu') the model is being trained on.
    """
    # Set the model to training mode
    model.train()

    # iterate through data loaded
    running_loss = 0.0
    for batch_id, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fct(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        # print(loss.item())

        # Print Statistics
        running_loss += loss.item()
        if batch_id % log_interval == log_interval - 1:
            logger.info(f"[{batch_id + 1:5d}/{len(train_loader):>5d}] Loss: {running_loss / log_interval:.4e}")
            running_loss = 0.0


def test_batchwise(
    test_loader: DataLoader,
    model: nn.Module,
    loss_fcts: Dict[str, Callable],
    target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates the given model on the test loader and computes all the desired losses. 
    Losses are computed per batch and then averaged over all batches.

    Args:
        test_loader (DataLoader): DataLoader containing the testing data.
        model (nn.Module): The model being evaluated.
        loss_fcts (Dict[str, Callable]): Dictionary containing the loss functions for all losses to be computed.
        target_transform (Callable[[torch.Tensor], torch.Tensor]): 
            The transform function that was applied to preprocess the targets. Should have a method `inverse_transform`
        device (str): Device (i.e. 'cuda' or 'cpu') the model is

    Returns:
        (Dict[str, float]): Dictionary containing all computed losses.
    """
    # Set the model to evaluation mode
    model.eval()

    batch_size = test_loader.batch_size
    output_size = test_loader.dataset[0][1].shape[0]

    losses = {key: 0.0 for key in loss_fcts}

    last_out = torch.zeros((batch_size, output_size)).to(device)
    n_outputs = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # during evaluation, take loss between untransformed predicition and targets -> apply inverse normalization
            if target_transform:
                output = target_transform.inverse_transform(output)
                target = target_transform.inverse_transform(target)

            for key in loss_fcts:
                loss = loss_fcts[key]
                losses[key] += loss(output, target).item()

            # keep track of number of distinct outputs to make sure model isn't collapsing
            out_batch_size = output.shape[0]
            if (output - last_out[:out_batch_size, :]).any():
                n_outputs += 1
            last_out = output

    losses = {k: v / len(test_loader) for k, v in losses.items()}
    print(f"number of distinct outputs: {n_outputs}")

    return losses


def test(
    test_loader: DataLoader,
    model: nn.Module,
    loss_fcts: Dict[str, Callable],
    target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
    device: str = "cpu",
    return_preds: bool = False,
    return_targets: bool = False,
) -> Union[
    Dict[str, float], Tuple[Dict[str, float], torch.Tensor], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]
]:
    """
    Evaluates the given model on the test loader and computes all the desired losses

    Args:
        test_loader (DataLoader): DataLoader containing the testing data.
        model (nn.Module): The model being evaluated.
        loss_fcts (Dict[str, Callable]): Dictionary containing the loss functions for all losses to be computed.
        target_transform (Callable[[torch.Tensor], torch.Tensor]): 
            The transform function that was applied to preprocess the targets. Should have a method `inverse_transform`
        device (str): Device (i.e. 'cuda' or 'cpu') the model is
        return_preds (bool): If True, predictions are also returned.
        return_targets (bool): If True, targets are also returned.

    Returns:
        (Dict[str, float]): Dictionary containing all computed losses.
        Optionally also returns the targets and/or the predictions.
    """
    model.eval()

    preds = []
    targets = []

    batch_size = test_loader.batch_size
    output_size = test_loader.dataset[0][1].shape[0]

    last_out = torch.zeros((batch_size, output_size)).to(device)
    n_outputs = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)

            if target_transform:
                output = target_transform.inverse_transform(output)
                target = target_transform.inverse_transform(target)

            # collect targets and predictions
            targets.append(target)
            preds.append(output)

            # keep track of number of distinct outputs to make sure model isn't collapsing
            out_batch_size = output.shape[0]
            if (output - last_out[:out_batch_size, :]).any():
                n_outputs += 1
                # print(f"number of distinct outputs: {n_outputs}")
                # print(f"output: {output}")
            last_out = output

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # compute losses
    losses = {}
    for key in loss_fcts:
        loss = loss_fcts[key]
        losses.update({key: loss(preds, targets).item()})

    if return_preds and return_targets:
        return losses, preds, targets
    elif return_preds:
        return losses, preds
    elif return_targets:
        return losses, targets
    else:
        return losses
