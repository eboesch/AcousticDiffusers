import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Union
import math
import numpy as np

def expand(x: Union[int, Tuple[int, int]]):
    return (x, x) if isinstance(x, int) else x


def calc_output_size(in_shape: Tuple[int, int], kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1):
    """
    Calculates output height and width after input was been feed through a conv or pool layer
    """
    kernel = expand(kernel)
    stride = expand(stride)
    padding = expand(padding)
    dilation = expand(dilation)

    h_out = math.floor(((in_shape[0] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0]) + 1)
    w_out = math.floor(((in_shape[1] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1]) + 1)
    # print([h_out, w_out])
    return [h_out, w_out]


class CNN(nn.Module):
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
        dropout_rate_1: int = 0.25,
        dropout_rate_2: int = 0.5,
        linear_size: int = 64
    ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=cnn_out_channels_1, kernel_size=cnn_kernel_size, stride=cnn_stride, padding=cnn_padding)
        self.conv2 = nn.Conv2d(in_channels=cnn_out_channels_1, out_channels=cnn_out_channels_2, kernel_size=cnn_kernel_size, stride=cnn_stride, padding=cnn_padding)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.dropout1 = nn.Dropout(dropout_rate_1)
        self.dropout2 = nn.Dropout(dropout_rate_2)

        conv1_shape = calc_output_size(input_shape[1:], cnn_kernel_size, cnn_stride)
        pool1_shape = calc_output_size(conv1_shape, pool_kernel_size, pool_stride)
        conv2_shape = calc_output_size(pool1_shape, cnn_kernel_size, cnn_stride)
        pool2_shape = calc_output_size(conv2_shape, pool_kernel_size, pool_stride)
        flattened_size = int(cnn_out_channels_2*pool2_shape[0]*pool2_shape[1])

        self.fc1 = nn.Linear(flattened_size, linear_size)
        self.fc2 = nn.Linear(linear_size, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(model: nn.Module, train_loader: DataLoader, loss_fct, optimizer, log_interval, logger, device) -> None:
    # Set the model to training mode
    model.train()

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
        if batch_id % log_interval == log_interval-1:    
            logger.info(f'[{batch_id + 1:5d}/{len(train_loader):>5d}] Loss: {running_loss / log_interval:.4e}')
            running_loss = 0.0


class MRE(nn.Module):
    """
    Mean Relative Error
    """
    def __init__(self, eps=1e-8):
        super(MRE, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred)) / (torch.abs(target) + self.eps)) * 100
    
def compute_derivatives(u, domain_size_x=2.0, domain_size_y=2.0, domain_size_z=2.0):
    """
    Compute ux, uxx, uy, uyy, uz, uzz from a 3D field using Fourier differentiation.
    Matches approach used in the document (input in physical space).

    Args:
        u (Tensor): Shape (batch, nx, ny, nz)
        domain_size (float): Physical size of the square domain

    Returns:
        ux, uxx, uy, uyy, uz, uzz (Tensors): Each of shape (batch, nx, ny, nz)
    """
    batchsize, nx, ny, nz = u.shape
    device = u.device

    # Compute wavenumbers (assume 2π periodic)
    kx = torch.fft.fftfreq(nx, d=domain_size_x / nx).to(device) * 2 * np.pi
    ky = torch.fft.fftfreq(ny, d=domain_size_y / ny).to(device) * 2 * np.pi
    kz = torch.fft.fftfreq(nz, d=domain_size_z / nz).to(device) * 2 * np.pi

    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    kx = kx.unsqueeze(0)  # shape (1, nx, ny)
    ky = ky.unsqueeze(0)
    kz = kz.unsqueeze(0)


    # FFT of input
    u_hat = torch.fft.fftn(u)

    # Derivatives in Fourier space
    ux_hat = 1j * kx * u_hat
    uy_hat = 1j * ky * u_hat
    uz_hat = 1j * kz * u_hat

    uxx_hat = -kx ** 2 * u_hat
    uyy_hat = -ky ** 2 * u_hat
    uzz_hat = -kz ** 2 * u_hat


    # Inverse FFT to return to physical space
    ux = torch.fft.ifftn(ux_hat).real
    uy = torch.fft.ifftn(uy_hat).real
    uz = torch.fft.ifftn(uz_hat).real
    uxx = torch.fft.ifftn(uxx_hat).real
    uyy = torch.fft.ifftn(uyy_hat).real
    uzz = torch.fft.ifftn(uzz_hat).real

    return {'u': u, 'ux': ux, 'uxx': uxx, 'uy': uy, 'uyy': uyy, 'uz': uz, 'uzz': uzz}

class PDEloss(nn.Module):
    def __init__(self, device):
        super(PDEloss, self).__init__()
        self.device = device

    def forward(self, u):
        derivs = compute_derivatives(u, domain_size=100)

        uxx = derivs['uxx']
        uyy = derivs['uyy']
        uzz = derivs['uzz']

        #TODO: figure out how to get the frequency from the input
        f = 100       # frequency in Hz
        c = 343 # Speed of sound in m/s
        k = 2*np.pi * f / c # wavenumber
        pde = uxx + uyy + uzz + k**2 *u
        return torch.mean(pde**2)


def test(test_loader, model, loss_fcts, device):
    # Set the model to evaluation mode
    model.eval()
    losses = np.zeros(len(loss_fcts))

    batch_size = 16
    output_size = 17
    last_out = torch.zeros((batch_size, output_size)).to(device)
    n_outputs = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            for idx, loss in enumerate(loss_fcts):
                losses[idx] += loss(output, target).item()

            out_batch_size = output.shape[0]
            if (output - last_out[:out_batch_size, :]).any():
                n_outputs += 1
            last_out = output
            
    losses /= len(test_loader)
    print(f"number of distinct outputs: {n_outputs}")

    return losses    




