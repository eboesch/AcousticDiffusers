import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MRE(nn.Module):
    """
    Mean Relative Error
    """
    def __init__(self, reduction='mean', eps=1e-8):
        super(MRE, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "reduction must be one of 'mean', 'sum', or 'none'"
        
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred, target):
        elementwise_loss = torch.abs((target - pred)) / (torch.abs(target) + self.eps)
        if self.reduction == 'mean':
            return torch.mean(elementwise_loss)*100
        elif self.reduction == 'sum':
            return torch.sum(elementwise_loss)*100
        else:
            return elementwise_loss*100
    
def compute_derivatives_3D(u, domain_size_x=2.0, domain_size_y=2.0, domain_size_z=2.0):
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

def compute_derivatives_2D(u, domain_size=2.0):
    """
    Compute ux, uxx, uy, uyy from a 2D field using Fourier differentiation.
    Matches approach used in the document (input in physical space).

    Args:
        u (Tensor): Shape (batch, nx, ny)
        domain_size (float): Physical size of the square domain

    Returns:
        ux, uxx, uy, uyy (Tensors): Each of shape (batch, nx, ny)
    """
    batchsize, nx, ny = u.shape
    device = u.device

    # Compute wavenumbers (assume 2π periodic)
    kx = torch.fft.fftfreq(nx, d=domain_size / nx).to(device) * 2 * np.pi
    ky = torch.fft.fftfreq(ny, d=domain_size / ny).to(device) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    kx = kx.unsqueeze(0)  # shape (1, nx, ny)
    ky = ky.unsqueeze(0)

    # FFT of input
    u_hat = torch.fft.fft2(u)

    # Derivatives in Fourier space
    ux_hat = 1j * kx * u_hat
    uy_hat = 1j * ky * u_hat
    uxx_hat = -kx ** 2 * u_hat
    uyy_hat = -ky ** 2 * u_hat

    # Inverse FFT to return to physical space
    ux = torch.fft.ifft2(ux_hat).real
    uy = torch.fft.ifft2(uy_hat).real
    uxx = torch.fft.ifft2(uxx_hat).real
    uyy = torch.fft.ifft2(uyy_hat).real

    return {'u': u, 'ux': ux, 'uxx': uxx, 'uy': uy, 'uyy': uyy}

def PDEloss(u, freq, reduction='mean'):
    derivs = compute_derivatives_2D(u, domain_size=100)

    uxx = derivs['uxx']
    uyy = derivs['uyy']

    #TODO: figure out how to get the frequency from the input
    # f = 100       # frequency in Hz
    c = 343 # Speed of sound in m/s
    k = 2*np.pi * freq / c # wavenumber
    pde = uxx + uyy + k**2 *u

    if reduction=='sum':
        return torch.sum(pde**2)
    elif reduction=='mean':
        return torch.mean(pde**2)
    else:
        return pde

class PDEloss3D(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, u, freq):
        derivs = compute_derivatives_3D(u, domain_size=100)

        uxx = derivs['uxx']
        uyy = derivs['uyy']
        uzz = derivs['uzz']

        #TODO: figure out how to get the frequency from the input
        # f = 100       # frequency in Hz
        c = 343 # Speed of sound in m/s
        k = 2*np.pi * freq / c # wavenumber
        pde = uxx + uyy + uzz + k**2 *u
        return torch.mean(pde**2)


def compute_diffusion_coeffs(pressures: torch.Tensor) -> torch.Tensor:
    """
    Computes diffusion coefficients from pressure values.
    
    Args:
        pressures (np.ndarray): Array of pressure values. Shape: (batch, n_frequencies, [spatial dimensions]). 
        E.g. (batch, n_frequencies, nx_listeners, ny_listeners) or (batch, n_frequencies, nx_listeners, ny_listeners, nz_listeners).

        
    Returns:
        np.ndarray: Array of diffusion coefficients. Shape: (batch, n_frequencies,).
    """
 
    B, F = pressures.shape[:2] # Batch size, Number of frequencies
    L = torch.prod(torch.tensor(pressures.shape[2:])) # Number of Listeners (multiply over all spatial dimensions)

    # Flatten all spatial/listener dimensions
    pressures = pressures.view(B, F, L) # shape: (batch, n_frequencies, L)

    y = torch.abs(pressures)**2 # shape: (batch, n_frequencies, L)

    sum_of_squares = torch.sum(y**2, dim=2) # shape: (batch, n_frequencies,)
    coeffs = (torch.sum(y, dim=2)**2 - sum_of_squares)/( (L-1)*sum_of_squares + 1e-20) # shape: (batch, n_frequencies,)

    return coeffs


if __name__ == "__main__":

    image_path = "n01532829/n01532829_124.JPEG"
    sample = np.load('mini_imagenet/simulations/n01532829/n01532829_124.npy')
    sample = sample.reshape(1, sample.shape[0], 32,32,32)
    sample = torch.tensor(sample, dtype=torch.complex64)


    # TODO: fix the function! computed coefficients are WAY off
    coeffs = compute_diffusion_coeffs(sample)
    print(f"Computed Coefficients: {coeffs}")


    df = pd.read_csv("mini_imagenet/simulations/coeffs.csv")
    target = df.loc[df["path"]==image_path,:].drop(columns="path").to_numpy()[0]
    print(f"Target: {target}")

    print(f"Difference: {torch.tensor(target)-coeffs}")

