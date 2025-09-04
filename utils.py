import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd

ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]

def read_list(file_path):
    list = []
    with open(file_path, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            list.append(x)
    
    return list


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

def fft_laplacian(u, domain_size=2.0):
    """
    Compute ux, uxx, uy, uyy from a 2D field using Fourier differentiation.
    Matches approach used in the document (input in physical space).

    Args:
        u (Tensor): Shape (batch, nx, ny)
        domain_size (float): Physical size of the square domain

    Returns:
        ux, uxx, uy, uyy (Tensors): Each of shape (batch, nx, ny)
    """
    batchsize, n_freq, nx, ny = u.shape
    device = u.device


    # Compute wavenumbers (assume 2π periodic)
    kx = torch.fft.fftfreq(nx, d=domain_size / nx).to(device) * 2 * np.pi
    ky = torch.fft.fftfreq(ny, d=domain_size / ny).to(device) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    kx = kx.unsqueeze(0)  # shape (1, nx, ny)
    ky = ky.unsqueeze(0)

    # FFT of input
    u_hat = torch.fft.fft2(u, dim=(-2,-1))

    # laplacian in fourier space = uxx_hat + uyy_hat, for uxx_hat = -kx ** 2 * u_hat
    lap_hat = -(kx**2 + ky**2) * u_hat

    # Inverse FFT to return to physical space
    lap_u = torch.fft.ifft2(lap_hat, s=(nx,ny), dim=(-2,-1)).real
    return lap_u


def autograd_laplacian(u, coords):
    grads = autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2x = autograd.grad(grads[:,0], coords, grad_outputs=torch.ones_like(grads[:,0]), create_graph=True)[0][:,0]
    d2y = autograd.grad(grads[:,1], coords, grad_outputs=torch.ones_like(grads[:,1]), create_graph=True)[0][:,1]
    return d2x + d2y

class PDEloss(nn.Module):
    def __init__(self, reduction='sum', deriv_type='spectral'):
        super().__init__()

        assert reduction in ['mean', 'sum', 'none'], "reduction must be one of 'mean', 'sum', or 'none'"
        assert deriv_type in ['spectral', 'spectral_complex', 'spherical'], "deriv_type must be one of 'spectral', 'spectral_complex', or 'spherical'"
        self.reduction = reduction
        self.deriv_type = deriv_type

        # if deriv_type == 'autograd':
        #     x = torch.linspace(-50, 50, 256)
        #     y = torch.linspace(-50, 50, 256)
        #     xx, yy = torch.meshgrid(x, y, indexing="xy")
        #     self.coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (nx*ny, 2)
        #     self.coords.requires_grad_(True)

    @property
    def name(self):
        return f"PDE_{self.reduction}_reduction_{self.deriv_type}_loss"
        
    # def autograd_real_loss(self, u, freq):
    #     lap_u = laplacian(u, self.coords)

    #     c = 343 # Speed of sound in m/s
    #     k = 2*np.pi * freq / c # wavenumber
    #     if not torch.is_tensor(k):
    #         k = torch.tensor(float(k), dtype=self.coords.dtype, device=self.coords.device).expand_as(u)
    #     else:
    #         # k = k.view(-1).expand_as(u) if k.numel() > 1 else 
    #         k = k.expand_as(u)

    #     pde = lap_u + k**2 * u

    #     if self.reduction=='sum':
    #         return torch.sum(pde**2)
    #     elif self.reduction=='mean':
    #         return torch.mean(pde**2)
    #     else:
    #         return pde


    
    def fft_loss(self, u, freq):
        """
        freq: frequency in Hz
        
        """

        n_freqs = len(freq)

        lap = fft_laplacian(u, domain_size=100)

        c = 343 # Speed of sound in m/s
        freq = torch.Tensor(freq)
        k = 2*torch.pi * freq / c # wavenumber
        k = k.to(u.device, u.dtype).view(1, n_freqs, 1, 1)

        pde = lap + k**2 * u

        if self.reduction=='sum':
            return torch.sum(pde**2)
        elif self.reduction=='mean':
            return torch.mean(pde**2)
        else:
            return pde
        
    def fft_complex_loss(self, u, freq):
        """
        freq: frequency in Hz
        u: shape (batch, 2*n_freqs, nx, ny )
        
        """
        n_freqs = len(freq)
        batchsize, _, nx, ny = u.shape
        u_complex = u.view(batchsize, n_freqs, 2, nx, ny)
        u_real = u_complex[:,:,0,:,:]
        u_img = u_complex[:,:,1,:,:]
        

        lap_real = fft_laplacian(u_real, domain_size=100)
        lap_img = fft_laplacian(u_img, domain_size=100)

        c = 343 # Speed of sound in m/s
        freq = torch.Tensor(freq)
        k = 2*torch.pi * freq / c # wavenumber
        k = k.to(u.device, u.dtype).view(1, n_freqs, 1, 1)

        pde_real = lap_real + k**2 * u_real
        pde_img = lap_img + k**2 * u_real


        if self.reduction=='sum':
            return torch.sum(pde_real**2 + pde_img**2)
        elif self.reduction=='mean':
            return torch.mean(pde_real**2 + pde_img**2)
        else:
            return pde_real + pde_img
        
    def forward(self, u, freq):
        if self.deriv_type == "spectral":
            return self.fft_loss(u, freq)
        elif self.deriv_type == 'spectral_complex':
            return self.fft_complex_loss(u, freq)
        elif self.deriv_type == 'spherical':
            raise NotImplementedError

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
 
    batchsize, n_freq = pressures.shape[:2] # Batch size, Number of frequencies 
    L = torch.prod(torch.tensor(pressures.shape[2:])) # Number of Listeners (multiply over all spatial dimensions)

    # Flatten all spatial/listener dimensions and take absolute value
    if n_freq == 2*len(ALL_FREQ_BANDS):
        # complex pressures are encoded in 2 channels per frequency 
        pressures_complex = pressures.view(batchsize, len(ALL_FREQ_BANDS), 2, -1)
        # y = torch.abs(pressures)**2 # shape: (batch, n_frequencies, L)
        re, im = pressures_complex[:, :, 0], pressures_complex[:, :, 1]
        y = re**2 + im**2 + 1e-12
    elif n_freq == len(ALL_FREQ_BANDS):
        pressures = pressures.view(batchsize, n_freq, L)
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


