import numpy as np
import pandas as pd
import os
import warnings
import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Dict, Type

ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]


def read_list(file_path: str) -> List[str]:
    """
    Reads a list from a file and returns it.

    Args:
        file_path (str): Path to the file that should be read

    Returns:
        List[str]: List of lines in the provided file
    """
    list = []
    with open(file_path, "r") as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            list.append(x)

    return list


def load_model_from_checkpoint(model_cls: Type[nn.Module], save_folder: str, save_name: str = "best_model") -> nn.Module:
    """
    Loads an FNO-based model of class `model_class` from `save_folder`.

    Args:
        model_cls (Type[nn.Module]): The class of the model being loaded.
        save_folder (str): Path to the location the model was saved.
        save_name (str): Name of the model. Default is 'best_model'.

    Returns:
        (nn-Module): Instance of the model_class with weights restored from the saved model.
    """

    metadata_filepath = os.path.join(save_folder, save_name + "_metadata.pkl")

    # Force old behavior torch.load behavior
    init_kwargs = torch.load(metadata_filepath, weights_only=False)

    version = init_kwargs.pop("_version")
    if hasattr(model_cls, "_version") and version != model_cls._version:
        print(version)
        warnings.warn(
            f"Checkpoint saved for version {version} of model {model_cls._name} but current code is version {model_cls._version}"
        )

    if "args" in init_kwargs:
        init_args = init_kwargs.pop("args")
    else:
        init_args = []
    instance = model_cls(*init_args, **init_kwargs)

    # Load state dict
    state_dict_path = os.path.join(save_folder, save_name + "_state_dict.pt")
    state = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    instance.load_state_dict(state)

    return instance


class MRE(nn.Module):
    """
    Class for computing the mean relative error:   |x-y| / (|y|+ eps)

    Attributes:
        redution (str): How the loss should be reduced. If 'mean', take the mean over all values. 
            If 'sum', sum all values. If 'none', don't reduce and return elementwise loss tensor instead.
        eps (float): Small value to add to denominator to avoid division by zero.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        """
        Constructs MRE instance.

        Args:
            redution (str): How the loss should be reduced. If 'mean', take the mean over all values. 
                If 'sum', sum all values. If 'none', don't reduce and return elementwise loss tensor instead.
            eps (float): Small value to add to denominator to avoid division by zero.
        """
        super(MRE, self).__init__()
        assert reduction in ["mean", "sum", "none"], "reduction must be one of 'mean', 'sum', or 'none'"

        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the relative error between prediction and target, and reduces according to instance specifications.

        Args:
            pred (Tensor): The prediction that should match the target.
            target (Tensor): Groundtruth target, used to scale the relative error.
        """
        elementwise_loss = torch.abs((target - pred)) / (torch.abs(target) + self.eps)
        if self.reduction == "mean":
            return torch.mean(elementwise_loss) * 100
        elif self.reduction == "sum":
            return torch.sum(elementwise_loss) * 100
        else:
            return elementwise_loss * 100


def compute_derivatives_3D(
    u: torch.Tensor, domain_size_x: float = 2.0, domain_size_y: float = 2.0, domain_size_z: float = 2.0
) -> Dict[str, torch.Tensor]:
    """
    Compute ux, uxx, uy, uyy, uz, uzz from a 3D field using Fourier differentiation.

    Args:
        u (Tensor): Tensor to compute the derivatives of. Shape (batch, nx, ny, nz)
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

    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")
    kx = kx.unsqueeze(0)  # shape (1, nx, ny)
    ky = ky.unsqueeze(0)
    kz = kz.unsqueeze(0)

    # FFT of input
    u_hat = torch.fft.fftn(u)

    # Derivatives in Fourier space
    ux_hat = 1j * kx * u_hat
    uy_hat = 1j * ky * u_hat
    uz_hat = 1j * kz * u_hat

    uxx_hat = -(kx**2) * u_hat
    uyy_hat = -(ky**2) * u_hat
    uzz_hat = -(kz**2) * u_hat

    # Inverse FFT to return to physical space
    ux = torch.fft.ifftn(ux_hat).real
    uy = torch.fft.ifftn(uy_hat).real
    uz = torch.fft.ifftn(uz_hat).real
    uxx = torch.fft.ifftn(uxx_hat).real
    uyy = torch.fft.ifftn(uyy_hat).real
    uzz = torch.fft.ifftn(uzz_hat).real

    return {"u": u, "ux": ux, "uxx": uxx, "uy": uy, "uyy": uyy, "uz": uz, "uzz": uzz}


def compute_derivatives_2D(u: torch.Tensor, domain_size: float = 2.0) -> Dict[str, torch.Tensor]:
    """
    Compute ux, uxx, uy, uyy from a 2D field using Fourier differentiation.

    Args:
        u (Tensor): Tensor to compute the derivatives of. Shape (batch, nx, ny)
        domain_size (float): Physical size of the square domain

    Returns:
        ux, uxx, uy, uyy (Tensors): Each of shape (batch, nx, ny)
    """
    batchsize, nx, ny = u.shape
    device = u.device

    # Compute wavenumbers (assume 2π periodic)
    kx = torch.fft.fftfreq(nx, d=domain_size / nx).to(device) * 2 * np.pi
    ky = torch.fft.fftfreq(ny, d=domain_size / ny).to(device) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    kx = kx.unsqueeze(0)  # shape (1, nx, ny)
    ky = ky.unsqueeze(0)

    # FFT of input
    u_hat = torch.fft.fft2(u)

    # Derivatives in Fourier space
    ux_hat = 1j * kx * u_hat
    uy_hat = 1j * ky * u_hat
    uxx_hat = -(kx**2) * u_hat
    uyy_hat = -(ky**2) * u_hat

    # Inverse FFT to return to physical space
    ux = torch.fft.ifft2(ux_hat).real
    uy = torch.fft.ifft2(uy_hat).real
    uxx = torch.fft.ifft2(uxx_hat).real
    uyy = torch.fft.ifft2(uyy_hat).real

    return {"u": u, "ux": ux, "uxx": uxx, "uy": uy, "uyy": uyy}


def fft_laplacian(u: torch.Tensor, domain_size: float = 2.0) -> torch.Tensor:
    """
    Computes the Laplacian using Fourier differentiation.

    Args:
        u (Tensor): Tensor to compute the Laplacian of. Shape (batch, nx, ny)
        domain_size (float): Physical size of the square domain

    Returns:
        Tensor: The laplacian of the given input u.
    """
    batchsize, n_freq, nx, ny = u.shape
    device = u.device

    # Compute wavenumbers (assume 2π periodic)
    kx = torch.fft.fftfreq(nx, d=domain_size / nx).to(device) * 2 * np.pi
    ky = torch.fft.fftfreq(ny, d=domain_size / ny).to(device) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    kx = kx.unsqueeze(0)  # shape (1, nx, ny)
    ky = ky.unsqueeze(0)

    # FFT of input
    u_hat = torch.fft.fft2(u, dim=(-2, -1))

    # laplacian in fourier space = uxx_hat + uyy_hat, for uxx_hat = -kx ** 2 * u_hat
    lap_hat = -(kx**2 + ky**2) * u_hat

    # Inverse FFT to return to physical space
    lap_u = torch.fft.ifft2(lap_hat, s=(nx, ny), dim=(-2, -1)).real
    return lap_u


def autograd_laplacian(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplacian using autograd.

    Args:
        u (Tensor): Tensor to compute the Laplacian of. Shape (batch, nx, ny)
        coords (Tensor): Coordinates to at which to compute the derivatives

    Returns:
        Tensor: The laplacian of the given input u.
    """
    grads = autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2x = autograd.grad(grads[:, 0], coords, grad_outputs=torch.ones_like(grads[:, 0]), create_graph=True)[0][:, 0]
    d2y = autograd.grad(grads[:, 1], coords, grad_outputs=torch.ones_like(grads[:, 1]), create_graph=True)[0][:, 1]
    return d2x + d2y


class PDEloss(nn.Module):
    """
    Class that bundles all variations of the PDE loss for the helmholtz equation: ∇^2 * u + k^2 * u = 0
    where u is a 2D pressure field

    Attributes:
        redution (str): How the loss should be reduced. If 'mean', take the mean over all values. 
            If 'sum', sum all values. If 'none', don't reduce and return elementwise loss tensor instead.
        deriv_type (str): How the derivatives should be computed. 
            For 'spectral', derivatives are computed in Fourier space and input is real-valued. 
            For 'spectral_complex', derivatives are computed in Fourier space and input is interpreted as complex values.
    """

    def __init__(self, reduction="sum", deriv_type="spectral"):
        """
        Constructs a PDEloss instance of given specifications.

        Args:
            redution (str): How the loss should be reduced. If 'mean', take the mean over all values. 
                If 'sum', sum all values. If 'none', don't reduce and return elementwise loss tensor instead.
            deriv_type (str): How the derivatives should be computed. 
                For 'spectral', derivatives are computed in Fourier space and input is real-valued. 
                For 'spectral_complex', derivatives are computed in Fourier space and input is interpreted as complex values.
        """
        super().__init__()

        assert reduction in ["mean", "sum", "none"], "reduction must be one of 'mean', 'sum', or 'none'"
        assert deriv_type in [
            "spectral",
            "spectral_complex",
            "spherical",
        ], "deriv_type must be one of 'spectral', 'spectral_complex', or 'spherical'"
        self.reduction = reduction
        self.deriv_type = deriv_type

    @property
    def name(self):
        """
        Returns a string containing the specifications of the PDE loss instance
        """
        return f"PDE_{self.reduction}_reduction_{self.deriv_type}_loss"

    def fft_loss(self, u: torch.Tensor, freq: List) -> torch.Tensor:
        """
        Computes spectral physics loss for real-valued pressures.

        Args:
            u (Tensor): 2D pressure fields, one for each frequency. Shape (batch, n_freqs, nx, ny )
            freq (List): List of frequencies in Hz

        Returns:
            Tensor: Tensor containing the computed PDE loss
        """

        n_freqs = len(freq)

        lap = fft_laplacian(u, domain_size=100)

        c = 343  # Speed of sound in m/s
        freq = torch.Tensor(freq)
        k = 2 * torch.pi * freq / c  # wavenumber
        k = k.to(u.device, u.dtype).view(1, n_freqs, 1, 1)

        pde = lap + k**2 * u

        if self.reduction == "sum":
            return torch.sum(pde**2)
        elif self.reduction == "mean":
            return torch.mean(pde**2)
        else:
            return pde

    def fft_complex_loss(self, u: torch.Tensor, freq: List) -> torch.Tensor:
        """
        Computes spectral physics loss for complex pressures.

        Args:
            u (Tensor): A 2D pressure field for each frequency, with complex values 
                encoded in separate channels. Shape (batch, 2*n_freqs, nx, ny )
            freq (List): List of frequencies in Hz

        Returns:
            Tensor: Tensor containing the computed PDE loss
        """
        n_freqs = len(freq)
        batchsize, _, nx, ny = u.shape
        u_complex = u.view(batchsize, n_freqs, 2, nx, ny)
        u_real = u_complex[:, :, 0, :, :]
        u_img = u_complex[:, :, 1, :, :]

        # compute spectral laplacians
        lap_real = fft_laplacian(u_real, domain_size=100)
        lap_img = fft_laplacian(u_img, domain_size=100)

        c = 343  # Speed of sound in m/s
        freq = torch.Tensor(freq)
        k = 2 * torch.pi * freq / c  # wavenumber
        k = k.to(u.device, u.dtype).view(1, n_freqs, 1, 1)

        # compute physical loss
        pde_real = lap_real + k**2 * u_real
        pde_img = lap_img + k**2 * u_real

        if self.reduction == "sum":
            return torch.sum(pde_real**2 + pde_img**2)
        elif self.reduction == "mean":
            return torch.mean(pde_real**2 + pde_img**2)
        else:
            return pde_real + pde_img

    def forward(self, u: torch.Tensor, freq: List) -> torch.Tensor:
        """
        Calls the loss function implementation matching the specifications of the PDEloss instance.

        Args:
            u (Tensor): A 2D pressure field for each frequency, with complex values 
                encoded in separate channels. Shape (batch, 2*n_freqs, nx, ny )
            freq (List): List of frequencies in Hz

        Returns:
            Tensor: Tensor containing the computed PDE loss
        """
        if self.deriv_type == "spectral":
            return self.fft_loss(u, freq)
        elif self.deriv_type == "spectral_complex":
            return self.fft_complex_loss(u, freq)
        elif self.deriv_type == "spherical":
            raise NotImplementedError


class PDEloss3D(nn.Module):
    """
    Class that bundles all variations of the PDE loss for the helmholtz equation: ∇^2 * u + k^2 * u = 0
    where u is a 3D pressure field

    Attributes:
        redution (str): How the loss should be reduced. If 'mean', take the mean over all values. 
            If 'sum', sum all values. If 'none', don't reduce and return elementwise loss tensor instead.
    """

    def __init__(self, reduction: str = "sum"):
        """
        Constructs a PDEloss3D instance of given specifications.

        Args:
            redution (str): How the loss should be reduced. If 'mean', take the mean over all values. 
                If 'sum', sum all values. If 'none', don't reduce and return elementwise loss tensor instead.
        """
        super().__init__()

        assert reduction in ["mean", "sum", "none"], "reduction must be one of 'mean', 'sum', or 'none'"
        self.reduction = reduction

    def forward(self, u: torch.Tensor, freq: List) -> torch.Tensor:
        """
        Computes spectral physics loss for real-valued pressures.

        Args:
            u (Tensor): 2D pressure fields, one for each frequency. Shape (batch, n_freqs, nx, ny )
            freq (List): List of frequencies in Hz

        Returns:
            Tensor: Tensor containing the computed PDE loss
        """
        n_freqs = len(freq)
        derivs = compute_derivatives_3D(u, domain_size=100)

        uxx = derivs["uxx"]
        uyy = derivs["uyy"]
        uzz = derivs["uzz"]

        c = 343  # Speed of sound in m/s
        freq = torch.Tensor(freq)
        k = 2 * torch.pi * freq / c  # wavenumber
        k = k.to(u.device, u.dtype).view(1, n_freqs, 1, 1)

        pde = uxx + uyy + uzz + k**2 * u
        return torch.mean(pde**2)


def compute_diffusion_coeffs(pressures: torch.Tensor) -> torch.Tensor:
    """
    Computes diffusion coefficients from pressure values.

    Args:
        pressures (np.ndarray): Array of pressure values. Shape: (batch, n_frequencies, [spatial dimensions]).
        E.g. (batch, n_frequencies, nx_listeners, ny_listeners) or (batch, n_frequencies, nx_listeners, ny_listeners, nz_listeners).


    Returns:
        torch.Tensor: Tensor of diffusion coefficients. Shape: (batch, n_frequencies,).
    """

    batchsize, n_freq = pressures.shape[:2]  # Batch size, Number of frequencies
    L = torch.prod(torch.tensor(pressures.shape[2:]))  # Number of Listeners (multiply over all spatial dimensions)

    # Flatten all spatial/listener dimensions and take absolute value
    if n_freq == 2 * len(ALL_FREQ_BANDS):
        # complex pressures are encoded in 2 channels per frequency
        pressures_complex = pressures.view(batchsize, len(ALL_FREQ_BANDS), 2, -1)
        re, im = pressures_complex[:, :, 0], pressures_complex[:, :, 1]
        y = re**2 + im**2 + 1e-12
    elif n_freq == len(ALL_FREQ_BANDS):
        pressures = pressures.view(batchsize, n_freq, L)
        y = torch.abs(pressures) ** 2  # shape: (batch, n_frequencies, L)

    sum_of_squares = torch.sum(y**2, dim=2)  # shape: (batch, n_frequencies,)
    coeffs = (torch.sum(y, dim=2) ** 2 - sum_of_squares) / (
        (L - 1) * sum_of_squares + 1e-20
    )  # shape: (batch, n_frequencies,)

    return coeffs


if __name__ == "__main__":

    # test coefficient function by comparing to targets
    image_path = "n01532829/n01532829_124.JPEG"
    sample = np.load("mini_imagenet/simulations/n01532829/n01532829_124.npy")
    sample = sample.reshape(1, sample.shape[0], 32, 32, 32)
    sample = torch.tensor(sample, dtype=torch.complex64)

    coeffs = compute_diffusion_coeffs(sample)
    print(f"Computed Coefficients: {coeffs}")

    df = pd.read_csv("mini_imagenet/simulations/coeffs.csv")
    target = df.loc[df["path"] == image_path, :].drop(columns="path").to_numpy()[0]
    print(f"Target: {target}")

    print(f"Difference: {torch.tensor(target)-coeffs}")  # significant difference!
