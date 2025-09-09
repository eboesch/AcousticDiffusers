import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import v2

from typing import Any, Callable, Tuple, Dict, List


class RotateToLandscape(torch.nn.Module):
    """
    Module for rotating images in portrait mode to landscape mode
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Rotates images in portait mode to landscape mode, and returns images already in landscape mode as is.

        Args:
            image (Tensor): Input image to be rotated

        Returns:
            (Tensor): The input image in landscape mode
        """
        img_size = image.shape  # channels, height, width

        # if image is in portrait format, rotate it
        if img_size[2] < img_size[1]:
            image = torch.rot90(image, dims=(1, 2))
        return image


class MaxCenterCrop(torch.nn.Module):
    """
    Module for cropping an image to a central square of maximum possible size.
    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Crops the input image to a central square of maximum possible size.

        Args:
            image (Tensor): Input image to be cropped

        Returns:
            (Tensor): Central cropped part of the image
        """
        h, w = img.shape[1:]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[:, top : top + min_dim, left : left + min_dim]


class normalize_target(torch.nn.Module):
    """
    Module for normalizing Tensors by the specified means and stds

    Attributes:
        means (array): Array of means, one per channel.
        stds (array): Array of standard deviations, one per channel.
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray):
        """
        Constructs an instance of normalize_target with the given means and stds.

        Args:
            means (array): Array of means, one per channel.
            stds (array): Array of standard deviations, one per channel.
        """
        super().__init__()
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the targets with the instance-specific means and stds

        Args:
            targets (Tensor): Targets to be normalized

        Returns:
            (Tensor): Normalized targets
        """
        device = targets.device
        return (targets - self.means.to(device)) / self.stds.to(device)

    def inverse_transform(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Inverse of target normalization: De-normalizes the targets with the instance-specific means and stds

        Args:
            targets (Tensor): Targets to be de-normalized

        Returns:
            (Tensor): De-ormalized targets
        """
        device = targets.device
        return targets * self.stds.to(device) + self.means.to(device)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for transform:  Returns normalized targets.
        """
        return self.transform(targets)


SIZE = 256

"""
Image Transform that crops images to a maximal central square, resizes them to 
a uniform square size, and scales them to [0, 1].
"""
pipeline_transform = v2.Compose(
    [MaxCenterCrop(), v2.Resize(size=(SIZE, SIZE)), v2.ToDtype(torch.float32, scale=True)]  # scale to [0.0, 1.0]
)

"""
Image Transform that rotates images to landscape, resizes them to 
a uniform size of (375, 500) (the median size of all images), and scales them to [0, 1].
"""
median_size_transform = v2.Compose(
    [RotateToLandscape(), v2.Resize(size=(375, 500)), v2.ToDtype(torch.float32, scale=True)]  # scale to [0.0, 1.0]
)

"""
Image Transform that scales images to [0, 1].
"""
default_img_transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])  # scale to [0.0, 1.0]


def target_pad(target: torch.Tensor, size_h: int = SIZE, size_w: int = SIZE) -> torch.Tensor:
    """
    Pads targets with zeros to reach the provided size.

    Args:
        target (Tensor): Target tensore to be padded.
        size_h (int): Height the target should be padded to.
        size_w (int): Width the target should be padded to.

    Returns:
        (Tensor): Target padded with zeros to reach shape (size_h, size_w)
    """
    target = target.unsqueeze(-1)
    return torch.nn.functional.pad(
        target, (0, size_h - target.shape[1], 0, size_w - target.shape[0]), mode="constant", value=0.0
    )


def default_rgb_loader(path: str) -> torch.Tensor:
    """
    Loads the RGB image at `path` to a torch Tensor with 3 channels.

    Args:
        path (Tensor): Path to the desired image

    Returns:
        (Tensor): Image as a torch tensor with 3 channels.
    """
    # use a torch vision function to decode
    image = io.decode_image(path, mode=io.ImageReadMode.RGB)

    return image


def default_gray_loader(path: str) -> torch.Tensor:
    """
    Loads the grayscale image at `path` to a torch Tensor with 1 channel.

    Args:
        path (Tensor): Path to the desired image

    Returns:
        (Tensor): Image as a torch tensor with 1 channel.
    """
    # use a torch vision function to decode grayscale image
    image = io.decode_image(path, mode=io.ImageReadMode.GRAY)

    return image


def default_npy_loader(path: str) -> torch.Tensor:
    """
    Loads the npy file at `path` to a torch Tensor.

    Args:
        path (Tensor): Path to the desired image

    Returns:
        (Tensor): Numpy array as a torch tensor.
    """
    hmap = np.load(path)
    hmap = torch.from_numpy(hmap).type(torch.float32).unsqueeze(0)  # load the height map and add a channel dimension
    return hmap


class ImageDataset(Dataset):
    """
    Dataset Class for loading and preprocessing images.

    Attributes:
        targets (pd.DataFrame): DataFrame containing all targets and the path to the corresponding image.
        data_dir (str): Path to the directory in which the data is stored.
        loader (Callable[[str], torch.Tensor]): Function that loads images to Tensors.
        transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input images.
        target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
    """

    def __init__(
        self,
        data_dir: str,
        targets_path: str,
        data_paths: List[str] = None,
        loader: Callable[[str], torch.Tensor] = default_rgb_loader,
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Constructs ImageDataset instance.

        Args:
            data_dir (str): Path to the directory in which the data is stored.
            targets_path (str): Path the the file where targets are stored.
            data_paths (List[str]): Optional list of paths to images. If provided only those images are added to the dataset.
            loader (Callable[[str], torch.Tensor]): Function that loads images to Tensors.
            transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input images.
            target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
        """

        # fetch targets
        self.targets = pd.read_csv(targets_path)
        if data_paths is not None:  # if a list of data paths is provided, only use those
            self.targets = self.targets[self.targets["path"].isin(data_paths)]

        self.data_dir = data_dir
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def calculate_target_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates mean and std of instance targets per channel/dimension.

        Returns:
            means, stds (np.ndarray): Arrays containing the mean resp. std per channel.
        """
        means, stds = self.targets.iloc[:, 1:].mean().to_numpy(), self.targets.iloc[:, 1:].std().to_numpy()
        return means, stds

    def calculate_data_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates mean and std of instance data per channel.

        Returns:
            means, stds (np.ndarray): Arrays containing the mean resp. std per channel.
        """
        l = self.__len__()
        n_channels, W, H = self.__getitem__(0)[0].shape
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        print(f"Calculating mean and std over {l} images...")

        # iterate through all images and sum values
        for i in tqdm(range(l)):
            img = self.__getitem__(i)[0]
            mean += torch.sum(img, dim=(1, 2))
            std += torch.sum(img**2, dim=(1, 2))

        # average
        n = W * H * l
        mean /= n
        std /= n
        return mean, std

    def set_transform(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Sets the instance's `transform` to the provided transform.
        """
        self.transform = transform

    def set_target_transform(self, target_transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Sets the instance's `target_transform` to the provided target_transform.
        """
        self.target_transform = target_transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the datapoint and target at the given index `idx`.
        """
        target = torch.tensor(self.targets.iloc[idx, 1:].to_numpy(dtype=np.float32))
        img_path = os.path.join(self.data_dir, self.targets.iloc[idx, 0])

        image = self.loader(img_path)

        # apply data and target transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.targets)


class DictImageDataset(ImageDataset):
    """
    Dataset Class for loading and preprocessing images. Returns items as dictionaries {'x': image, 'y': target}.

    Attributes:
        targets (pd.DataFrame): DataFrame containing all targets and the path to the corresponding image.
        data_dir (str): Path to the directory in which the data is stored.
        loader (Callable[[str], torch.Tensor]): Function that loads images to Tensors.
        transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input images.
        target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
    """

    def __init__(
        self,
        data_dir,
        targets_path,
        data_paths: list[str] = None,
        loader: Callable[[str], Any] = default_rgb_loader,
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Constructs DictImageDataset instance.

        Args:
            data_dir (str): Path to the directory in which the data is stored.
            targets_path (str): Path the the file where targets are stored.
            data_paths (List[str]): Optional list of paths to images. If provided only those images are added to the dataset.
            loader (Callable[[str], torch.Tensor]): Function that loads images to Tensors.
            transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input images.
            target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
        """
        super().__init__(data_dir, targets_path, data_paths, loader, transform, target_transform)

    def calculate_data_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates mean and std of instance data per channel.

        Returns:
            means, stds (np.ndarray): Arrays containing the mean resp. std per channel.
        """
        l = self.__len__()
        n_channels, W, H = self.__getitem__(0)["x"].shape
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        print(f"Calculating mean and std over {l} images...")

        # iterate through all images and sum
        for i in tqdm(range(l)):
            img = self.__getitem__(i)["x"]
            mean += torch.sum(img, dim=(1, 2))
            std += torch.sum(img**2, dim=(1, 2))

        # average
        n = W * H * l
        mean /= n
        std /= n
        return mean, std

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Fetches the datapoint and target at the given index `idx` and returns them 
        as a dictionary where 'x' is the datapoint and 'y' is the target.
        """

        # fetch image and target as in superclass ImageDataset
        image, target = super().__getitem__(idx)

        # return image and target as dictionary
        return {"x": image, "y": target}


def replace_path_ending(path: str, file_type: str = "npy") -> str:
    """
    Replace the ending of a file path with a new ending `file_type`. E.g. change 'image.JPEG' to 'image.npy'.

    Args:
        path (str): Path whose ending should be changed
        file_type (str): New path ending that should override the original one

    Returns:
        (str): Path with the updated ending
    """
    return os.path.splitext(path)[0] + "." + file_type


class HmapDataset(ImageDataset):
    """
    Dataset Class for loading and preprocessing height maps.

    Attributes:
        targets (pd.DataFrame): DataFrame containing all targets and the path to the corresponding image.
        data_dir (str): Path to the directory in which the data is stored.
        loader (Callable[[str], torch.Tensor]): Function that loads heightmap files to Tensors.
        transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input heightmaps.
        target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
    """

    def __init__(
        self,
        data_dir,
        targets_path,
        data_paths: list[str] = None,
        file_type: str = "png",
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Constructs HmapDataset instance.

        Args:
            data_dir (str): Path to the directory in which the data is stored.
            targets_path (str): Path the the file where targets are stored.
            data_paths (List[str]): Optional list of paths to images. If provided only those images are added to the dataset.
            file_type (str): Format of the heightmaps, e.g. png or npy.
            transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input heightmaps.
            target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
        """
        super().__init__(
            data_dir,
            targets_path,
            data_paths=data_paths,
            loader=default_gray_loader,
            transform=transform,
            target_transform=target_transform,
        )

        if file_type == "npy":
            self.loader = default_npy_loader
        # change target paths to have the correct file types
        self.targets["path"] = self.targets["path"].map(lambda path: replace_path_ending(path, file_type))


class DictHmapDataset(DictImageDataset):
    """
    Dataset Class for loading and preprocessing height maps. Returns items as dictionaries {'x': hmap, 'y': target}.

    Attributes:
        targets (pd.DataFrame): DataFrame containing all targets and the path to the corresponding image.
        data_dir (str): Path to the directory in which the data is stored.
        loader (Callable[[str], torch.Tensor]): Function that loads heightmap files to Tensors.
        transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input heightmaps.
        target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
    """

    def __init__(
        self,
        data_dir,
        targets_path,
        data_paths: list[str] = None,
        file_type: str = "png",
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Constructs DictHmapDataset instance.

        Args:
            data_dir (str): Path to the directory in which the data is stored.
            targets_path (str): Path the the file where targets are stored.
            data_paths (List[str]): Optional list of paths to images. If provided only those images are added to the dataset.
            file_type (str): Format of the heightmaps, e.g. png or npy.
            transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to input heightmaps.
            target_transform (Callable[[torch.Tensor], torch.Tensor]): Preprocessing transform function to apply to targets.
        """
        super().__init__(
            data_dir,
            targets_path,
            data_paths=data_paths,
            loader=default_gray_loader,
            transform=transform,
            target_transform=target_transform,
        )

        if file_type == "npy":
            self.loader = default_npy_loader
        # change target paths to have the correct file types
        self.targets["path"] = self.targets["path"].map(lambda path: replace_path_ending(path, file_type))


if __name__ == "__main__":
    data_dir = "mini_imagenet/"
    targets_path = "mini_imagenet/coeffs.csv"

    df = pd.read_csv(targets_path)
    processed_paths = list(set(df["path"].tolist()))

    relative_paths = processed_paths[:3]

    # test ImageDataset class
    # dataset = ImageDataset(data_dir, targets_path, transform=pipeline_transform())
    # print(dataset[0])

    # test HmapDataset class
    data_dir = "hmaps/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = HmapDataset("hmaps_png/", targets_path, file_type="png", transform=default_img_transform())
    print(dataset[465])
    print(dataset.calculate_data_stats())
