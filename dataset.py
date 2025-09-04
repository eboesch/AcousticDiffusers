import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import io
from typing import Any, Callable
from PIL import Image
import pandas as pd
import numpy as np
import os
from torchvision.transforms import v2
# from working.gen_reliefs import center_crop, resize
import cv2
from tqdm import tqdm


class RotateToLandscape(torch.nn.Module):
    def forward(self, image):
        img_size = image.shape # channels, height, width

        # if image is in portrait format, rotate it
        if img_size[2] < img_size[1]:
            image = torch.rot90(image, dims=(1,2))
        return image


size = 256
# size = 375



class MaxCenterCrop(torch.nn.Module):
    def forward(self, img):
        h, w = img.shape[1:]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[:, top:top+min_dim, left:left+min_dim]


class normalize_target(torch.nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def transform(self, targets):   
        device = targets.device     
        return (targets - self.means.to(device)) / self.stds.to(device)
    
    def inverse_transform(self, targets):
        device = targets.device 
        return targets * self.stds.to(device) + self.means.to(device)
    
    def forward(self, targets):
        return self.transform(targets)

pipeline_transform = v2.Compose([
    MaxCenterCrop(),
    v2.Resize(size=(size, size)),
    v2.ToDtype(torch.float32, scale=True) # scale to [0.0, 1.0]
])

# default_img_transform = v2.Compose([
#     RotateToLandscape(),
#     v2.Resize(size=(375, 500)),
#     v2.ToDtype(torch.float32, scale=True) # scale to [0.0, 1.0]
# ])

default_img_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True) # scale to [0.0, 1.0]
])

# size = 20
# def target_expand(target):
#     long = target.expand(size, -1)
#     return long.expand(size, long.shape[1])

def target_pad(target):
    target = target.unsqueeze(-1)
    return torch.nn.functional.pad(target, (0, size - target.shape[1], 0, size - target.shape[0]), mode='constant', value=0.0)


def default_rgb_loader(path: str) -> Any:
    # use a torch vision function to decode
    image = io.decode_image(path, mode=io.ImageReadMode.RGB)
    
    return image

def default_gray_loader(path: str) -> Any:
    # use a torch vision function to decode grayscale image
    image = io.decode_image(path, mode=io.ImageReadMode.GRAY)

    return image

def default_npy_loader(path: str) -> Any:
    hmap = np.load(path)
    hmap = torch.from_numpy(hmap).type(torch.float32).unsqueeze(0)  # load the height map and add a channel dimension
    return hmap

class ImageDataset(Dataset):
    def __init__(self, data_dir, targets_path, data_paths: list[str] = None, loader: Callable[[str], Any] = default_rgb_loader, transform: Any = None, target_transform: Any = None):

        self.targets = pd.read_csv(targets_path)
        if data_paths is not None: # if a list of data paths is provided, only use those
            self.targets = self.targets[self.targets['path'].isin(data_paths)]

        self.data_dir = data_dir

        self.loader = loader
    
        self.transform = transform
        
        self.target_transform = target_transform
  

    def calculate_target_stats(self):
        means, stds = self.targets.iloc[:, 1:].mean().to_numpy(), self.targets.iloc[:, 1:].std().to_numpy()
        return means, stds
    
    def calculate_data_stats(self):
        l = self.__len__()
        n_channels, W, H = self.__getitem__(0)[0].shape
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        print(f"Calculating mean and std over {l} images...")
        for i in tqdm(range(l)):
            img = self.__getitem__(i)[0]
            mean += torch.sum(img, dim=(1,2))
            std += torch.sum(img**2, dim=(1,2))
        
        n = W*H*l
        mean /= n
        std /= n
        return mean, std
    
    def set_transform(self, transform):
        self.transform = transform

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform

    def __getitem__(self, idx):
        target = torch.tensor(self.targets.iloc[idx, 1:].to_numpy(dtype=np.float32))#, dtype = torch.float32)
        img_path = os.path.join(self.data_dir, self.targets.iloc[idx, 0])

        image = self.loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target
    
        
    def __len__(self):        
        return len(self.targets)
    



class DictImageDataset(ImageDataset):
    def __init__(self, data_dir, targets_path, data_paths: list[str] = None, loader: Callable[[str], Any] = default_rgb_loader, transform: Any = None, target_transform: Any = None):
         super().__init__(data_dir, targets_path, data_paths, loader, transform, target_transform)

    def calculate_data_stats(self):
        l = self.__len__()
        n_channels, W, H = self.__getitem__(0)["x"].shape
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        print(f"Calculating mean and std over {l} images...")
        for i in tqdm(range(l)):
            img = self.__getitem__(i)["x"]
            mean += torch.sum(img, dim=(1,2))
            std += torch.sum(img**2, dim=(1,2))
        
        n = W*H*l
        mean /= n
        std /= n
        return mean, std

    def __getitem__(self, idx) -> dict:

        image, target = super().__getitem__(idx)

        return {"x": image, "y": target}
    
def replace_path_ending(path: str, file_type: str='npy') -> str:
    """
    Replace the ending of a file path with a new ending `file_type`. E.g. change 'image.JPEG' to 'image.npy'.
    """
    return os.path.splitext(path)[0] + "." + file_type

class HmapDataset(ImageDataset):
    def __init__(self, data_dir, targets_path, data_paths: list[str] = None, file_type: str="png", transform: Any = None, target_transform: Any = None):
        
        super().__init__(data_dir, targets_path, data_paths=data_paths, loader=default_gray_loader, transform=transform, target_transform=target_transform)

        if file_type == 'npy':
            self.loader = default_npy_loader 
        self.targets['path'] = self.targets['path'].map(lambda path: replace_path_ending(path, file_type))

# class DictHmapDataset(HmapDataset):
#     def __init__(self, data_dir, targets_path, data_paths: list[str] = None, file_type: str="png", transform: Any = None, target_transform: Any = None):
#          super().__init__(data_dir, targets_path, data_paths, file_type, transform, target_transform)


#     def __getitem__(self, idx) -> dict:        
#         image, target = super().__getitem__(idx)
#         return {"x": image, "y": target}

class DictHmapDataset(DictImageDataset):
    def __init__(self, data_dir, targets_path, data_paths: list[str] = None, file_type: str="png", transform: Any = None, target_transform: Any = None):
        
        super().__init__(data_dir, targets_path, data_paths=data_paths, loader=default_gray_loader, transform=transform, target_transform=target_transform)

        if file_type == 'npy':
            self.loader = default_npy_loader 
        self.targets['path'] = self.targets['path'].map(lambda path: replace_path_ending(path, file_type))




if __name__ == "__main__":
    data_dir = "mini_imagenet/"
    targets_path = "mini_imagenet/coeffs.csv"

    df = pd.read_csv(targets_path)
    processed_paths = list(set(df['path'].tolist()))

    relative_paths = processed_paths[:3]

    # test ImageDataset class
    # dataset = ImageDataset(data_dir, targets_path, transform=pipeline_transform())
    # print(dataset[0])


    # test HmapDataset class
    data_dir = "hmaps/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # dataset = HmapDataset(data_dir, targets_path, file_type='npy', transform=default_img_transform())
    # print(dataset[465])
    dataset = HmapDataset("hmaps_png/", targets_path, file_type='png', transform=default_img_transform())
    # print(dataset[465])
    print(dataset.calculate_data_stats())

    t=0