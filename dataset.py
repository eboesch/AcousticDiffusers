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


class RotateToLandscape(torch.nn.Module):
    def forward(self, image):
        img_size = image.shape # channels, height, width

        # if image is in portrait format, rotate it
        if img_size[2] < img_size[1]:
            image = torch.rot90(image, dims=(1,2))
        return image


# size = 256
size = 375



class MaxCenterCrop(torch.nn.Module):
    def forward(self, img):
        h, w = img.shape[1:]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[:, top:top+min_dim, left:left+min_dim]

class resize(torch.nn.Module):
    def forward(self, img, size=256):
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


pipeline_transform = v2.Compose([
    MaxCenterCrop(),
    # resize(),
    v2.Resize(size=(size, size)),
    v2.ToDtype(torch.float32, scale=True) # scale to [0.0, 1.0]
])

default_transform = v2.Compose([
    RotateToLandscape(),
    v2.Resize(size=(375, 500)),
    v2.ToDtype(torch.float32, scale=True) # scale to [0.0, 1.0]
])

# size = 20
# def target_expand(target):
#     long = target.expand(size, -1)
#     return long.expand(size, long.shape[1])

def target_pad(target):
    target = target.unsqueeze(-1)
    return torch.nn.functional.pad(target, (0, size - target.shape[1], 0, size - target.shape[0]), mode='constant', value=0.0)


def default_loader(path: str) -> Any:
    # use a torch vision function to decode
    image = io.decode_image(path, mode=io.ImageReadMode.RGB)
    

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(img_path, "rb") as f:
    #     image = Image.open(f)
    # image = Image.open(path)
    # image = image.convert("RGB")

    return image


class ImageDataset(Dataset):
    def __init__(self, relative_paths, data_dir, targets_path, loader: Callable[[str], Any] = default_loader, transform_mode: str = "default", target_transform: Any = None):

        self.targets = pd.read_csv(targets_path)

        self.data_dir = data_dir
        self.relative_paths = relative_paths

        self.loader = loader
    
        if transform_mode == "default":
            self.transform = default_transform
        elif transform_mode == "pipeline":
            self.transform = pipeline_transform
        else:
            raise ValueError("Unknown transform type.")
        self.target_transform = target_transform

        # for path in relative_paths:
        #     data =  


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
    def __init__(self, relative_paths, data_dir, targets_path, loader: Callable[[str], Any] = default_loader, transform_mode: str = "default", target_transform: Any = None):
         super().__init__(relative_paths, data_dir, targets_path, loader, transform_mode, target_transform)


    def __getitem__(self, idx) -> dict:
        target = torch.tensor(self.targets.iloc[idx, 1:].to_numpy(dtype=np.float32))#, dtype = torch.float32)
        img_path = os.path.join(self.data_dir, self.targets.iloc[idx, 0])

        image = self.loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return {"x": image, "y": target}
    



# class ImageDataset(ImageFolder):
#     def __init__(self, data_dir: str, targets_path: str, *args: Any, **kwargs: Any) -> None:
#         super().__init__(data_dir, *args, **kwargs) # problem: reads *all* files, but we only have targets for a few
#         paths, _ = zip(*self.imgs)

#         targets = pd.read_csv(targets_path)


#         self.targets = [targets.loc[targets["path"]==str(path),:].drop(columns="path").to_numpy()[0] for path in paths]
#         self.samples = self.imgs = list(zip(paths, self.targets))




if __name__ == "__main__":
    data_dir = "mini_imagenet/"
    targets_path = "mini_imagenet/coeffs.csv"

    df = pd.read_csv(targets_path)
    processed_paths = list(set(df['path'].tolist()))

    relative_paths = processed_paths[:3]

    dataset = ImageDataset(relative_paths, data_dir, targets_path, transform_mode="pipeline")
    print(dataset[0])

    t=0