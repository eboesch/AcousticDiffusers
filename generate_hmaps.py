"""Script for generating heightmaps"""

import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from run_pipeline import preprocess
from working.gen_reliefs import optimize_tex, preprocess_tex


def height_map_pipeline(image_path: str, device: str = "cpu") -> np.array:
    """
    Generates a height map from an image and optimizes it for bas-relief.

    Args:
        image_path (str): Path to the image that should be processed.
        device (str): Device to run the model on, either 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Height map of the image, shape (260, 260) with 2 columns of 0 padding on every side. dtype: float64
    """
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    image = Image.open(image_path).convert("RGB")

    # generate height map
    hmap = preprocess(image, model, processor, device=device)  # Ensure preprocess supports device

    ht = 0.07

    # generate relief
    ht_low = -ht
    ht_high = ht
    tex = (hmap - np.min(hmap)) / (np.max(hmap) - np.min(hmap)) * (ht_high - ht_low) + ht_low
    tex = optimize_tex(tex, edge_border=2, iterations=100, smoothness_wt=3, device=device)
    tex = preprocess_tex(tex, edge_border=2)
    return tex


if __name__ == "__main__":

    data_dir = "mini_imagenet/"
    targets_path = "mini_imagenet/coeffs.csv"
    targets = pd.read_csv(targets_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test the height_map_pipeline function
    # image_path = "n01532829/n01532829_124.JPEG"
    # start = time.time()
    # tex = height_map_pipeline("mini_imagenet/" + image_path, device=device) # shape: 260x260 (256x256 plus 2 padding on each side)
    # end = time.time()
    # print(f"time: {end - start:.4f}s")
    # # print(tex[2:258, 2:258])

    output_dir = "hmaps_png/"
    ht = 0.07
    h_min = -ht
    h_max = ht

    # Only process images that have coefficients
    processed_paths = list(set(targets["path"].tolist()))
    image_paths = [data_dir + path for path in processed_paths]

    # Process all images
    # image_paths = [
    #     os.path.join(root, file)
    #     for root, _, files in os.walk(data_dir)
    #     for file in files
    #     if file.lower().endswith(('.jpeg', '.jpg'))
    # ]

    for path in tqdm(image_paths):
        img_dir = path.split("/")[1]
        os.makedirs(os.path.join(output_dir, img_dir), exist_ok=True)
        file = path.split("/")[-1].split(".")[0]
        # output_path = os.path.join(output_dir, img_dir, f"{file}.npy")
        # output_path = os.path.join(output_dir, img_dir, f"{file}.jpeg")
        output_path = os.path.join(output_dir, img_dir, f"{file}.png")
        if Path(output_path).exists():
            print(f"Skipping {output_path}, already exists.")
            continue

        hmap = height_map_pipeline(path, device=device)

        # normalize height map to 0-255
        hmap = (hmap - h_min) / (h_max - h_min) * 255

        im = Image.fromarray(hmap)
        im = im.convert("L")  # convert image to greyscale
        # im.save(output_path, quality=95) # for saving jpeg files
        im.save(output_path, compress=0)  # for saving png files
        # np.save(output_path, hmap) # for saving npy files
