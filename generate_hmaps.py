import numpy as np
import pandas as pd
import os
from run_pipeline import preprocess
from working.gen_reliefs import optimize_tex, preprocess_tex
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import time
from tqdm import tqdm
from pathlib import Path


def height_map_pipeline(image_path: str, device: str = 'cpu'):
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

    # os.makedirs("tmp", exist_ok=True)
    # rnd_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    # obj_path = f"tmp/{rnd_name}.obj"
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

    

    output_dir = "hmaps/"

    # Only process images that have coefficients
    processed_paths = list(set(targets['path'].tolist()))
    # processed_paths = ["n01532829/n01532829_7398.JPEG" ]
    image_paths = [data_dir + path for path in processed_paths]

    # Process all images
    # image_paths = [
    #     os.path.join(root, file)
    #     for root, _, files in os.walk(data_dir)
    #     for file in files
    #     if file.lower().endswith(('.jpeg', '.jpg'))
    # ]

    for path in tqdm(image_paths):
        img_dir = path.split("/")[0]
        os.makedirs(os.path.join(output_dir, img_dir), exist_ok=True)
        file = path.split("/")[-1].split(".")[0]
        output_path = os.path.join(output_dir, img_dir, f"{file}.npy")
        if Path(output_path).exists():
            print(f"Skipping {output_path}, already exists.")
            continue

        hmap = height_map_pipeline(path, device=device)
        np.save(output_path, hmap)



