import os
import random
import string
import argparse
from pipeline.build import acoustics3d as ac3d
import pipeline.pyoptim.helpers as H
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from working.gen_reliefs import gen_hmap, center_crop, gen_relief, resize, contrast

ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]

def pipeline(mesh_fname, silent=True, src_pt=[0, 100, 0], lr=50):
    diffbem = ac3d.DiffBEM(128, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array(src_pt), lr, 5, False)
    diffbem.silent = silent
    Ps, Es = H.read_mesh(mesh_fname)
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
        print(f"freq {freq_band} Hz done: {values[-1]}")
    return values

def save(save_path, save_name, coeffs):
    with open(save_path, "a") as f:
        f.write(f"{save_name}," + ",".join(str(c) for c in coeffs) + "\n")

def preprocess(image, model, processor, device, size=256):

    # --- Predict Depth ---
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # --- Resize to original image size ---
    depth_resized = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth_resized.cpu().numpy()
    depth_norm = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()))

    hmap = gen_hmap(depth_norm)
    # hmap = contrast(depth_norm, method='clahe')
    hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap_square = center_crop(0.333 * hmap_norm + 0.666 * depth_norm)
    # hmap_square = center_crop(depth_norm)
    hmap_resized = resize(hmap_square, size)
    return 1 - hmap_resized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate relief maps from images in a directory.")
    parser.add_argument("--dir", required=True, help="Path to the folder containing JPEG images.")
    parser.add_argument("--csv_file", default="info.csv", help="Name of CSV file (within output folder).")
    args = parser.parse_args()

    # --- Detect device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    # Load CSV if exists
    if os.path.exists(args.csv_file):
        df = pd.read_csv(args.csv_file)
        processed_paths = set(df['path'].tolist())
    else:
        processed_paths = set()

    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.dir)
        for file in files
        if file.lower().endswith(('.jpeg', '.jpg'))
    ]

    for i, image_path in tqdm(enumerate(image_paths)):
        relative_path = '/'.join(image_path.split('/')[1:])
        print(relative_path)
        if relative_path in processed_paths:
            print(relative_path, 'has already been processed, skipping...')
            continue

        image = Image.open(image_path).convert("RGB")
        hmap = preprocess(image, model, processor, device=device)  # Ensure preprocess supports device

        os.makedirs("tmp", exist_ok=True)
        rnd_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        obj_path = f"tmp/{rnd_name}.obj"
        gen_relief(hmap, obj_path, ht=0.07, device=device)
        coeffs = pipeline(obj_path)
        save(args.csv_file, relative_path, coeffs)

        if os.path.exists(obj_path):
            os.remove(obj_path)
            