"""Script for running inference on a trained and saved CNN model"""

import numpy as np
import pandas as pd
import random
import os
import logging
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset.dataset import ImageDataset, HmapDataset, normalize_target, pipeline_transform, default_img_transform
from models.cnn import CNN, test, PhysicsCNN
from utils import MRE, read_list, load_model_from_checkpoint, compute_diffusion_coeffs


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%d-%H-%M-%S")

    # Training settings
    seed = 0
    epochs = 10
    batch_size = 16  # 64
    lr = 1e-3
    log_interval = 100
    save_inferences = True
    loggin_dir = "logs/"

    model_code = "2025-09-04-11-06-45"  # CNN, hmap
    # model_code = "2025-09-04-11-16-40" # CNN, image
    model_type = "best_model"
    model_cls = CNN # CNN or PhysicsCNN
    data_split = "test"  # test, val or train

    model_dir = loggin_dir + model_code + "/"
    model_path = loggin_dir + model_code + "/" + model_type + ".pth"
    inferences_path = f"logs/{model_code}/{timestamp}_{model_type}_{data_split}_inferences.csv"

    logging.basicConfig(
        handlers=[logging.FileHandler(f"{model_dir}{timestamp}_{model_type}_{data_split}_losses.log")],
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M",
    )

    torch.manual_seed(seed)
    random.seed(seed)

    data_dir = "mini_imagenet/"
    hmap_dir = "hmaps_png/"
    targets_path = "mini_imagenet/coeffs.csv"

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    # logger.info("Using %s device", device)

    # ---------------------------------------- Dataset ----------------------------------------
    dataset_type = "hmap"  # "hmap" or "image"
    dataset_split = "pre_split"  # "pre_split", "random_split" or "subset"

    train_paths = read_list("dataset/train_paths.txt")
    val_paths = read_list("dataset/val_paths.txt")
    test_paths = read_list("dataset/test_paths.txt")

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 1 - train_ratio - val_ratio

    # if not using the pre-defined data splits, overwrite the path lists
    if dataset_split == "subset":
        # take only a subsample of the data
        n_samples = 1000
        sample_paths = random.sample(train_paths + val_paths + test_paths, n_samples)
        train_paths = sample_paths[: int(len(sample_paths) * train_ratio)]
        val_paths = sample_paths[
            int(len(sample_paths) * train_ratio) : int(len(sample_paths) * (train_ratio + val_ratio)) + 1
        ]
        test_paths = sample_paths[int(len(sample_paths) * (train_ratio + val_ratio)) + 1 :]
    elif dataset_split == "random_split":
        # split paths randomly into train/val/test
        all_paths = random.shuffle(train_paths + val_paths + test_paths)
        train_paths = all_paths[: int(len(all_paths) * train_ratio)]
        val_paths = all_paths[int(len(all_paths) * train_ratio) : int(len(all_paths) * (train_ratio + val_ratio)) + 1]
        test_paths = all_paths[int(len(all_paths) * (train_ratio + val_ratio)) + 1 :]

    # create datasets
    if dataset_type == "image":
        transform = pipeline_transform
        train_dataset = ImageDataset(data_dir, targets_path, train_paths, transform=transform)
        val_dataset = ImageDataset(data_dir, targets_path, val_paths, transform=transform)
        test_dataset = ImageDataset(data_dir, targets_path, test_paths, transform=transform)
    elif dataset_type == "hmap":
        transform = default_img_transform
        train_dataset = HmapDataset(hmap_dir, targets_path, train_paths, transform=transform)
        val_dataset = HmapDataset(hmap_dir, targets_path, val_paths, transform=transform)
        test_dataset = HmapDataset(hmap_dir, targets_path, test_paths, transform=transform)

    # logger.info(f"Using {dataset_split} {dataset_type} Dataset with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")
    input_shape = list(train_dataset[0][0].shape)
    output_size = int(train_dataset[0][1].shape[0])
    # logger.info(f"Input shape: {input_shape}")
    # logger.info(f"Target size: {output_size}")

    # ------------------ Data Transforms ------------------
    # data normalization transform
    data_mean, data_std = train_dataset.calculate_data_stats()  # compute mean and std of training set per channel
    transform.transforms.append(v2.Normalize(data_mean, data_std))
    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    # target normalization transform
    target_mean, target_std = train_dataset.calculate_target_stats()  # compute mean and std of targets per channel
    target_transform = normalize_target(means=target_mean, stds=target_std)
    train_dataset.set_target_transform(target_transform)
    val_dataset.set_target_transform(target_transform)
    test_dataset.set_target_transform(target_transform)

    # logger.info(f"Data transform:\n{transform}")
    # logger.info(f"Target transform: {target_transform}")

    if data_split == "val":
        test_dataset = val_dataset
    elif data_split != "test":
        raise NotImplementedError(f"data_split {data_split} not implemented, currently only 'test' and 'val' are supported.")

    # ------------------ DataLoaders ------------------
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---------------------------------------- Model ----------------------------------------
    # logger.info("Loading Model...")
    # load model
    if os.path.isfile(model_path):
        # for older models only state_dict was saved
        model = CNN(input_shape=input_shape, output_size=output_size).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))  # load the saved best model
    else:
        # for newer models state_dict + metadata was saved
        model = load_model_from_checkpoint(model_cls=model_cls, save_folder=model_dir, save_name="best_model")
        model = model.to(device)

    # ------------------ Losses ------------------
    mse_eval_loss_fct = nn.MSELoss()
    mae_eval_loss_fct = nn.L1Loss()
    rel_eval_loss_fct = MRE()
    eval_losses = {"MSE": mse_eval_loss_fct, "MAE": mae_eval_loss_fct, "MRE": rel_eval_loss_fct}

    # ---------------------------------------- Evaluation ----------------------------------------
    # logger.info("Starting Evaluation...")

    model.eval()  # Set model to evaluation mode

    preds = []
    targets = []

    print("Total: ", len(test_dataset))
    last_out = torch.zeros((batch_size, output_size)).to(device)
    n_outputs = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            if model_cls == PhysicsCNN:
                output = compute_diffusion_coeffs(output)

            # compute losses on unnormalized outputs to ensure comparability
            if target_transform:
                output = target_transform.inverse_transform(output)
                target = target_transform.inverse_transform(target)

            # store targets and predictions
            targets.append(target)
            preds.append(output)

            # track number of distinct outputs to ensure model isn't collapsing
            out_batch_size = output.shape[0]
            if (output - last_out[:out_batch_size, :]).any():
                n_outputs += 1
                # print(f"number of distinct outputs: {n_outputs}")
                # print(f"output: {output}")
            last_out = output

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # compute losses
    losses = {}
    for key in eval_losses:
        loss = eval_losses[key]
        losses.update({key: loss(preds, targets).item()})

    # logger.info(f"number of distinct outputs: {n_outputs}")
    print(f"number of distinct outputs: {n_outputs}")

    # ---------------------------------------- Store Inferences ----------------------------------------
    if save_inferences:
        df = pd.DataFrame()

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        frequncies = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]
        for i, freq in enumerate(frequncies):
            df[f"targets_{freq}"] = targets[:, i]
            df[f"preds_{freq}"] = preds[:, i]
            df[f"mae_{freq}"] = np.abs(targets[:, i] - preds[:, i])
            df[f"mse_{freq}"] = (targets[:, i] - preds[:, i]) ** 2
            df[f"mre_{freq}"] = np.abs((targets[:, i] - preds[:, i])) / (np.abs(targets[:, i]) + 1e-12) * 100

        df.to_csv(inferences_path, index=False)

    # Report losses
    print(
        f'MSE Evaluation loss: {losses["MSE"]:.4e},   MAE Evaluation loss: {losses["MAE"]:.4e},   Relative Validation Loss: {losses["MRE"]:.4f}'
    )
    logger.info(
        f'MSE Evaluation loss: {losses["MSE"]:.4e},   MAE Evaluation loss: {losses["MAE"]:.4e},   Relative Validation Loss: {losses["MRE"]:.4f}'
    )
