"""Script for finetuning a pretrained vision model"""

import random
import logging
from datetime import datetime
import os

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import ImageDataset, HmapDataset, normalize_target, PIL_rgb_loader, PIL_gray_loader
from models.cnn import train_epoch, test
from utils import MRE, read_list


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_dir = f"logs/{timestamp}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logging.basicConfig(
        handlers=[logging.FileHandler(model_dir + "/logs.log")],
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M",
    )

    blurb = "Tiny ViT, hmap, batch_size = 32, lr = 1e-3 "
    logger.info(
        f"--------------------------------------------\nStarting Script\n{blurb}\n--------------------------------------------"
    )

    # Training settings
    seed = 0
    epochs = 10
    batch_size = 32
    lr = 1e-3
    log_interval = 100
    save_model = True

    logger.info(f"lr: {lr}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"epochs: {epochs}")

    torch.manual_seed(seed)
    random.seed(seed)

    data_dir = "mini_imagenet/"
    hmap_dir = "hmaps_png/"
    targets_path = "mini_imagenet/coeffs.csv"

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    logger.info("Using %s device", device)

    # ---------------------------------------- Dataset ----------------------------------------
    logger.info("Creating Dataset...")

    dataset_type = "hmap"  # "hmap" or "image"
    dataset_split = "pre_split"  # "pre_split", "random_split" or "subset"

    # if dataset_type == "image":
    #     in_channels = 3
    # elif dataset_type == "hmap":
    #     in_channels = 1
    # TODO: currently hmaps are read as 3 channel RGBs. the model's default preprocessing expects 3 channels, 
    # so to read hmaps as grayscale would require changing the preprocessing too.
    in_channels = 3 

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
        train_dataset = ImageDataset(data_dir, targets_path, train_paths, loader=PIL_rgb_loader)
        val_dataset = ImageDataset(data_dir, targets_path, val_paths, loader=PIL_rgb_loader)
        test_dataset = ImageDataset(data_dir, targets_path, test_paths, loader=PIL_rgb_loader)
    elif dataset_type == "hmap":
        train_dataset = HmapDataset(hmap_dir, targets_path, train_paths, loader=PIL_rgb_loader)
        val_dataset = HmapDataset(hmap_dir, targets_path, val_paths, loader=PIL_rgb_loader)
        test_dataset = HmapDataset(hmap_dir, targets_path, test_paths, loader=PIL_rgb_loader)

    logger.info(
        f"Using {dataset_split} {dataset_type} Dataset with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples"
    )
    output_size = int(train_dataset[0][1].shape[0])
    logger.info(f"Target size: {output_size}")


    # ---------------------------------------- Model ----------------------------------------
    logger.info("Creating Model...")

    # model_name = 'resnet50'
    model_name = "deit_tiny_patch16_224" # Tiny ViT
    # model_name = 'mobilenetv3_large_100'

    model = timm.create_model(model_name, pretrained=True, in_chans=in_channels, num_classes=output_size)
    # model = timm.create_model(model_name, pretrained=True, in_chans=in_channels, num_classes=output_size, global_pool="avg")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # ------------------ Data Transforms ------------------
    # data normalization transform
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg) # use the model's default preprocessing
    transform = timm.data.create_transform(**data_cfg)
    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    # target normalization transform
    target_mean, target_std = train_dataset.calculate_target_stats()  # compute mean and std of targets per channel
    target_transform = normalize_target(means=target_mean, stds=target_std)
    train_dataset.set_target_transform(target_transform)
    val_dataset.set_target_transform(target_transform)
    test_dataset.set_target_transform(target_transform)

    logger.info(f"Data transform:\n{transform}")
    logger.info(f"Target transform: {target_transform}")

    # ------------------ DataLoaders ------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # ------------------ Losses ------------------
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    rel_loss = MRE()
    eval_losses = {"MSE": mse_loss, "MAE": mae_loss, "MRE": rel_loss}
    val_criterion = "MRE"

    training_loss = mae_loss

    logger.info("Model: %s", model)
    logger.info(f"Using {training_loss} training loss")

    # ---------------------------------------- Training ----------------------------------------
    logger.info("Starting Training...")
    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        logger.info(f"Epoch {epoch+1}")


        # train a single epoch
        train_epoch(model, train_loader, training_loss, optimizer, log_interval, logger, device)
        # compute validation loss for all eval_losses
        val_losses = test(val_loader, model, eval_losses, val_dataset.target_transform, device)

        logger.info(
            f'MSE Validation Loss: {val_losses["MSE"]:.4e},   MAE Validation Loss: {val_losses["MAE"]:.4e},   Relative Validation Loss: {val_losses["MRE"]:.4f}'
        )
        print(
            f'MSE Validation Loss: {val_losses["MSE"]:.4e},   MAE Validation Loss: {val_losses["MAE"]:.4e},   Relative Validation Loss: {val_losses["MRE"]:.4f}'
        )

        # save model with best validation loss
        if val_losses[val_criterion] < best_val_loss:
            best_val_loss = val_losses[val_criterion]
            if save_model:
                torch.save(model.state_dict(), model_dir + "best_model.pth")

    if save_model:
        torch.save(model.state_dict(), model_dir + "final_model.pth")

    # ---------------------------------------- Evaluation ----------------------------------------
    logger.info("Starting Evaluation...")
    # Evaluation
    if save_model:  # load best model to evaluate
        model.load_state_dict(torch.load(model_dir + "best_model.pth", weights_only=True))  # load the saved best model
        model = model.to(device)

    test_losses = test(test_loader, model, eval_losses, test_dataset.target_transform, device)

    # report losses
    logger.info(
        f'MSE Evaluation loss: {test_losses["MSE"]:.4e},   MAE Evaluation loss: {test_losses["MAE"]:.4e},   Relative Evaluation Loss: {test_losses["MRE"]:.4f}'
    )
    print(
        f'MSE Evaluation loss: {test_losses["MSE"]:.4e},   MAE Evaluation loss: {test_losses["MAE"]:.4e},   Relative Evaluation Loss: {test_losses["MRE"]:.4f}'
    )
