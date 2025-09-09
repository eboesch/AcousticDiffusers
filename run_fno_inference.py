"""Script for running inference on a trained and saved FNO-based model"""
import numpy as np
import pandas as pd
import random
import logging
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from neuralop.models import FNO, TFNO
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from dataset import DictImageDataset, DictHmapDataset, normalize_target, default_img_transform, pipeline_transform
from custom_fno import CustomTrainer, PhysicsTrainer, FNOtoScalar, FNO_MAE, FNO_MSE, FNO_MRE, FNO_CNN, load_fno_from_checkpoint
from utils import PDEloss, read_list


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    timestamp = datetime.now().strftime("%d-%H-%M-%S")


    # Training settings
    seed = 0
    epochs = 10
    batch_size = 16#64
    lr = 1e-3
    log_interval = 100
    save_inferences = True
    loggin_dir = "logs/"

    model_code = "2025-09-03-22-57-41" # physics FNO, hmaps
    # model_code = "2025-09-04-08-27-59" # physics FNO, images
    # model_code = "2025-09-03-20-57-36" # FNO to scalar, hmaps
    # model_code = "2025-09-03-22-53-35" # FNO to scalar, images    
    model_type = "best_model"
    data_split = "test"
    model_class = FNO
    # model_class = FNOtoScalar

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
    dataset_type = "image"  # "hmap" or "image"
    dataset_split = "mini"  # "pre_split", "random_split" or "subset"

    train_paths = read_list('train_paths.txt')
    val_paths = read_list('val_paths.txt')
    test_paths = read_list('test_paths.txt')


    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 1-train_ratio-val_ratio

    # if not using the pre-defined data splits, overwrite the path lists
    if dataset_split == "subset":
        # take only a subsample of the data
        n_samples = 1000
        sample_paths = random.sample(train_paths+val_paths+test_paths, n_samples)
        train_paths = sample_paths[:int(len(sample_paths)*train_ratio)]
        val_paths = sample_paths[int(len(sample_paths)*train_ratio): int(len(sample_paths)*(train_ratio+val_ratio))+1]
        test_paths = sample_paths[int(len(sample_paths)*(train_ratio+val_ratio))+1:]
    elif dataset_split == "random_split":
        # split paths randomly into train/val/test
        all_paths = random.shuffle(train_paths+val_paths+test_paths)
        train_paths = all_paths[:int(len(all_paths)*train_ratio)]
        val_paths = all_paths[int(len(all_paths)*train_ratio): int(len(all_paths)*(train_ratio+val_ratio))+1]
        test_paths = all_paths[int(len(all_paths)*(train_ratio+val_ratio))+1:]
    
    # create datasets
    if dataset_type == "image":
        transform = pipeline_transform
        train_dataset = DictImageDataset(data_dir, targets_path, train_paths, transform=transform)
        val_dataset = DictImageDataset(data_dir, targets_path, val_paths, transform=transform)
        test_dataset = DictImageDataset(data_dir, targets_path, test_paths, transform=transform)
    elif dataset_type == "hmap":
        transform = default_img_transform
        train_dataset = DictHmapDataset(hmap_dir, targets_path, train_paths, transform=transform)
        val_dataset = DictHmapDataset(hmap_dir, targets_path, val_paths, transform=transform)
        test_dataset = DictHmapDataset(hmap_dir, targets_path, test_paths, transform=transform)

    # logger.info(f"Using {dataset_split} {dataset_type} Dataset with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")
    input_shape = list(train_dataset[0]["x"].shape)
    output_size = int(train_dataset[0]["y"].shape[0])
    # logger.info(f"Input shape: {input_shape}")
    # logger.info(f"Target size: {output_size}")


    # ------------------ Data Transforms ------------------   
    # data normalization transform
    data_mean, data_std = train_dataset.calculate_data_stats() # compute mean and std of training set per channel
    transform.transforms.append(v2.Normalize(data_mean, data_std)) 
    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    # target normalization transform
    target_mean, target_std = train_dataset.calculate_target_stats() # compute mean and std of targets per channel
    target_transform = normalize_target(means=target_mean, stds=target_std)

    # logger.info(f"Data transform:\n{transform}")
    # logger.info(f"Target transform: {target_transform}")

    if data_split == "val":
        test_dataset = val_dataset

    # ------------------ DataLoaders ------------------ 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # ---------------------------------------- Model ----------------------------------------
    # logger.info("Loading Model...")
    # load model
    model = load_fno_from_checkpoint(model_cls=model_class, save_folder=model_dir, save_name='best_model')
    model.to(device)


    # ------------------ Trainers ------------------ 
    if model_class == FNOtoScalar:
        trainer = CustomTrainer(model=model, 
                            n_epochs=epochs,
                            log_interval=log_interval,
                            logger=logger,
                            device=device, 
                            data_processor=DefaultDataProcessor(out_normalizer=target_transform).to(device),
                            eval_interval=1,
                            use_distributed=False,
                            verbose=True)
    else:
        trainer = PhysicsTrainer(model=model, 
                            n_epochs=epochs,
                            log_interval=log_interval,
                            logger=logger,
                            device=device, 
                            data_processor=DefaultDataProcessor(out_normalizer=target_transform).to(device),
                            eval_interval=1,
                            use_distributed=False,
                            verbose=True)

    # ------------------ Losses ------------------ 
    l2loss = LpLoss(d=1, p=2)
    l1loss = LpLoss(d=1, p=1)
    h1loss = H1Loss(d=1)
    mseloss = FNO_MSE()
    maeloss = FNO_MAE()
    relloss = FNO_MRE()
    physics_loss = PDEloss(reduction='mean', deriv_type='spectral_complex')

    train_loss = l2loss
    eval_losses = {"H1": h1loss, "L1": l1loss, "L2": l2loss, "MSE": mseloss, "MAE": maeloss, "MRE": relloss}


    # logger.info(f'### MODEL ###\n {model}')
    # logger.info(f'### OPTIMIZER ###\n {optimizer}')
    # logger.info(f'### SCHEDULER ###\n {scheduler}')
    # losses_str = "### LOSSES ###\n"
    # losses_str += f" * Train: {train_loss.name}\n"
    # losses_str += f" * Physics: {physics_loss.name}\n"
    # losses_str += " * Eval:\n"
    # losses_str += "\n".join([f"    - {k}: {v.name}" for k, v in eval_losses.items()])
    # logger.info(losses_str)
    # sys.stdout.flush()



    # ---------------------------------------- Evaluation ----------------------------------------
    # logger.info("Starting Evaluation...")
   
    preds = []
    targets = []

    trainer.model = trainer.model.to(trainer.device)
    if trainer.data_processor is not None and trainer.data_processor.device != trainer.device:
        trainer.data_processor = trainer.data_processor.to(trainer.device)
    
    # Set model to evaluation mode
    trainer.model.eval()
    if trainer.data_processor:
        trainer.data_processor.eval()

    errors = {f"{loss_name}": 0 for loss_name in eval_losses.keys()}

    print("Total: ", len(test_dataset))
    last_out = torch.zeros((batch_size, output_size)).to(device)
    n_outputs = 0
    trainer.n_samples = 0
    with torch.no_grad(): 
        for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
 
            # collect losses and predictions for a batch
            eval_step_losses, output = trainer.eval_one_batch(sample, eval_losses, return_output=True)

            for loss_name, val_loss in eval_step_losses.items():
                errors[f"{loss_name}"] += val_loss
            
            # store targets and predictions
            targets.append(sample["y"])
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

    # average losses
    for key in errors.keys():
        errors[key] /= trainer.n_samples

    losses = {}       
    for key in eval_losses:
        loss = eval_losses[key]
        losses.update({key: loss(preds, targets).item()/trainer.n_samples }) # FNO losses use sum reduction -> still need to average

    # logger.info(f"number of distinct outputs: {n_outputs}")
    print(f"number of distinct outputs: {n_outputs}")

    # ---------------------------------------- Store Inferences ----------------------------------------
    if save_inferences:
        df = pd.DataFrame()

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        frequncies = [100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000]
        for i, freq in enumerate(frequncies):
            df[f'targets_{freq}'] = targets[:, i]
            df[f'preds_{freq}'] = preds[:, i]
            df[f'mae_{freq}'] = np.abs(targets[:, i] - preds[:, i])
            df[f'mse_{freq}'] = (targets[:, i] - preds[:, i])**2
            df[f'mre_{freq}'] = np.abs((targets[:, i] - preds[:, i])) / (np.abs(targets[:, i]) + 1e-12) * 100

        df.to_csv(inferences_path, index=False)

    
    # Report losses
    msg = ""
    for metric, value in errors.items():
        if isinstance(value, float) or isinstance(value, torch.Tensor):
            msg += f"{metric}={value:.4e}, "
    print(msg)
    print(f'MSE Evaluation loss: {losses["MSE"]:.4e},   MAE Evaluation loss: {losses["MAE"]:.4e},   Relative Validation Loss: {losses["MRE"]:.4f}')
    logger.info(f'MSE Evaluation loss: {losses["MSE"]:.4e},   MAE Evaluation loss: {losses["MAE"]:.4e},   Relative Validation Loss: {losses["MRE"]:.4f}')