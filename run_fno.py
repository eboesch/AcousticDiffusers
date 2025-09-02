import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import neuralop
from neuralop.models import FNO, TFNO
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.training import Trainer, AdamW
from neuralop.losses.data_losses import LpLoss, H1Loss, PointwiseQuantileLoss #, HdivLoss, MSELoss
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.utils import count_model_params
import pandas as pd
import logging
from datetime import datetime
from dataset import DictImageDataset, DictHmapDataset, target_pad, normalize_target, default_img_transform, pipeline_transform
from custom_fno import CustomTrainer, PhysicsTrainer, FNOtoScalar, FNO_MAE, FNO_MSE, FNO_MRE, FNO_CNN
from utils import PDEloss, read_list

import matplotlib.pyplot as plt
import sys
import os
import random
import wandb


logger = logging.getLogger(__name__)

if __name__ == "__main__":

   timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

   logging.basicConfig(
      handlers=[logging.FileHandler(f"logs/{timestamp}_logs.log")],
      level=logging.INFO,
      format="%(asctime)s | %(message)s",
      datefmt="%H:%M",
   )
   logger.info(
      "--------------------------------------------\nStarting Script\nComplex pressures, hmaps\n--------------------------------------------"
   )

   # Training settings
   seed = 0
   epochs = 10
   batch_size = 32
   lr = 8e-3
   weight_decay = 1e-4
   lmbda = 1
   log_interval = 100
   save_model = True
   model_path = f"logs/{timestamp}_model.pth"

   logger.info(f"lr: {lr}")
   logger.info(f"weight_decay: {weight_decay}")
   logger.info(f"lambda: {lmbda}")
   logger.info(f"batch_size: {batch_size}")
   logger.info(f"epochs: {epochs}")
   

   torch.manual_seed(seed)
   random.seed(seed)

   device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
   logger.info("Using %s device", device)

   data_dir = "mini_imagenet/"
   hmap_dir = 'hmaps_png/'
   targets_path = "mini_imagenet/coeffs.csv"


   # Create dataset
   logger.info("Creating Dataset...")
   
   dataset_type = "hmap"  # "hmap" or "image"
   dataset_split = "mini"  # "pre_split", "random_split" or "mini"

   train_paths = read_list('train_paths.txt')
   val_paths = read_list('val_paths.txt')
   test_paths = read_list('test_paths.txt')


   train_ratio = 0.7
   val_ratio = 0.1
   test_ratio = 1-train_ratio-val_ratio

   if dataset_split == "mini":
      sample_paths = random.sample(train_paths+val_paths+test_paths, 1000)
      train_paths = sample_paths[:int(len(sample_paths)*train_ratio)]
      val_paths = sample_paths[int(len(sample_paths)*train_ratio): int(len(sample_paths)*(train_ratio+val_ratio))+1]
      test_paths = sample_paths[int(len(sample_paths)*(train_ratio+val_ratio))+1:]
   elif dataset_split == "random_split":
      all_paths = random.shuffle(train_paths+val_paths+test_paths)
      train_paths = all_paths[:int(len(all_paths)*train_ratio)]
      val_paths = all_paths[int(len(all_paths)*train_ratio): int(len(all_paths)*(train_ratio+val_ratio))+1]
      test_paths = all_paths[int(len(all_paths)*(train_ratio+val_ratio))+1:]


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

   logger.info(f"Using {dataset_split} {dataset_type} Dataset with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")
   input_shape = list(train_dataset[0]["x"].shape)
   output_size = int(train_dataset[0]["y"].shape[0])
   logger.info(f"Input shape: {input_shape}")
   logger.info(f"Target size: {output_size}")

   # Set up transforms
    
   # data normalization transform
   data_mean, data_std = train_dataset.calculate_data_stats()
   transform.transforms.append(v2.Normalize(data_mean, data_std)) #normalize_target(means=data_mean, stds=data_std)
   train_dataset.set_transform(transform)
   val_dataset.set_transform(transform)
   test_dataset.set_transform(transform)

   # target normalization transform
   target_mean, target_std = train_dataset.calculate_target_stats()
   target_transform = normalize_target(means=target_mean, stds=target_std)
   train_dataset.set_target_transform(target_transform)
   val_dataset.set_target_transform(target_transform)
   test_dataset.set_target_transform(target_transform)

   logger.info(f"Data transform:\n{transform}")
   logger.info(f"Target transform: {target_transform}")
   

   # Set up Dataloaders 
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


   # create FNO model
   logger.info("Creating Model and Trainer...")
   model = TFNO(n_modes=(16, 16), hidden_channels=32, in_channels=input_shape[0], out_channels=output_size)
   # model = FNOtoScalar(n_freqs=17, hidden_channels=32)
   # model = FNO_CNN(input_shape=input_shape, output_size=output_size)

   # you can improve the previous models by simply using a Tucker Tensorized FNO with just a few parameters
   # model = TFNO(n_modes=(16, 16), hidden_channels=64,
   #                in_channels=3,
   #                out_channels=1,
   #                factorization='tucker',
   #                implementation='factorized',
   #                rank=0.05)
   model.to(device)
   
   
   # Create the trainer
   # trainer = CustomTrainer(model=model, 
   #                      n_epochs=epochs,
   #                      log_interval=log_interval,
   #                      logger=logger,
   #                      device=device, 
   #                      data_processor=DefaultDataProcessor().to(device),
   #                      eval_interval=1,
   #                      use_distributed=False,
   #                      verbose=True)
   trainer = PhysicsTrainer(model=model, 
                        n_epochs=epochs,
                        log_interval=log_interval,
                        logger=logger,
                        device=device, 
                        data_processor=DefaultDataProcessor().to(device),
                        eval_interval=1,
                        use_distributed=False,
                        verbose=True)


   # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
   optimizer = AdamW(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

   l2loss = LpLoss(d=1, p=2)
   l1loss = LpLoss(d=1, p=1)
   h1loss = H1Loss(d=1)
   mseloss = FNO_MSE()
   maeloss = FNO_MAE()
   relloss = FNO_MRE()
   physics_loss = PDEloss(reduction='mean', deriv_type='spectral')

   train_loss = l2loss
   eval_losses = {"H1": h1loss, "L1": l1loss, "L2": l2loss, "MSE": mseloss, "MAE": maeloss, "MRE": relloss}


   logger.info(f'### MODEL ###\n {model}')
   logger.info(f'### OPTIMIZER ###\n {optimizer}')
   logger.info(f'### SCHEDULER ###\n {scheduler}')
   losses_str = "### LOSSES ###\n"
   losses_str += f" * Train: {train_loss.name}\n"
   losses_str += f" * Physics: {physics_loss.name}\n"
   losses_str += " * Eval:\n"
   losses_str += "\n".join([f"    - {k}: {v.name}" for k, v in eval_losses.items()])
   logger.info(losses_str)
   sys.stdout.flush()

   # train the model
   logger.info("Starting Training...")
   trainer.train(train_loader=train_loader,
            test_loaders={"val_data": val_loader},
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            training_loss=train_loss,
            physics_loss=physics_loss,
            lmbda=lmbda,
            eval_losses=eval_losses)



   logger.info("Starting Evaluation...")
   trainer.evaluate_all(epoch=epochs,
                         eval_losses=eval_losses,
                         test_loaders={"test_data": test_loader})
     
   # save the weights of the trained model:
   model.save_checkpoint(save_folder='logs/', save_name='example_fno')

   # load saved weights
   # model = FNO.from_checkpoint(save_folder='./checkpoints/', save_name='example_fno')
   
