"""Script for training an FNO-based model"""
import os
import sys
import random
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from neuralop.models import FNO, TFNO
from neuralop.training import AdamW
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.utils import count_model_params

from dataset import DictImageDataset, DictHmapDataset, normalize_target, default_img_transform, pipeline_transform
from models.fno import CustomTrainer, PhysicsTrainer, FNOtoScalar, FNO_MAE, FNO_MSE, FNO_MRE, FNO_CNN, load_fno_from_checkpoint
from utils import PDEloss, read_list



logger = logging.getLogger(__name__)

if __name__ == "__main__":

   timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
   model_dir = f"logs/{timestamp}/"
   if not os.path.exists(model_dir):
      os.makedirs(model_dir)

   logging.basicConfig(
      handlers=[logging.FileHandler(model_dir+"/logs.log")],
      level=logging.INFO,
      format="%(asctime)s | %(message)s",
      datefmt="%H:%M",
   )

   blurb = "Hmaps, complex spectral physics loss, mean reduction, data+target normalization, l2 training loss, no further changes"
   logger.info(
      f"--------------------------------------------\nStarting Script\n{blurb}\n--------------------------------------------"
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


   # ---------------------------------------- Dataset ----------------------------------------
   logger.info("Creating Dataset...")
   
   dataset_type = "hmap"  # "hmap" or "image"
   dataset_split = "pre_split"  # "pre_split", "random_split" or "mini"

   train_paths = read_list('train_paths.txt')
   val_paths = read_list('val_paths.txt')
   test_paths = read_list('test_paths.txt')


   train_ratio = 0.7
   val_ratio = 0.1
   test_ratio = 1-train_ratio-val_ratio

   # if not using the pre-defined data splits, overwrite the path lists
   if dataset_split == "mini":
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

   logger.info(f"Using {dataset_split} {dataset_type} Dataset with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")
   input_shape = list(train_dataset[0]["x"].shape)
   output_size = int(train_dataset[0]["y"].shape[0])
   logger.info(f"Input shape: {input_shape}")
   logger.info(f"Target size: {output_size}")


   # ------------------ Data Transforms ------------------   
   # data normalization transform
   # if (dataset_split == "pre_split") and (dataset_type=="hmaps"):
   #    # if using pre-defined dataset split, use pre-computed mean and std to save some time
   #    data_mean = [0.5404]
   #    data_std = [0.3486]
   # else:
   data_mean, data_std = train_dataset.calculate_data_stats() # compute mean and std of training set per channel
   transform.transforms.append(v2.Normalize(data_mean, data_std))
   train_dataset.set_transform(transform)
   val_dataset.set_transform(transform)
   test_dataset.set_transform(transform)

   # target normalization transform
   target_mean, target_std = train_dataset.calculate_target_stats() # compute mean and std of targets per channel
   target_transform = normalize_target(means=target_mean, stds=target_std)


   logger.info(f"Data transform:\n{transform}")
   logger.info(f"Target transform: {target_transform}")
   

   # ------------------ DataLoaders ------------------ 
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


   # ---------------------------------------- Model ----------------------------------------
   logger.info("Creating Model and Trainer...")
   model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=input_shape[0], out_channels=2*output_size, n_layers=4)
   # model = FNOtoScalar(in_channels=input_shape[0], output_size=output_size, hidden_channels=32)
   # model = FNO_CNN(input_shape=input_shape, output_size=output_size)
   model.to(device)
   
   
   # ------------------ Trainers ------------------ 
   # For FNOtoScalar Model use this trainer
   # trainer = CustomTrainer(model=model, 
   #                      n_epochs=epochs,
   #                      log_interval=log_interval,
   #                      logger=logger,
   #                      device=device, 
   #                      data_processor=DefaultDataProcessor(out_normalizer=target_transform).to(device),
   #                      eval_interval=1,
   #                      use_distributed=False,
   #                      verbose=True)

   # For FNO with phyiscs loss use this trainer
   trainer = PhysicsTrainer(model=model, 
                        n_epochs=epochs,
                        log_interval=log_interval,
                        logger=logger,
                        device=device, 
                        data_processor=DefaultDataProcessor(out_normalizer=target_transform).to(device),
                        eval_interval=1,
                        use_distributed=False,
                        verbose=True)


   optimizer = AdamW(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

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

   # ---------------------------------------- Training ----------------------------------------
   logger.info("Starting Training...")
   trainer.train(train_loader=train_loader,
            test_loaders={"val": val_loader},
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            training_loss=train_loss,
            physics_loss=physics_loss,
            lmbda=lmbda,
            eval_losses=eval_losses,
            save_best='val_MRE',
            save_dir=model_dir)

   # save the weights of the trained model:
   model.save_checkpoint(save_folder=model_dir, save_name='final_model')

   
   # ---------------------------------------- Evaluation ----------------------------------------
   logger.info("Starting Evaluation...")

   # load best model   
   model = load_fno_from_checkpoint(model_cls=model.__class__, save_folder=model_dir, save_name='best_model')

   trainer.evaluate_all(epoch=epochs,
                        eval_losses=eval_losses,
                        test_loaders={"test_data": test_loader})
     
   
