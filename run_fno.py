import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
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
from dataset import DictImageDataset, target_pad
from custom_fno import CustomTrainer, FNOtoScalar
import matplotlib.pyplot as plt
import sys
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
        "--------------------------------------------\nStarting Script\n--------------------------------------------"
     )

     # Training settings
     seed = 0
     epochs = 10
     batch_size = 32
     lr = 1e-3
     log_interval = 100
     save_model = False
     model_path = f"logs/{timestamp}_model.pth"

     logger.info(f"lr: {lr}")
     logger.info(f"batch_size: {batch_size}")
     logger.info(f"epochs: {epochs}")
    

     torch.manual_seed(seed)

     device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
     logger.info("Using %s device", device)

     data_dir = "mini_imagenet/"
     targets_path = "mini_imagenet/coeffs.csv"

     # Fetch dataset
     # df = pd.read_csv(targets_path)
     # processed_paths = list(set(df['path'].tolist()))

     logger.info("Creating Dataset...")
     transform_mode="pipeline"
     logger.info("Using %s transform", transform_mode)
     target_transform_mode="pad"
     logger.info("Using %s target transform", target_transform_mode)
     dataset = DictImageDataset(data_dir, targets_path, transform_mode=transform_mode, target_transform=target_pad)
     logger.info(f"Using Image Dataset with {len(dataset)} samples")

     # Split Dataset
     train_ratio = 0.7
     val_ratio = 0.1
     test_ratio = 1-train_ratio-val_ratio

     generator = torch.Generator().manual_seed(seed)
     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_ratio, val_ratio, test_ratio], generator=generator)

     # Set up Dataloaders 
     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


     """
     # load a pre-created dataset
     train_loader, test_loaders, data_processor = load_darcy_flow_small(
          n_train=1000, batch_size=32,
          test_resolutions=[32], n_tests=[100],
          test_batch_sizes=[32],
     )
     """

     # create FNO model
     logger.info("Creating Model and Trainer...")
     model = TFNO(n_modes=(16, 16), hidden_channels=64, in_channels=3, out_channels=1)

     # you can improve the previous models by simply using a Tucker Tensorized FNO with just a few parameters
     # model = TFNO(n_modes=(16, 16), hidden_channels=64,
     #                in_channels=3,
     #                out_channels=1,
     #                factorization='tucker',
     #                implementation='factorized',
     #                rank=0.05)
     model.to(device)
     
     
     # Create the trainer
     trainer = CustomTrainer(model=model, 
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
                                lr=8e-3,
                                weight_decay=1e-4)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

     l2loss = LpLoss(d=2, p=2)
     h1loss = H1Loss(d=2)

     train_loss = l2loss
     eval_losses = {"h1": h1loss, "l2": l2loss}


     logger.info(f'### MODEL ###\n {model}')
     logger.info(f'### OPTIMIZER ###\n {optimizer}')
     logger.info(f'### SCHEDULER ###\n {scheduler}')
     logger.info('### LOSSES ###')
     logger.info(f'\n * Train: {train_loss}')
     logger.info(f'\n * Test: {eval_losses}')
     sys.stdout.flush()

     # train the model
     logger.info("Starting Training...")
     trainer.train(train_loader=train_loader,
               test_loaders={"val_data": val_loader},
               optimizer=optimizer,
               scheduler=scheduler,
               regularizer=False,
               training_loss=train_loss,
               eval_losses=eval_losses)



     logger.info("Starting Evaluation...")
     trainer.evaluate_all(epoch=epochs,
                         eval_losses=eval_losses,
                         test_loaders={"test_data": test_loader})
     """  
     # save the weights of the trained model:
     model.save_checkpoint(save_folder='./checkpoints/', save_name='example_fno')

     # load saved weights
     model = FNO.from_checkpoint(save_folder='./checkpoints/', save_name='example_fno')
     """
