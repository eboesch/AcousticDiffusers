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
import matplotlib.pyplot as plt
import sys
import wandb

if __name__ == "__main__":

     timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

     # Training settings
     seed = 0
     epochs = 10
     batch_size = 32
     lr = 1e-3
     log_interval = 100
     save_model = False
     model_path = f"{timestamp}_model.pth"

     torch.manual_seed(seed)

     device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"

     data_dir = "mini_imagenet/"
     targets_path = "mini_imagenet/coeffs.csv"

     # Fetch dataset
     df = pd.read_csv(targets_path)
     processed_paths = list(set(df['path'].tolist()))

     transform_mode="pipeline"
     dataset = DictImageDataset(data_dir, targets_path, transform_mode=transform_mode, target_transform=target_pad)

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
     model = FNO(n_modes=(16, 16), hidden_channels=64,
                    in_channels=3, out_channels=1)

     # you can improve the previous models by simply using a Tucker Tensorized FNO with just a few parameters
     # model = TFNO(n_modes=(16, 16), hidden_channels=64,
     #                in_channels=3,
     #                out_channels=1,
     #                factorization='tucker',
     #                implementation='factorized',
     #                rank=0.05)
     model.to(device)

     # Create the trainer
     trainer = Trainer(model=model, 
                         n_epochs=epochs,
                         device=device, 
                         data_processor=DefaultDataProcessor().to(device),
                         wandb_log=True,
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

     train_loss = h1loss
     eval_losses = {"h1": h1loss, "l2": l2loss}


     print('\n### MODEL ###\n', model)
     print('\n### OPTIMIZER ###\n', optimizer)
     print('\n### SCHEDULER ###\n', scheduler)
     print('\n### LOSSES ###')
     print(f'\n * Train: {train_loss}')
     print(f'\n * Test: {eval_losses}')
     sys.stdout.flush()

     # train the model
     trainer.train(train_loader=train_loader,
               test_loaders={"test_data": test_loader},
               optimizer=optimizer,
               scheduler=scheduler,
               regularizer=False,
               training_loss=train_loss,
               eval_losses=eval_losses)


     """  
     # save the weights of the trained model:
     model.save_checkpoint(save_folder='./checkpoints/', save_name='example_fno')

     # load saved weights
     model = FNO.from_checkpoint(save_folder='./checkpoints/', save_name='example_fno')
     """
