from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings
from tqdm import tqdm

import torch
from torch.cuda import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

from neuralop.training import Trainer
import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from neuralop.training.training_state import load_training_state, save_training_state

from neuralop.models import FNO
from models.models import MRE

class FNOtoScalar(nn.Module):
    def __init__(self, n_freqs, hidden_channels=32):
        super().__init__()
        self.fno = FNO(
            n_modes=(16, 16),
            in_channels=3,           # image input
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # integrate over domain
        self.fc = nn.Linear(hidden_channels, n_freqs)

    def forward(self, x, **kwargs):
        # x: (batch, 1, H, W)
        latent = self.fno(x)                 # (batch, hidden, H, W)
        pooled = self.pool(latent)           # (batch, hidden, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, hidden)
        out = self.fc(pooled)                # (batch, n_freqs)
        return out


class CustomTrainer(Trainer):
    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        log_interval: int, 
        logger,
        device: str='cpu',
        mixed_precision: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,
    ):
        """
        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is False
            whether to log results to wandb
        device : torch.device, or str 'cpu' or 'cuda'
        mixed_precision : bool, default is False
            whether to use torch.autocast to compute mixed precision
        data_processor : DataProcessor class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        eval_interval : int, default is 1
            how frequently to evaluate model and log training stats
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        """

        super().__init__(model=model,
        n_epochs=n_epochs,
        wandb_log=False,
        device=device,
        mixed_precision=mixed_precision,
        data_processor=data_processor,
        eval_interval=eval_interval,
        log_output=log_output,
        use_distributed=use_distributed,
        verbose=verbose)

        self.log_interval = log_interval
        self.logger = logger

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        save_every: int=None,
        save_best: int=None,
        save_dir: Union[str, Path]="./ckpt",
        resume_from_dir: Union[str, Path]=None,
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here. 

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, 
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        
        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders
            
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)
        
        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, 'reduction'):
            if training_loss.reduction == "mean":
                warnings.warn(f"{training_loss.reduction=}. This means that the loss is "
                              "initialized to average across the batch dim. The Trainer "
                              "expects losses to sum across the batch dim.")

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        # if self.use_distributed and dist.is_initialized():
        #     device_id = dist.get_rank()
        #     self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert self.save_best in metrics,\
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float('inf')
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f'Training on {len(train_loader.dataset)} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
        self.logger.info(f'Training on {len(train_loader.dataset)} samples')
        self.logger.info(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
              f'         on resolutions {[name for name in test_loaders]}.')
            
        
        for epoch in range(self.start_epoch, self.n_epochs):
            self.logger.info(f"Epoch {epoch+1}")
            train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                  self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time
            )
            
            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch,
                                                eval_losses=eval_losses,
                                                test_loaders=test_loaders)

                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics
    
    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        running_loss = 0.0
        
        # track number of training examples in batch
        self.n_samples = 0

        for idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

            running_loss += loss.item()
            if idx % self.log_interval == self.log_interval-1:    
                self.logger.info(f'[{idx + 1:5d}/{len(train_loader):>5d}] Loss: {running_loss / self.log_interval:.4e}')
                running_loss = 0.0

        # LR scheduler step
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        # time for epoch
        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr
            )

        self.logger.info(f"[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}")
        # if avg_lasso_loss is not None:
        #     msg += f", avg_lasso={avg_lasso_loss:.4f}"

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time
    
    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader
       
        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4e}, "
            if self.wandb_log:
                values_to_log[metric] = value       
        
        msg = f"Eval: " + msg[:-2] # cut off last comma+space
        self.logger.info(msg)
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=True)
            



class FNO_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, y_pred, y, **kwargs):
        return self.loss(y_pred, y)

class FNO_MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, y_pred, y, **kwargs):
        return self.loss(y_pred, y)
    
class FNO_MRE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = MRE()

    def forward(self, y_pred, y, **kwargs):
        return self.loss(y_pred, y)