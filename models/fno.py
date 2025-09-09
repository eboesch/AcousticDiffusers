import sys
import os
import warnings
from tqdm import tqdm
from timeit import default_timer
from pathlib import Path
from logging import Logger
from typing import Union, Dict, List, Tuple, Callable, Type

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from neuralop.training import Trainer
from neuralop.losses import LpLoss
from neuralop.models import FNO, TFNO
from neuralop.models.base_model import BaseModel

from models.cnn import CNN
from utils import MRE, compute_diffusion_coeffs


class FNOtoScalar(BaseModel):
    """
    FNO + Global Pooling model architecture

    The input is processed by an FNO block, then each channel is pooled to 
    a single value, before being fed through a linear layer.
    """

    def __init__(
        self,
        in_channels: int,
        output_size: int,
        hidden_channels: int = 32,
        n_layers: int = 4,
        n_modes: int = 16,
        **kwargs,
    ):
        """
        Constructs DictHmapDataset instance.

        Args:
            in_channels (int): How many channels the input has.
            output_size (int): The desired size of the output.
            hidden_channels (int): The number of hidden channels in the FNO.
            n_layers (int): How many Fourier layers the FNO should have.
            n_modes (int): How many modes to keep in the Fourier Layers.
        """
        super().__init__()
        self.fno = FNO(
            n_modes=(n_modes, n_modes),
            in_channels=in_channels,  # image input
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )
        self.pool = nn.AdaptiveMaxPool2d(1)  # integrate over domain
        self.fc = nn.Linear(hidden_channels, output_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (batch, 1, H, W)
        x = self.fno(x)  # (batch, hidden, H, W)
        x = self.pool(x)  # (batch, hidden, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, hidden)
        out = self.fc(x)  # (batch, n_freqs)
        return out


class FNO_CNN(CNN):
    """
    Wrapper of the CNN class from models.py to make it compatible with the FNO training structure.
    """

    def __init__(
        self,
        input_shape,
        output_size: int,
        cnn_out_channels_1: int = 16,
        cnn_out_channels_2: int = 32,
        cnn_kernel_size: int = 3,
        cnn_stride: int = 1,
        cnn_padding: int = 0,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        linear_size: int = 64,
    ):
        super().__init__(
            input_shape=input_shape,
            output_size=output_size,
            cnn_out_channels_1=cnn_out_channels_1,
            cnn_out_channels_2=cnn_out_channels_2,
            cnn_kernel_size=cnn_kernel_size,
            cnn_stride=cnn_stride,
            cnn_padding=cnn_padding,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            linear_size=linear_size,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = super().forward(x)
        return out


class CustomTrainer(Trainer):
    """
    A customized Trainer class for training neural-operators on given datasets, 
    that logs training progress to a logger. Most of the framework is as in neuralop's Trainer.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        log_interval: int,
        logger: Logger,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor: nn.Module = None,
        eval_interval: int = 1,
        use_distributed: bool = False,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        log_interval : int
            interval at which training loss should be logged
        logger : Logger
            Logger instance to write logs to
        device : torch.device, or str 'cpu' or 'cuda'
        mixed_precision : bool, default is False
            whether to use torch.autocast to compute mixed precision
        data_processor : DataProcessor class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        eval_interval : int, default is 1
            how frequently to evaluate model and log training stats
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        """

        super().__init__(
            model=model,
            n_epochs=n_epochs,
            wandb_log=False,
            device=device,
            mixed_precision=mixed_precision,
            data_processor=data_processor,
            eval_interval=eval_interval,
            log_output=False,
            use_distributed=use_distributed,
            verbose=verbose,
        )

        self.log_interval = log_interval
        self.logger = logger

    def train(
        self,
        train_loader: DataLoader,
        test_loaders: Dict[str, DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        regularizer=None,
        training_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        eval_losses: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        save_every: int = None,
        save_best: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Trains the given model on the given dataset.

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
        if hasattr(training_loss, "reduction"):
            if training_loss.reduction == "mean":
                warnings.warn(
                    f"{training_loss.reduction=}. This means that the loss is "
                    "initialized to average across the batch dim. The Trainer "
                    "expects losses to sum across the batch dim."
                )

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        # accumulated wandb metrics
        # self.wandb_epoch_metrics = None

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
            assert (
                self.save_best in metrics
            ), f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float("inf")
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        # log dataset sizes
        if self.verbose:
            print(f"Training on {len(train_loader.dataset)} samples")
            print(
                f"Validating on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on dataset {[name for name in test_loaders]}."
            )
            sys.stdout.flush()
        self.logger.info(f"Training on {len(train_loader.dataset)} samples")
        self.logger.info(
            f"Validating on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
            f"         on dataset {[name for name in test_loaders]}."
        )

        # ------------------------------------ Start Training ------------------------------------
        for epoch in range(self.start_epoch, self.n_epochs):
            self.logger.info(f"Epoch {epoch+1}")
            train_err, avg_loss, avg_lasso_loss, epoch_train_time = self.train_one_epoch(
                epoch, train_loader, training_loss
            )
            epoch_metrics = dict(
                train_err=train_err, avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss, epoch_train_time=epoch_train_time
            )

            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch, eval_losses=eval_losses, test_loaders=test_loaders)

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

    def train_one_epoch(
        self, epoch: int, train_loader: DataLoader, training_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Tuple[float, float, float, float]:
        """
        Train_one_epoch trains self.model on train_loader for one epoch and returns training metrics.

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        training_loss: training.losses function
            cost function to minimize

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0

        # set model and data processor to train mode
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        running_loss = 0.0

        # track number of training examples in batch
        self.n_samples = 0

        # iterate through batches
        for idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):

            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

            # report training loss
            running_loss += loss.item()
            if idx % self.log_interval == self.log_interval - 1:
                self.logger.info(f"[{idx + 1:5d}/{len(train_loader):>5d}] Loss: {running_loss / self.log_interval:.4e}")
                running_loss = 0.0

        # LR scheduler step
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        elif self.scheduler is not None:
            self.scheduler.step()

        # compute time for epoch
        epoch_train_time = default_timer() - t1

        # average
        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]

        # log training stats
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )
        self.logger.info(f"[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}")

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def log_eval(self, epoch: int, eval_metrics: dict):
        """
        Log outputs from evaluation on all test loaders to stdout and logger.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader

        """
        # values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4e}, "
            # if self.wandb_log:
            #     values_to_log[metric] = value

        msg = f"Eval: " + msg[:-2]  # cut off last comma+space
        self.logger.info(msg)
        print(msg)
        sys.stdout.flush()

        # if self.wandb_log:
        #     wandb.log(data=values_to_log,
        #               step=epoch+1,
        #               commit=True)


ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]


class PhysicsTrainer(CustomTrainer):
    """
    A customized Trainer class for training neural-operators on given datasets, 
    that uses an additional physics loss and that logs training progress to a logger. 
    Most of the framework is as in neuralop's Trainer.
    """

    def train(
        self,
        train_loader: DataLoader,
        test_loaders: Dict[str, DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        regularizer=None,
        training_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        physics_loss: Callable[[torch.Tensor, List], torch.Tensor] = None,
        eval_losses: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        lmbda: float = 1.0,
        save_every: int = None,
        save_best: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
    ):
        """
        Trains the given model on the given dataset.

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
        physics_loss: Callable[[torch.Tensor, List], torch.Tensor] function, optional, default is None
            physics informed cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        lmbad: float, optional, default is 1
            regularization parameter that controls the weight of the physics loss. 
            Higher values mean higher weight for physics.
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
        self.physics_loss = physics_loss
        self.lmbda = lmbda
        return super().train(
            train_loader,
            test_loaders,
            optimizer,
            scheduler,
            regularizer=regularizer,
            training_loss=training_loss,
            eval_losses=eval_losses,
            save_every=save_every,
            save_best=save_best,
            save_dir=save_dir,
            resume_from_dir=resume_from_dir,
        )

    def train_one_batch(
        self,
        idx: int,
        sample: Dict[str, torch.Tensor],
        training_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Run one batch of input through model and return training loss on outputs.

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch
        training_loss: training.losses function
            cost function to minimize

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()

        # preprossess/load data
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].shape[0]

        # forward pass
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)

        if self.epoch == 0 and idx == 0 and self.verbose:
            print(f"Raw outputs of shape {out.shape}")
            self.logger.info(f"Raw outputs of shape {out.shape}")

        loss = 0.0

        # sample['x'].shape: batch, n_channels, H, W
        # sample['y'].shape: batch, n_freqs
        # out.shape: batch, n_freqs, H, W

        # compute loss
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                if self.physics_loss is not None:
                    # compute PDE loss for each frequency band
                    pde_l = self.physics_loss(out, ALL_FREQ_BANDS)
                    loss += self.lmbda * pde_l

                # compute coeffs
                coeff_pred = compute_diffusion_coeffs(out)

                # postprocess outputs
                if self.data_processor is not None:
                    coeff_pred, sample = self.data_processor.postprocess(coeff_pred, sample)

                loss += training_loss(coeff_pred, **sample)
        else:
            if self.physics_loss is not None:
                # compute PDE loss for each frequency band
                pde_l = self.physics_loss(out, ALL_FREQ_BANDS)
                loss += self.lmbda * pde_l

            # compute coeffs
            coeff_pred = compute_diffusion_coeffs(out)

            # postprocess outputs
            if self.data_processor is not None:
                coeff_pred, sample = self.data_processor.postprocess(coeff_pred, sample)

            loss += training_loss(coeff_pred, **sample)

        if self.regularizer:
            loss += self.regularizer.loss

        return loss

    def eval_one_batch(
        self,
        sample: Dict[str, torch.Tensor],
        eval_losses: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        return_output: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        Run inference on one batch and return eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False

        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        # preprossess/load data
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)
        out = compute_diffusion_coeffs(out)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        eval_step_losses = {}

        # compute all eval losses
        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None


# Losses ----------------------------------------------------------------------------------------


class FNO_MSE(nn.Module):
    """
    Wrapper for MSE loss that corresponds with FNO training architecture, 
    i.e. averaging first over spatial dimensions and then reducing over batch and channel dimension.

    Attributes:
        d (int): The number of spatial dimensions of the input.
        reduction (str): Determines how the loss should be reduced. Either 'sum' or 'mean'.
        loss (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function being wrapped, here MSE loss.
    """

    def __init__(self, d: int = 1, reduction: str = "sum"):
        """
        Constructs an instance of FNO_MSE.

        Args:
            d (int): The number of spatial dimensions of the input.
            reduction (str): Determines how the loss should be reduced. Either 'sum' or 'mean'.
        """
        super().__init__()
        self.d = d
        self.reduction = reduction
        self.loss = nn.MSELoss(reduction="none")

    @property
    def name(self):
        return f"MSE_{self.d}D_{self.reduction}_reduction_loss"

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the loss between prediction and target tensors, averaging first 
        over spatial dimensions and then reducing over batch and channel dimension.

        Args:
            y_pred (torch.Tensor): Predicted tensor.
            y (torch.Tensor): Target tensor of the same shape as `y_pred`.

        Returns:
            torch.Tensor: Scalar tensor representing the reduced loss.
        """
        # y: Shape (B, C, D1, D2, ... , Dd)

        # flatten spatial dimensions of prediction and target
        y_pred_flat = torch.flatten(y_pred, start_dim=-self.d)
        y_flat = torch.flatten(y, start_dim=-self.d)  # Shape (B, C, N = D1*...*Dd)

        # compute elementwise loss between prediction and target
        elementwise_loss = self.loss(y_pred_flat, y_flat)  # Shape (B, C, N)

        # average over last (i.e. flattened spatial) dimension
        batchwise_loss = torch.mean(elementwise_loss, dim=-1)  # Shape (B, C)

        # sum resp. average over remaining dimensions
        if self.reduction == "sum":
            loss = torch.sum(batchwise_loss)
        else:
            loss = torch.mean(batchwise_loss)

        return loss.squeeze()


class FNO_MAE(FNO_MSE):
    """
    Wrapper for MAE loss that corresponds with FNO training architecture, 
    i.e. averaging first over spatial dimensions and then reducing over batch and channel dimension.

    Attributes:
        d (int): The number of spatial dimensions of the input.
        reduction (str): Determines how the loss should be reduced. Either 'sum' or 'mean'.
        loss (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function being wrapped, here MAE loss.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="none")

    @property
    def name(self):
        return f"MAE_{self.d}D_{self.reduction}_reduction_loss"


class FNO_MRE(FNO_MSE):
    """
    Wrapper for MSR loss that corresponds with FNO training architecture, 
    i.e. averaging first over spatial dimensions and then reducing over batch and channel dimension.

    Attributes:
        d (int): The number of spatial dimensions of the input.
        reduction (str): Determines how the loss should be reduced. Either 'sum' or 'mean'.
        loss (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function being wrapped, here MRE loss.
    """

    def __init__(self):
        super().__init__()
        self.loss = MRE(reduction="none")

    @property
    def name(self):
        return f"MRE_{self.d}D_{self.reduction}_reduction_loss"


def load_fno_from_checkpoint(model_cls: Type[nn.Module], save_folder: str, save_name: str = "best_model") -> nn.Module:
    """
    Loads an FNO-based model of class `model_class` from `save_folder`.

    Args:
        model_cls (Type[nn.Module]): The class of the model being loaded.
        save_folder (str): Path to the location the model was saved.
        save_name (str): Name of the model. Default is 'best_model'.

    Returns:
        (nn-Module): Instance of the model_class with weights restored from the saved model.
    """

    metadata_filepath = os.path.join(save_folder, save_name + "_metadata.pkl")

    # Force old behavior torch.load behavior
    init_kwargs = torch.load(metadata_filepath, weights_only=False)

    version = init_kwargs.pop("_version")
    if hasattr(model_cls, "_version") and version != model_cls._version:
        print(version)
        warnings.warn(
            f"Checkpoint saved for version {version} of model {model_cls._name} but current code is version {model_cls._version}"
        )

    if "args" in init_kwargs:
        init_args = init_kwargs.pop("args")
    else:
        init_args = []
    instance = model_cls(*init_args, **init_kwargs)

    # Load state dict
    state_dict_path = os.path.join(save_folder, save_name + "_state_dict.pt")
    state = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    instance.load_state_dict(state)

    return instance
