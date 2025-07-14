import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from dataset import ImageDataset
from models.models import CNN, train, test, MRE
import pandas as pd
import logging
from datetime import datetime


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
    batch_size = 16#64
    lr = 1e-3
    log_interval = 100
    save_model = False
    model_path = f"{timestamp}_model.pth"


    logger.info(f"lr: {lr}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"epochs: {epochs}")
    
    torch.manual_seed(seed)

    data_dir = "mini_imagenet/"
    targets_path = "mini_imagenet/coeffs.csv"

    df = pd.read_csv(targets_path)
    processed_paths = list(set(df['path'].tolist()))
    

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    # create dataset
    logger.info("Creating Dataset...")
    transform_mode="pipeline"
    logger.info("Using %s transform", transform_mode)
    dataset = ImageDataset(processed_paths, data_dir, targets_path, transform_mode=transform_mode)

    # split data into train & test set

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 1-train_ratio-val_ratio

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_ratio, val_ratio, test_ratio], generator=generator)
    input_shape = list(train_dataset[0][0].shape)
    output_size = int(train_dataset[0][1].shape[0])

    # train_dataset = ImageDataset(train_data, train_label)
    # val_dataset = ImageDataset
    # test_dataset = ImageDataset(test_data, test_label)

    # Set up Dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Set up model
    logger.info("Creating Model...")
    model = CNN(input_shape=input_shape, output_size=output_size).to(device)
    # model = CNN(input_shape=input_shape, output_size=output_size, cnn_out_channels_1=32, cnn_out_channels_2=64).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # loss_fct = nn.MSELoss()
    loss_fct = nn.L1Loss()
    mse_eval_loss_fct = nn.MSELoss()
    mae_eval_loss_fct = nn.L1Loss()
    rel_eval_loss_fct = MRE()

    logger.info("Model: %s", model)
    logger.info("Using MAE training loss")


    # logger.info("Using StepLR scheduler")
    # step_size = 4
    # gamma = 0.1
    # logger.info(f"step_size: {step_size}")
    # logger.info(f"gamma: {gamma}")
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    logger.info("Using ReduceLROnPlateau scheduler")
    patience = 1
    gamma = 0.1
    threshold = 1e-3
    logger.info(f"patience: {patience}")
    logger.info(f"factor: {gamma}")
    logger.info(f"threshold: {threshold}")
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=gamma, patience=patience)

    # Training
    logger.info("Starting Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        logger.info(f"Epoch {epoch+1}")
        
        train(model, train_loader, loss_fct, optimizer, log_interval, logger, device)
        val_losses = test(val_loader, model, [mse_eval_loss_fct, mae_eval_loss_fct, rel_eval_loss_fct], device)

        logger.info(f'MAE Validation Loss: {val_losses[0]:.4e},   MSE Validation Loss: {val_losses[1]:.4e},   Relative Validation Loss: {val_losses[2]:.4f}')
        print(f'MAE Validation Loss: {val_losses[0]:.4e},   MSE Validation Loss: {val_losses[1]:.4e},   Relative Validation Loss: {val_losses[2]:.4f}')
    
        logger.info(f"Last learning rate: {scheduler.get_last_lr()}")
        scheduler.step(val_losses[2])

    if save_model:
        torch.save(model.state_dict(), model_path)


    logger.info("Starting Evaluation...")
    # Evaluation
    # if save_model:
    #     model.load_state_dict(torch.load(model_path, weights_only=True))  # load the saved best model
    #     model = model.to(device)
    test_losses = test(test_loader, model, [mse_eval_loss_fct, mae_eval_loss_fct, rel_eval_loss_fct], device)
    logger.info(f'MAE Evaluation loss: {test_losses[0]:.4e},   MSE Evaluation loss: {test_losses[1]:.4e},   Relative Validation Loss: {test_losses[2]:.4f}')
    print(f'MAE Evaluation loss: {test_losses[0]:.4e},   MSE Evaluation loss: {test_losses[1]:.4e},   Relative Validation Loss: {test_losses[2]:.4f}')