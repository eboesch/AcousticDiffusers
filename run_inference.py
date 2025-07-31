import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from dataset import ImageDataset, normalize_target
from models.models import CNN, train, test, MRE
import pandas as pd
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm


# logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # logging.basicConfig(
    #     handlers=[logging.FileHandler(f"logs/{timestamp}_logs.log")],
    #     level=logging.INFO,
    #     format="%(asctime)s | %(message)s",
    #     datefmt="%H:%M",
    # )
    # logger.info(
    #     "--------------------------------------------\nStarting Script\n--------------------------------------------"
    # )

    # Training settings
    seed = 0
    epochs = 10
    batch_size = 16#64
    lr = 1e-3
    log_interval = 100
    save_inferences = True
    # model_path = f"logs/2025-07-23-14-47-03_model.pth"
    # model_path = f"logs/2025-07-29-18-03-15_model.pth" # fixed, MSE loss
    # model_path = f"logs/2025-07-29-18-34-43_model.pth" # fixed, MAE loss
    # model_path = f"logs/2025-07-29-18-57-46_model.pth" # fixed, MRE loss
    model_path = f"logs/2025-07-31-10-30-14_model.pth" # fixed, MAE loss, normalization
    # model_path = f"logs/2025-07-24-10-53-19_model.pth"
    # inferences_path = f"logs/{timestamp}_inferences.pkl"
    inferences_path = f"logs/{timestamp}_inferences.csv"



    # logger.info(f"lr: {lr}")
    # logger.info(f"batch_size: {batch_size}")
    # logger.info(f"epochs: {epochs}")
    
    torch.manual_seed(seed)

    data_dir = "mini_imagenet/"
    targets_path = "mini_imagenet/coeffs.csv"

    df = pd.read_csv(targets_path)
    processed_paths = list(set(df['path'].tolist()))
    

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
    # logger.info("Using %s device", device)

    # create dataset
    # logger.info("Creating Dataset...")
    transform_mode="pipeline"
    # logger.info("Using %s transform", transform_mode)
    dataset = ImageDataset(processed_paths, data_dir, targets_path, transform_mode=transform_mode)
    mean, std = dataset.calculate_stats()
    target_transform = normalize_target(means=mean, stds=std)
    dataset.set_target_transform(target_transform)

    # split data into train & test set

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 1-train_ratio-val_ratio

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_ratio, val_ratio, test_ratio], generator=generator)
    input_shape = list(train_dataset[0][0].shape)
    output_size = int(train_dataset[0][1].shape[0])

    # test_dataset = train_dataset

    # Set up Dataloaders 
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Set up model
    # logger.info("Loading Model...")
    model = CNN(input_shape=input_shape, output_size=output_size).to(device)
    # model = CNN(input_shape=input_shape, output_size=output_size, cnn_out_channels_1=32, cnn_out_channels_2=64).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True))  # load the saved best model
    model = model.to(device)

    mse_eval_loss_fct = nn.MSELoss()
    mae_eval_loss_fct = nn.L1Loss()
    rel_eval_loss_fct = MRE()
    loss_fcts = [mse_eval_loss_fct, mae_eval_loss_fct, rel_eval_loss_fct]

    # logger.info("Starting Evaluation...")
   
    model.eval()
    losses = np.zeros(len(loss_fcts))

    
    # features = np.zeros((len(test_dataset), input_shape[1], input_shape[2], input_shape[3]))
    preds = np.zeros((len(test_dataset), output_size))
    targets = np.zeros((len(test_dataset), output_size))

    print("total", len(test_dataset))
    index_start = 0
    last_out = torch.zeros((batch_size, output_size)).to(device)
    n_outputs = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            # print("start", index_start)
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if target_transform:
                output = target_transform.inverse(output)
                target = target_transform.inverse(target)
            
            index_end = index_start + data.shape[0]
            # features[index_start:index_end, :, :, :] = data.numpy()
            targets[index_start:index_end, :] = target.squeeze(1).cpu().numpy()
            preds[index_start:index_end, :] = output.squeeze(1).cpu().numpy()


            out_batch_size = output.shape[0]
            if (output - last_out[:out_batch_size, :]).any():
                n_outputs += 1
                # print(f"number of distinct outputs: {n_outputs}")
                # print(f"output: {output}")
            last_out = output

            for idx, loss in enumerate(loss_fcts):
                losses[idx] += loss(output, target).item()

            index_start = index_end
            
    losses /= len(test_loader)

    # logger.info(f"number of distinct outputs: {n_outputs}")
    print(f"number of distinct outputs: {n_outputs}")


    if save_inferences:
        df = pd.DataFrame()

        frequncies = [100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000]
        for i, freq in enumerate(frequncies):
            df[f'targets_{freq}'] = targets[:, i]
            df[f'preds_{freq}'] = preds[:, i]
            df[f'mae_{freq}'] = np.abs(targets[:, i] - preds[:, i])
            df[f'mse_{freq}'] = (targets[:, i] - preds[:, i])**2
            df[f'mre_{freq}'] = np.abs((targets[:, i] - preds[:, i])) / (np.abs(targets[:, i]) + 1e-8) * 100

        df.to_csv(inferences_path, index=False)


        # df['targets'] = list(targets)
        # df['preds'] = list(preds)
        # df['mae'] = list(np.abs(targets - preds))
        # df['mse'] = list((targets - preds)**2)
        # df['mre'] = list(np.abs((targets - preds)) / (np.abs(targets) + 1e-8) * 100)

        # df.to_pickle(inferences_path)
        


    # logger.info(f'MSE Evaluation loss: {losses[0]:.4e},   MAE Evaluation loss: {losses[1]:.4e},   Relative Validation Loss: {losses[2]:.4f}')
    print(f'MSE Evaluation loss: {losses[0]:.4e},   MAE Evaluation loss: {losses[1]:.4e},   Relative Validation Loss: {losses[2]:.4f}')