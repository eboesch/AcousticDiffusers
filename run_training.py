import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from dataset import ImageDataset
from models.models import CNN, train, test
import pandas as pd



if __name__ == "__main__":
    # Training settings
    seed = 0
    epochs = 10
    batch_size = 64
    lr = 1e-3
    log_interval = 100
    save_model = False
    
    torch.manual_seed(seed)

    data_dir = "mini_imagenet/"
    targets_path = "mini_imagenet/coeffs.csv"

    df = pd.read_csv(targets_path)
    processed_paths = list(set(df['path'].tolist()))
    

    device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"

    # create dataset
    dataset = ImageDataset(processed_paths, data_dir, targets_path)

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
    model = CNN(input_shape=input_shape, output_size=output_size).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fct = nn.MSELoss()
    eval_loss_fct = nn.MSELoss()

    print("Model summary: ", model)
    

    # Training

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train(model, train_loader, loss_fct, optimizer, log_interval, device)
        test(val_loader, model, eval_loss_fct, device)
        # scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "model.pt")


    # Evaluation
    model = model.to(device)
    eval_loss_fct = nn.L1Loss()
    test_loss = test(test_loader, model, eval_loss_fct, device)