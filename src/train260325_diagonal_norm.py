# train.py
import argparse
import importlib.util
import os
import torch
import torch.optim as optim
import tqdm
from model import SimpleResNet1D
from loss import SupConLossv2
from torch.utils.data import DataLoader
from dataset import NPYDatasetv3Norm
import wandb
import numpy as np
import sys


# Check for existing checkpoints
# Function to extract epoch number from filename
def get_epoch_number(filename):
    # Assumes filename format: 'checkpoint_epoch_<number>.pt'
    return int(filename.split('_')[-1].split('.')[0])

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            features = model(inputs)
            features = features.unsqueeze(1)
            loss = criterion(features, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def main(config_path, run_time=0):
    config = load_config(config_path)
    
    #weight dir
    save_dir = os.path.join('..', 'weights', config.RUN_NAME, f"{run_time}")
    os.makedirs(save_dir, exist_ok=True)
    
    #sample test
    sample_list = config.SAMPLE_LIST
    dim = config.IN_FEATURES if hasattr(config, 'IN_FEATURES') else 51

    # Load dataset
    dataset1 = NPYDatasetv3Norm(
        csv_path=config.CSV_PATH, 
        dim=dim
    )
    testset = NPYDatasetv3Norm(
        csv_path=config.CSV_PATH3, 
        dim=dim
    )

    if config.NORMALIZE: 
        # we use test mean to do the normalization for both train and test, this is because testing set has all four phonemes. 
        global_means, global_stds = testset.get_means_stds()
        print(f"Global means: {global_means}, Global stds: {global_stds}")
        dataset1.set_means_stds(global_means, global_stds)
        testset.set_means_stds(global_means, global_stds)
    else: 
        print("No normalization applied to the dataset.")
        global_means, global_stds = None, None
    
    dataloader1 = DataLoader(dataset1, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize model, loss, optimizer
    model = SimpleResNet1D(out_features=config.OUT_FEATURES).to(config.DEVICE)
    similarity_config = config.SIMILARITY
    # criterion = SupConLossv2(temperature=config.TEMPERATURE,similarity=similarity_config)
    criterion = SupConLossv2(temperature=config.TEMPERATURE,similarity=similarity_config,v2=False)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Get and sort checkpoint files based on epoch number
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')],
        key=get_epoch_number
    )

    if checkpoint_files:
        latest_checkpoint = os.path.join(save_dir, checkpoint_files[-1])
        print(f"Loading checkpoint from {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint))
        last_epoch = get_epoch_number(checkpoint_files[-1])
            
    else:
        print("No checkpoint found, starting training from scratch.")
        last_epoch = 0 
         #test the sample before the training
        # sample_test_wrapper(sample_list,model,config.DEVICE,similarity_config,0)   


    start_epoch = last_epoch + 1
    for epoch in range(start_epoch, start_epoch+config.EPOCHS):
        epoch_loss = 0
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader1, desc=f'd1_Epoch {epoch}/{start_epoch+config.EPOCHS}')):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            features = model(inputs)
            features = features.unsqueeze(1)  # Add view dimension if needed
            loss = criterion(features, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            #wandb.log({"batch_loss": loss.item(), "epoch": epoch})
        avg_loss = epoch_loss / (len(dataloader1))
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        # wandb.log({"train_loss": avg_loss, "epoch": epoch})
        test_loss = evaluate(model, testloader, criterion, config.DEVICE)
        # wandb.log({"test_loss": test_loss, "epoch": epoch})
        
        # Save the latest checkpoint, overwriting the previous one
        checkpoint_path_latest = os.path.join(save_dir, 'checkpoint_latest.pt')
        torch.save(model.state_dict(), checkpoint_path_latest)

        # Save a checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path_epoch = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path_epoch)
        
        #test some samples
        # sample_test_wrapper(sample_list,model,config.DEVICE,similarity_config,epoch+1)
    # wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    config = load_config(args.config)
    run_times_start, run_times_end = config.RUN_TIMES_START, config.RUN_TIMES_END

    for run_time in range(run_times_start, run_times_end): 
        print(f"NOW TRAINING: RUN {run_time}")
        main(args.config, run_time)