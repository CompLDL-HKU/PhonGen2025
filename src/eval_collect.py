# train.py
import argparse
import importlib.util
import os
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from model import SimpleResNet1D
from loss import SupConLoss
from torch.utils.data import DataLoader
from dataset import NPYDatasetInfoCollect
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

def sample_test_wrapper(sample_list,model,device,epoch):
    length = len(sample_list)
    left_index = 0
    for i in range(length//2):
        # Log start of processing pair
        #print(f"Processing pair {left_index}-{left_index+1}")
        left_name = sample_list[left_index].split('/')[-1].split('.')[0]
        right_name = sample_list[left_index+1].split('/')[-1].split('.')[0]
        #print(f"{left_name}*{right_name}")
        similarity = sample_test(sample_list[left_index], sample_list[left_index+1],model,device)
        wandb.log({f"{left_name}*{right_name}": similarity,"epoch": epoch})
        left_index = left_index + 2 # 0vs1, 2vs3, 4vs5

def sample_test(path1,path2,model,device):
    array1 = np.load(path1)
    array1_flat = array1.flatten()
    array1_input = array1_flat.reshape(1, 1, 51)
    tensor_input1 = torch.from_numpy(array1_input).float()
    tensor_input1 = tensor_input1.to(device)
    with torch.no_grad():
        feature1 = model(tensor_input1)

    array2 = np.load(path2)    
    array2_flat = array2.flatten() 
    array2_input = array2_flat.reshape(1, 1, 51)
    tensor_input2 = torch.from_numpy(array2_input).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_input2 = tensor_input2.to(device)
    with torch.no_grad():
        feature2 = model(tensor_input2)
    
    feat1 = feature1.cpu().numpy().flatten()
    norm_feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feature2.cpu().numpy().flatten()
    norm_feat2 = feat2 / np.linalg.norm(feat2)
    cosine_sim = np.dot(norm_feat1, norm_feat2)
    return cosine_sim.item()


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

def batch_meta_to_df(batch_meta: dict) -> pd.DataFrame:
    """
    Convert a batch metadata dict (lists + tensors) into a pandas DataFrame.
    """
    clean_dict = {}
    for k, v in batch_meta.items():
        if torch.is_tensor(v):
            # 1D tensor -> list of scalars
            clean_dict[k] = v.detach().cpu().tolist()
        else:
            # already list of strings, bools, etc.
            clean_dict[k] = v
    return pd.DataFrame(clean_dict)

@torch.no_grad()
def evaluate_collect_outputs(
    model,
    data_loader,
    device,
    npy_path="outputs.npy",
    csv_path="outputs_meta.csv",
    to_float32=True
):
    """
    Runs the model over data_loader, stacks all output vectors into a single .npy file,
    and writes a CSV with metadata + the index of each vector in the .npy array.

    Assumptions:
      - Each batch item is a dict with {input_key: tensor, ...metadata...}.
      - model(inputs) returns a 2D tensor [B, D] (if 1D, it will be unsqueezed).
    """
    model.eval()
    model.to(device)

    all_vecs = []
    csv_frames = []   # list to hold batch-level DataFrames

    for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(data_loader)):
        inputs = inputs.to(device)

        # forward pass -> vectors
        vec = model(inputs)  # expected [B, D]
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        vec = vec.detach().cpu()
        if to_float32 and vec.dtype != torch.float32:
            vec = vec.float()

        # append vectors
        all_vecs.append(vec)

        # append metadata DataFrame
        batch_df = batch_meta_to_df(targets)
        csv_frames.append(batch_df)

    if not all_vecs:
        raise RuntimeError("No vectors collected. Check your loader and input_key.")
    mat = torch.cat(all_vecs, dim=0).numpy()
    np.save(npy_path, mat)

    # save CSV
    csv_df = pd.concat(csv_frames, ignore_index=True)
    csv_df.to_csv(csv_path, index=False)

    print(f"Saved vectors: {mat.shape} -> {os.path.abspath(npy_path)}")
    print(f"Saved metadata rows: {len(csv_df)} -> {os.path.abspath(csv_path)}")

    return


def main(config_path):
    config = load_config(config_path)
    
    #weight dir
    save_dir = os.path.join(config.PROJECT_BASE_PATH, 'weights', config.RUN_NAME)
    os.makedirs(save_dir, exist_ok=True)
    eval_save_dir = os.path.join(config.PROJECT_BASE_PATH, 'eval_outputs', config.RUN_NAME)
    os.makedirs(eval_save_dir, exist_ok=True)

    # Load dataset
    dataset = NPYDatasetInfoCollect(
        csv_path=config.CSV_PATH,
        base_path=config.NPY_BASE_PATH,
        max_samples=config.MAX_SAMPLES,
        contain_all=True
    )
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model, loss, optimizer
    model = SimpleResNet1D(out_features=config.OUT_FEATURES).to(config.DEVICE)

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
        print("No checkpoint found, end running evaluation.")
        sys.exit(0)

    # Run evaluation and collect outputs
    evaluate_collect_outputs(
        model=model,
        data_loader=dataloader,
        device=config.DEVICE,
        npy_path=os.path.join(eval_save_dir, f"vec_{last_epoch}.npy"),
        csv_path=os.path.join(eval_save_dir, f"meta_{last_epoch}.csv"),
        to_float32=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    main(args.config)