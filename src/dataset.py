import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYDataset(Dataset):
    def __init__(self, csv_path, base_path, max_samples=5000, train_only=True):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        #df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data

        # Filter by train/test split
        if train_only:
            df = df[df['train'] == 'yes']
        else:
            df = df[df['train'] == 'no']

        # Limit to max_samples
        df = df.iloc[:max_samples]

        # Store DataFrame
        self.df = df
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.base_path, row['path'])
        data = np.load(npy_path)  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D
        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)

        label = row['label_idx']
        return data_tensor, label


class NPYDatasetInfoCollect(Dataset): 
    """
    Dataset class that returns data along with metadata information.
    For EVLUATION only. 
    """
    def __init__(self, csv_path, base_path, max_samples=5000, train_only=True, contain_all=False):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        #df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data

        # Filter by train/test split
        # Here modified to optionally load all data
        if contain_all: 
            pass
        else: 
            if train_only:
                df = df[df['train'] == 'yes']
            else:
                df = df[df['train'] == 'no']

        # Limit to max_samples
        df = df.iloc[:max_samples]

        # Store DataFrame
        self.df = df
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.base_path, row['path'])
        data = np.load(npy_path)  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D
        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)

        info = {
            "uid": row["uid"],
            "path": row["path"],
            "cog": row["cog"],
            "fri_dur": row["fri_dur"],
            "word": row["word"],
            "consonant": row["consonant"],
            "vowel": row["vowel"],
            "train": row["train"],
            "label": row["label"],       # string or raw label
            "label_idx": row["label_idx"]  # numerical class index
        }

        return data_tensor, info