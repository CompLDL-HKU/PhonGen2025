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
    def __init__(self, csv_path, base_path, max_samples=5000, train_only=True, contain_all=False, minmax=False):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
            contain_all (bool): If True, load all data regardless of train/test split.
        """
        self.minmax = minmax
        self.min= [0, 0, 0, 0, 0, 0, 0, 200, 0, 72, 180, 153, 99, 81, 2461.5, 931.5, 393.3, 3000, 70, 180, 0.45, 0.9, 45, 54, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 72, 180, 153, 99, 81, 2461.5, 931.5, 393.3]

        self.max= [1, 1, 1, 1, 1, 1, 1, 400, 400, 88, 220, 187, 121, 99, 3709.2, 3037.1, 859.1, 8000, 200, 220, 0.55, 1.1, 55, 66, 400, 400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 400, 400, 88, 220, 187, 121, 99, 3709.2, 3037.1, 859.1]
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
        # df = df.iloc[:max_samples]

        # Store DataFrame
        self.df = df
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # npy_path = os.path.join(self.base_path, row['path'])
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D
        # Apply Min–Max normalization if enabled
        if getattr(self, "minmax", False):
            min_vals = np.array(self.min)
            max_vals = np.array(self.max)
            data_tensor = (data_tensor - min_vals) / (max_vals - min_vals + 1e-8)  # avoid /0
            data = np.clip(data, 0.0, 1.0)  # keep clean range

        data_tensor = torch.tensor(data_tensor, dtype=torch.float32).unsqueeze(0)  # add channel dimension (1, N)

        info = {
            "uid": row["uid"],
            "path": row["path"],
            "cog": row["cog"],
            "fri_dur": row["fri_dur"],
            # "voicing": row["voicing"],    # commented for diagonal
            "word": row["word"],
            "consonant": row["consonant"],
            "vowel": row["vowel"],
            "train": row["train"],
            # "label": row["label"],       # string or raw label
            # "label_idx": row["label_idx"]  # numerical class index
        }

        return data_tensor, info

# Oh since you gave me a way to try, I want to first adapt the old code to account for the new contrasts: there are 4 consonants: s c ts tc, and 8 vowels


class NPYDatasetv2(Dataset):
    def __init__(self, csv_path):
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


        # Store DataFrame
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D
        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)

        label = row['train']
        return data_tensor, label
    

class NPYDatasetv2Norm(Dataset):
    def __init__(self, csv_path):
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

        self.min= [0, 0, 0, 0, 0, 0, 0, 200, 0, 72, 180, 153, 99, 81, 2461.5, 931.5, 393.3, 3000, 70, 180, 0.45, 0.9, 45, 54, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 72, 180, 153, 99, 81, 2461.5, 931.5, 393.3]
        self.max= [1, 1, 1, 1, 1, 1, 1, 400, 400, 88, 220, 187, 121, 99, 3709.2, 3037.1, 859.1, 8000, 200, 220, 0.55, 1.1, 55, 66, 400, 400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 400, 400, 88, 220, 187, 121, 99, 3709.2, 3037.1, 859.1]
        self.min_vals = np.array(self.min)
        self.max_vals = np.array(self.max)

        # Store DataFrame
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D

        # Apply Min–Max normalization
        data_tensor = (data_tensor - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)  # avoid /0
        data = np.clip(data, 0.0, 1.0)  # keep clean range

        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)

        label = row['train']
        return data_tensor, label