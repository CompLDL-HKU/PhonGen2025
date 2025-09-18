import numpy as np
from model import SimpleResNet1D
from dataset import NPYDataset
from torch.utils.data import DataLoader
import argparse
import importlib.util
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main(config_path):
    config = load_config(config_path)

    #load df to get dictionary
    df = pd.read_csv("/mnt/storage/qisheng/github/PhonGen2025/plots/sampled_dataset.csv")
    label_dict = dict(zip(df['label_idx'], df['label']))

    # Load dataset
    dataset = NPYDataset(
        csv_path="/mnt/storage/qisheng/github/PhonGen2025/plots/sampled_dataset.csv",
        base_path=config.NPY_BASE_PATH,
        max_samples=config.MAX_SAMPLES,
        train_only=True
    )
    testset = NPYDataset(
        csv_path="/mnt/storage/qisheng/github/PhonGen2025/plots/sampled_dataset.csv",
        base_path=config.NPY_BASE_PATH,
        max_samples=config.MAX_SAMPLES,
        train_only=False
    )
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize model
    model = SimpleResNet1D(out_features=config.OUT_FEATURES).to(config.DEVICE)
    model.load_state_dict(torch.load("/mnt/storage/qisheng/github/PhonGen2025/weights/BS32_LR1E4_FEAT128_TEMP007/checkpoint_epoch_200.pt"))
    features_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
         
        for inputs, labels in tqdm(dataloader, desc="traindata"):
            #data shape: (batch_size, channels, length)
            features = model(inputs.to(config.DEVICE))  # shape: (batch_size, 128)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy().astype(int))
        
         
        for inputs, labels in tqdm(testloader, desc="testdata"):
            # data shape: (batch_size, channels, length)
            features = model(inputs.to(config.DEVICE))  # shape: (batch_size, 128)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy().astype(int))
       
    # Concatenate all features and labels

        features = np.vstack(features_list)  # shape: (total_samples, 128)
        labels = np.concatenate(labels_list)
        labels = [label_dict.get(x) for x in labels]
        
        #TO DO
        #this part should be added to config settings
        selected_feat_idx = []
        selected_label = []
        selected_item = ['itsi','ici','itci','isi','LtsL','LtcL','LcL','LsL']
        for idx in range(len(labels)):
            if labels[idx] in selected_item:
                selected_feat_idx.append(idx)
                selected_label.append(labels[idx])
        features = features[selected_feat_idx]        
        labels = selected_label

        unique_labels = list(set(labels))
        num_classes =len(unique_labels)
        # Use the recommended approach
        color_map = matplotlib.colormaps['hsv']
        # Generate colors by sampling the colormap
        color_list = [color_map(i / num_classes) for i in range(num_classes)]

        # Create a mapping from label to color
        label_to_color = {label: color_list[i] for i, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in labels]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings = tsne.fit_transform(features)

        plt.figure(figsize=(8,6))
        plt.scatter(embeddings[:,0], embeddings[:,1], c=colors, alpha=0.7)

        # Create legend
        handles = [mpatches.Patch(color=label_to_color[label], label=label) for label in unique_labels]
        plt.legend(handles=handles, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("t-SNE visualization of model features")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig('/mnt/storage/qisheng/github/PhonGen2025/plots/tsne_plot2_selected.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    main(args.config)