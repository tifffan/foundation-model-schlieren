# latent_space_visualization.py

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from dataset import FeatureDataset
from autoencoders import UNetAutoencoder, UNetVAE
import random
import re
import os

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For CUDA deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_hyperparams_from_path(checkpoint_path):
    # Get the parent directory containing the hyperparameters
    parent_dir = os.path.dirname(checkpoint_path)
    hyperparams_dir = os.path.basename(parent_dir)
    # Now parse the hyperparameters from hyperparams_dir

    # Regular expression pattern to match hyperparameters
    pattern = (
        r"r_(?P<random_seed>\d+)"
        r"_ld_(?P<latent_dim>\d+)"
        r"_conv_(?P<conv_channels>(?:\d+_?)+)"
        r"_lr_(?P<learning_rate>[\d.e-]+)"
        r"_bs_(?P<batch_size>\d+)"
        r"_wd_(?P<weight_decay>[\d.e-]+)"
        r"(?:_beta_(?P<beta>[\d.e-]+))?"
    )
    match = re.search(pattern, hyperparams_dir)
    if match:
        hyperparams = match.groupdict()
        # Convert strings to appropriate types
        hyperparams['random_seed'] = int(hyperparams['random_seed'])
        hyperparams['latent_dim'] = int(hyperparams['latent_dim'])
        hyperparams['conv_channels'] = [int(ch) for ch in hyperparams['conv_channels'].split('_')]
        hyperparams['learning_rate'] = float(hyperparams['learning_rate'])
        hyperparams['batch_size'] = int(hyperparams['batch_size'])
        hyperparams['weight_decay'] = float(hyperparams['weight_decay'])
        if hyperparams.get('beta') is not None:
            hyperparams['beta'] = float(hyperparams['beta'])
        else:
            hyperparams['beta'] = None
        return hyperparams
    else:
        raise ValueError("Checkpoint path does not match expected pattern for hyperparameter extraction.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Post-processing script for visualizing embeddings.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file (numpy .npy format).')
    parser.add_argument('--model_keyword', type=str, default='unet_ae_3blk', help='Keyword to define model type.')
    parser.add_argument('--subsample_fraction', type=float, default=None, help='Fraction of data to subsample.')
    parser.add_argument('--subsample_size', type=int, default=None, help='Number of data samples to subsample.')
    parser.add_argument('--n_components', type=int, default=3, help='Number of PCA components to compute (2 or 3).')
    parser.add_argument('--no_tsne', action='store_true', help='If set, t-SNE visualizations will be skipped.')
    args = parser.parse_args()

    # Parse hyperparameters from checkpoint path
    hyperparams = parse_hyperparams_from_path(args.checkpoint)
    print("Extracted hyperparameters from checkpoint path:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    # Set random seed
    set_seed(hyperparams['random_seed'])

    # Load dataset
    dataset = FeatureDataset(
        data_path=args.data_path,
        subsample_fraction=args.subsample_fraction,
        subsample_size=args.subsample_size
    )

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=0)

    # Define model based on model_keyword
    conv_channels = hyperparams['conv_channels']
    is_vae = 'vae' in args.model_keyword.lower()

    # Instantiate model
    if 'unet_ae' in args.model_keyword:
        model = UNetAutoencoder(
            n_channels=dataset.features.shape[1],
            conv_channels=conv_channels,
            latent_dim=hyperparams['latent_dim'],
            linear_hidden_dim=256,
            bilinear=False
        )
    elif 'unet_vae' in args.model_keyword:
        model = UNetVAE(
            n_channels=dataset.features.shape[1],
            conv_channels=conv_channels,
            latent_dim=hyperparams['latent_dim'],
            linear_hidden_dim=256,
            bilinear=False
        )
    else:
        raise ValueError(f"Unknown model_keyword: {args.model_keyword}")

    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # Depending on how the state dict was saved, adjust the key accordingly
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Get embeddings
    embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            # For VAE, the encoder returns mu and logvar
            if is_vae:
                mu, logvar = model.encoder(inputs)
                z = mu  # Use the mean as the embedding
            else:
                z = model.encoder(inputs)
            embeddings.append(z.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    # Flatten original features for PCA/t-SNE
    original_features = dataset.features.view(len(dataset), -1).numpy()

    # Apply PCA to embeddings
    n_components = args.n_components
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3 for visualization purposes.")
    pca_embeddings = PCA(n_components=n_components)
    embeddings_pca = pca_embeddings.fit_transform(embeddings)

    # Apply PCA to original features
    pca_features = PCA(n_components=n_components)
    features_pca = pca_features.fit_transform(original_features)

    # Explained variance ratio
    explained_variance_embeddings = pca_embeddings.explained_variance_ratio_
    explained_variance_features = pca_features.explained_variance_ratio_
    print(f"Explained variance ratio of the first {n_components} components (embeddings): {explained_variance_embeddings}")
    print(f"Explained variance ratio of the first {n_components} components (original features): {explained_variance_features}")

    # Visualization functions
    def plot_2d(data, title, filename):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"2D plot saved as {filename}")

    def plot_3d(data, title, filename):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"3D plot saved as {filename}")

    # Visualization of PCA embeddings
    if n_components == 2:
        plot_2d(
            embeddings_pca,
            title='PCA of Embeddings (2D)',
            filename=f"embedding_pca_2d_ld_{hyperparams['latent_dim']}_{args.model_keyword}.png"
        )
    else:
        plot_3d(
            embeddings_pca,
            title='PCA of Embeddings (3D)',
            filename=f"embedding_pca_3d_ld_{hyperparams['latent_dim']}_{args.model_keyword}.png"
        )

    # Visualization of PCA features
    if n_components == 2:
        plot_2d(
            features_pca,
            title='PCA of Original Features (2D)',
            filename=f"feature_pca_2d_{args.model_keyword}.png"
        )
    else:
        plot_3d(
            features_pca,
            title='PCA of Original Features (3D)',
            filename=f"feature_pca_3d_{args.model_keyword}.png"
        )

    # Apply t-SNE to embeddings
    if not args.no_tsne:
        tsne_embeddings = TSNE(n_components=2, random_state=hyperparams['random_seed']).fit_transform(embeddings)
        plot_2d(
            tsne_embeddings,
            title='t-SNE of Embeddings (2D)',
            filename=f"embedding_tsne_2d_ld_{hyperparams['latent_dim']}_{args.model_keyword}.png"
        )

        # Apply t-SNE to original features
        tsne_features = TSNE(n_components=2, random_state=hyperparams['random_seed']).fit_transform(original_features)
        plot_2d(
            tsne_features,
            title='t-SNE of Original Features (2D)',
            filename=f"feature_tsne_2d_{args.model_keyword}.png"
        )

    else:
        print("t-SNE visualizations skipped as per the '--no_tsne' flag.")

if __name__ == '__main__':
    main()
