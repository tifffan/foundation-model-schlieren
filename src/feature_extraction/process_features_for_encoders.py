# process_features_for_encoders.py

from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import argparse
import matplotlib.pyplot as plt

def compute_explained_variance(features_flat):
    """
    Compute the cumulative explained variance ratio for PCA components.

    Args:
        features_flat (np.ndarray): Flattened input features with shape (num_samples, hidden_dim).

    Returns:
        np.ndarray: Cumulative explained variance ratio array.
    """
    hidden_dim = features_flat.shape[1]
    pca_full = PCA(n_components=min(hidden_dim, features_flat.shape[0]))
    pca_full.fit(features_flat)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    return cumulative_explained_variance

def reduce_features_by_pca(batch_features, reduced_feature_dim):
    """
    Perform PCA to reduce the dimensionality of features.

    Args:
        batch_features (np.ndarray): Input features with shape (batch_size, grid_size, grid_size, hidden_dim).
        reduced_feature_dim (int): Target dimensionality after PCA.

    Returns:
        np.ndarray: Reduced features with shape (batch_size, grid_size, grid_size, reduced_feature_dim).
    """
    batch_size, grid_size, _, hidden_dim = batch_features.shape
    features_flat = batch_features.reshape(-1, hidden_dim)  # Shape: (batch_size * grid_size * grid_size, hidden_dim)

    # Compute explained variance
    cumulative_explained_variance = compute_explained_variance(features_flat)

    # Perform PCA with reduced dimensions
    pca = PCA(n_components=reduced_feature_dim)
    reduced_flat = pca.fit_transform(features_flat)  # Shape: (batch_size * grid_size * grid_size, reduced_feature_dim)
    reduced_features = reduced_flat.reshape(batch_size, grid_size, grid_size, reduced_feature_dim)
    return reduced_features, cumulative_explained_variance

def main():
    parser = argparse.ArgumentParser(description="Process features for encoders.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input features file (numpy array).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the reduced features.')
    parser.add_argument('--reduced_dim', type=int, default=16, help='Target dimensionality after PCA.')
    parser.add_argument('--plot_explained_variance', action='store_true', help='Plot and save the explained variance curve.')
    args = parser.parse_args()
    
    # Ensure the output path includes a file name
    output_path = Path(args.output_path)
    if output_path.suffix != ".npy":
        print("Output path must include a file name.")
        return
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    features = np.load(args.input_path)  # Expected shape: (num_images, grid_size, grid_size, hidden_dim)
    print(f"Loaded features with shape: {features.shape}")

    batch_size, grid_size, _, hidden_dim = features.shape
    

    if args.plot_explained_variance:
        # Compute and save explained variance
        features_flat = features.reshape(-1, hidden_dim)  # Shape: (num_samples, hidden_dim)
        cumulative_explained_variance = compute_explained_variance(features_flat)
        explained_variance_path = Path(args.output_path).with_name(Path(args.output_path).stem + '_explained_variance.npy')
        np.save(explained_variance_path, cumulative_explained_variance)
        print(f"Saved cumulative explained variance to {explained_variance_path}")

        # Optionally, plot and save the explained variance curve
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
        plt.xlabel('Number of PCA Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance vs. Number of PCA Components')
        plt.grid(True)
        plot_path = Path(args.output_path).with_name(Path(args.output_path).stem + '_explained_variance.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved explained variance plot to {plot_path}")

    else:
        print("Skipping explained variance plot.")
        
    # Reduce features
    reduced_features, _ = reduce_features_by_pca(features, args.reduced_dim)
    print(f"Reduced features to shape: {reduced_features.shape}")

    # Save reduced features
    np.save(args.output_path, reduced_features)
    print(f"Saved reduced features to {args.output_path}")

if __name__ == "__main__":
    main()
