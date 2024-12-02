from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

from src.feature_extraction.feature_extraction_utils import *

def get_rgb_features(image_features):
    """
    Perform PCA to reduce a single image's features to RGB dimensions.

    Args:
        image_features (np.ndarray): Input features with shape (grid_size, grid_size, hidden_dim).

    Returns:
        np.ndarray: Reduced features with shape (grid_size, grid_size, 3).
    """
    print(f"image_features shape: {image_features.shape}")
    grid_size, _, hidden_dim = image_features.shape
    features_flat = image_features.reshape(-1, hidden_dim)  # Shape: (grid_size * grid_size, hidden_dim)
    pca = PCA(n_components=3)
    reduced_flat = pca.fit_transform(features_flat)  # Shape: (grid_size * grid_size, 3)
    rgb_features = reduced_flat.reshape(grid_size, grid_size, 3)
    return np.asarray(rgb_features, dtype=np.float32)


def visualize_image_and_feature(image_dir, features_path, num_samples=None, model_keyword="model"):
    """
    Visualize pairs of original images and their corresponding RGB feature maps.

    Args:
        image_dir (str): Directory containing the original images.
        features_path (str): Path to the .npy file containing features.
        num_samples (int, optional): Number of samples to visualize. If None, visualize all.
        model_keyword (str): String identifying the model.
    """
    # Load features
    features = np.load(features_path)
    print(f"Loaded features with shape: {features.shape}")

    # Load image paths
    image_paths = sorted(list(Path(image_dir).glob("*.jpg")) +
                         list(Path(image_dir).glob("*.png")) +
                         list(Path(image_dir).glob("*.jpeg")) +
                         list(Path(image_dir).glob("*.bmp")))

    if features.shape[0] != len(image_paths):
        raise ValueError("Number of features does not match number of images.")
    
    image_paths = sort_images_by_filename(image_paths)

    # Determine indices
    if num_samples is None:
        indices = np.arange(len(image_paths))  # All pairs
    else:
        indices = np.random.choice(len(image_paths), size=min(num_samples, len(image_paths)), replace=False)
    
    selected_features = features[indices]
    selected_images = [np.array(Image.open(image_paths[i]).convert("RGB")) for i in indices]

    # Plotting
    num_samples = len(indices)  # Adjust num_samples for plotting all pairs
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]  # Ensure axes is iterable

    for i, (img, feat) in enumerate(zip(selected_images, selected_features)):
        # Reduce feature to RGB dimensions
        rgb_features = get_rgb_features(feat)
        
        # Normalize RGB features
        rgb_features = (rgb_features - rgb_features.min()) / (rgb_features.max() - rgb_features.min() + 1e-8)

        # Convert to valid RGB format (0-255) for Matplotlib
        rgb_features = (rgb_features * 255).astype(np.uint8)

        # Plot Original Image
        axes[i][0].imshow(img)
        axes[i][0].axis('off')
        axes[i][0].set_title(f"Original Image {indices[i]}")

        # Plot RGB Feature Map
        axes[i][1].imshow(rgb_features)
        axes[i][1].axis('off')
        axes[i][1].set_title(f"RGB Feature Map for Image {indices[i]}")

    plt.tight_layout()

    # Save with timestamp and model_keyword and dataset_keyword
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_name = f"{model_keyword}_{dataset_keyword}_image_feature_pairs_{timestamp}.png"
    plt.savefig(figure_name)
    plt.close(fig)  # Close figure to free memory

    print(f"Visualization saved as {figure_name}")


if __name__ == "__main__":
    # Example usage
    # model_keyword = "poseidon"
    # dataset_keyword = "sim"
    # visualize_image_and_feature("data/raw/frames_test", "data/features/frames_test/poseidon_features.npy", num_samples=None, model_keyword=model_keyword)
    
    # model_keyword = "poseidon"
    # dataset_keyword = "exp"
    # visualize_image_and_feature("data/raw/exp_frames_test", "data/features/exp_frames_test/poseidon_features.npy", num_samples=None, model_keyword=model_keyword)
    
    model_keyword = "dust3r_224"
    dataset_keyword = "sim"
    visualize_image_and_feature("data/raw/frames_test", "data/features/frames_test/dust3r_224_features.npy", num_samples=10, model_keyword=model_keyword)
    
    # model_keyword = "dust3r_512"
    # dataset_keyword = "sim"
    # visualize_image_and_feature("data/raw/frames_test", "data/features/frames_test/dust3r_512_features.npy", num_samples=10, model_keyword=model_keyword)
