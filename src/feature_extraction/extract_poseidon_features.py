import torch
from transformers import AutoModel
from PIL import Image
from pathlib import Path
import numpy as np
import os
import argparse
import logging
import time

from feature_extraction_utils import *

def preprocess_image(image_path, grid_size=(128, 128)):
    """
    Preprocesses a grayscale image and creates 4 identical channels for model input.
    Args:
        image_path (str): Path to the image file.
        grid_size (tuple): Target size for resizing the image.
    Returns:
        torch.Tensor: Preprocessed tensor with 4 identical channels.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize(grid_size, Image.Resampling.LANCZOS)
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)
    image_tensor = image_tensor.repeat(4, 1, 1)  # Repeat to create 4 channels
    return image_tensor

# def save_catalog(image_paths, output_path, model_keyword="poseidon", dataset_keyword="sim"):
#     """
#     Saves a catalog mapping images to their feature indices.
#     Args:
#         image_paths (list): List of image paths.
#         output_path (str): Path to save the catalog file.
#         model_keyword (str): Keyword for the model.
#         dataset_keyword (str): Keyword for the dataset.
#     """
#     catalog_name = f"catalog_{model_keyword}_{dataset_keyword}.csv"
#     catalog_path = output_path.parent / catalog_name
#     with open(catalog_path, mode='w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Index", "Image_Path"])
#         for idx, image_path in enumerate(image_paths):
#             writer.writerow([idx, str(image_path)])
#     logging.info(f"Catalog saved to {catalog_path}")

# def save_metadata(output_dir, model_name, data_size, feature_dim, model_keyword="poseidon", dataset_keyword="sim"):
#     """
#     Saves metadata about the feature extraction.
#     Args:
#         output_path (str): Path to save the metadata file.
#         model_name (str): Name of the model used.
#         data_size (int): Total number of images processed.
#         feature_dim (tuple): Shape of the extracted features per image.
#     """
#     metadata_name = f"metadata_{model_keyword}_{dataset_keyword}.json"
#     metadata_path = output_dir / metadata_name
#     metadata = {
#         "model_name": model_name,
#         "data_size": data_size,
#         "feature_dim": feature_dim
#     }
#     with open(metadata_path, 'w') as metafile:
#         json.dump(metadata, metafile, indent=4)
#     logging.info(f"Metadata saved to {metadata_path}")
    

def extract_features(image_dir, output_path, batch_size=32, grid_size=(128, 128), device="cpu"):
    """
    Extracts features using the Poseidon-B model.
    Args:
        image_dir (str): Path to the directory containing images.
        output_path (str): Path to save the extracted features (numpy file).
        batch_size (int): Number of images to process in a batch.
        grid_size (tuple): Grid size for resizing images (128x128).
        device (str): Device to run the model on ("cpu" or "cuda").
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    start_time = time.time()

    # Ensure the output path includes a file name
    output_path = Path(output_path)
    if output_path.suffix != ".npy":
        output_path = output_path / "poseidon_features.npy"
    
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the Poseidon-L model
    model_name = "camlab-ethz/Poseidon-L"
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Collect image paths
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    logging.info(f"Found {len(image_paths)} images in {image_dir}")
    
    # Sort by run number and frame number
    image_paths = sort_images_by_filename(image_paths)

    all_features = []
    batch_images = []

    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            inputs = preprocess_image(image_path, grid_size).unsqueeze(0).to(device)  # Add batch dimension
            batch_images.append(inputs)

            # Process in batches
            if len(batch_images) == batch_size or i == len(image_paths) - 1:
                batch_tensor = torch.cat(batch_images, dim=0)  # Create a batch
                logging.info(f"Processing batch {i // batch_size + 1}")
                outputs = model(batch_tensor)
                features = outputs.last_hidden_state
                features = reshape_vector_feature_to_square_grid(features)
                all_features.append(features.cpu().numpy())
                batch_images = []  # Reset batch

    # Save features to a file
    all_features = np.concatenate(all_features, axis=0)
    np.save(output_path, all_features)  # Save features as .npy file
    logging.info(f"Extracted features saved to {output_path}")

    # Save catalog and metadata
    save_catalog(image_paths, output_dir, model_keyword="poseidon", dataset_keyword="exp")
    save_metadata(output_dir, model_name, len(image_paths), all_features.shape[1:], model_keyword="poseidon", dataset_keyword="exp")

    end_time = time.time()
    total_time = end_time - start_time
    time_per_frame = total_time / len(image_paths)
    logging.info(f"Total time: {total_time:.2f} seconds, Time per frame: {time_per_frame:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using Poseidon-B model.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save extracted features (numpy file).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size for resizing images.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (e.g., 'cpu', 'cuda').")
    args = parser.parse_args()

    extract_features(
        image_dir=args.image_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        grid_size=[args.grid_size, args.grid_size],
        device=args.device,
    )

# python src/feature_extraction/extract_poseidon_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/frames_test --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/frames_test/poseidon_features.npy

# python src/feature_extraction/extract_poseidon_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/sim_frames_672_432 --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/sim_frames_672_432/poseidon_features.npy

# python src/feature_extraction/extract_poseidon_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/exp_frames_672_432 --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/exp_frames_672_432/poseidon_features.npy

# python src/feature_extraction/extract_poseidon_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/exp_frames_test --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/exp_frames_test/poseidon_features.npy
