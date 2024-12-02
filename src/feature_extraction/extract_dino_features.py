import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import os
from pathlib import Path
import argparse
import logging
import time

from feature_extraction_utils import *

def preprocess_image(image_path, processor):
    """
    Preprocesses an image for input into the DINOv2 model.
    Args:
        image_path (str): Path to the image file.
        processor: Hugging Face image processor.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def extract_features(image_dir, output_path, batch_size=32, device="cpu"):
    """
    Extracts DINOv2 features from a directory of images.
    Args:
        image_dir (str): Path to the directory containing images.
        output_path (str): Path to save the extracted features (numpy file).
        batch_size (int): Number of images to process in a batch.
        device (str): Device to run the model on ("cpu" or "cuda").
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    start_time = time.time()
    
    # Ensure the output path includes a file name
    output_path = Path(output_path)
    if output_path.suffix != ".npy":
        output_path = output_path / "dino_features.npy"
    
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the DINOv2 model and processor
    model_name = "facebook/dinov2-large"
    model = AutoModel.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Collect image paths
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    # Sort by run number and frame number
    image_paths = sort_images_by_filename(image_paths)

    all_features = []
    batch_images = []

    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            inputs = preprocess_image(image_path, processor)
            batch_images.append(inputs)

            # Process in batches
            if len(batch_images) == batch_size or i == len(image_paths) - 1:
                # Concatenate batch inputs
                batch = {k: torch.cat([d[k] for d in batch_images]).to(device) for k in batch_images[0]}
                print(f"Processing batch {i // batch_size + 1}")
                outputs = model(**batch)
                # print(f"outputs shape: {outputs.shape}")
                features = outputs.last_hidden_state #.mean(dim=1)  # Average over patches
                
                features = features[:, 1:, :]
                
                features = reshape_vector_feature_to_square_grid(features)
                
                print(f"features shape: {features.shape}")
                all_features.append(features.cpu().numpy())
                batch_images = []  # Reset batch

    # Save features to a file
    all_features = np.concatenate(all_features, axis=0)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_features)
    logging.info(f"Extracted features saved to {output_path}")

    # Save catalog and metadata
    save_catalog(image_paths, output_dir, model_keyword="dino", dataset_keyword="exp")
    save_metadata(output_dir, model_name, len(image_paths), all_features.shape[1:], model_keyword="dino", dataset_keyword="exp")

    end_time = time.time()
    total_time = end_time - start_time
    time_per_frame = total_time / len(image_paths)
    logging.info(f"Total time: {total_time:.2f} seconds, Time per frame: {time_per_frame:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINOv2 features from images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save extracted features (numpy file).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (e.g., 'cpu', 'cuda').")
    args = parser.parse_args()

    extract_features(
        image_dir=args.image_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=args.device,
    )
