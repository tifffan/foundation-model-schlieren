import torch
from transformers import CLIPProcessor, CLIPModel
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
    Preprocesses an image for input into the CLIP model.
    Args:
        image_path (str): Path to the image file.
        processor: Hugging Face CLIP processor.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def extract_features(image_dir, output_path, batch_size=32, device="cpu"):
    """
    Extracts CLIP features from a directory of images.
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
        output_path = output_path / "clip_features.npy"
        
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the CLIP model and processor    
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
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
            
            # Print size/shape of one input image
            if i == 0:
                print(f"Input image size: {inputs['pixel_values'].shape}")

            # Process in batches
            if len(batch_images) == batch_size or i == len(image_paths) - 1:
                # Concatenate batch inputs
                batch = {k: torch.cat([d[k] for d in batch_images]).to(device) for k in batch_images[0]}
                
                print(f"Processing batch {i // batch_size + 1}")
                # print the shape of a image batch that functions as actual model input
                
                outputs = model.get_image_features(**batch)  # Extract image features
                print(f"outputs shape: {outputs.shape}")
                features = outputs / outputs.norm(dim=-1, keepdim=True)  # Normalize features
                print(f"features shape: {features.shape}")
                
                outputs2 = model.vision_model(pixel_values=batch["pixel_values"], output_hidden_states=True)
                patch_embeddings = outputs2.hidden_states[-1]
                print(f"patch_embeddings shape: {patch_embeddings.shape}")
                patch_embeddings_no_cls = patch_embeddings[:, 1:, :]  # Shape: [32, 49, 768]
                print(f"patch_embeddings_no_cls shape: {patch_embeddings_no_cls.shape}")
                
                batch_size = patch_embeddings_no_cls.size(0)
                hidden_dim = patch_embeddings_no_cls.size(2)
                grid_size = int(patch_embeddings_no_cls.size(1) ** 0.5)  # sqrt of num_patches
                grid_embeddings = patch_embeddings_no_cls.view(batch_size, grid_size, grid_size, hidden_dim)
                print(f"grid_embeddings shape: {grid_embeddings.shape}")
                
                features = grid_embeddings
                
                all_features.append(features.cpu().numpy())
                batch_images = []  # Reset batch

    # Save features to a file
    all_features = np.concatenate(all_features, axis=0)
    np.save(output_path, all_features)  # Save features as .npy file
    print(f"Extracted features saved to {output_path}")
    
    # Save catalog and metadata
    save_catalog(image_paths, output_dir, model_keyword="clip", dataset_keyword="sim")
    save_metadata(output_dir, model_name, len(image_paths), all_features.shape[1:], model_keyword="clip", dataset_keyword="sim")

    end_time = time.time()
    total_time = end_time - start_time
    time_per_frame = total_time / len(image_paths)
    logging.info(f"Total time: {total_time:.2f} seconds, Time per frame: {time_per_frame:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP features from images.")
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


# python src/feature_extraction/extract_clip_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/frames_test --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/frames_test/clip_features.npy

# python src/feature_extraction/extract_clip_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/exp_frames_test --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/exp_frames_test/clip_features.npy

# python src/feature_extraction/extract_clip_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/exp_frames_672_432 --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/exp_frames_672_432/clip_features.npy

# python src/feature_extraction/extract_clip_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/sim_frames_672_432 --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/sim_frames_672_432/clip_features.npy
