#!/usr/bin/env python3

import sys
import time
import logging
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
import argparse
import csv
import json

# Add the path to dust3r package if necessary
sys.path.append('/Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/dust3r')
from dust3r.model import AsymmetricCroCo3DStereo

from feature_extraction_utils import *

class CustomProcessor:
    def __init__(self, img_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, return_tensors=None):
        if isinstance(image, Image.Image):
            image = [image]
        elif isinstance(image, list):
            if not all(isinstance(img, Image.Image) for img in image):
                raise TypeError("All items in image list should be PIL.Image instances.")
        else:
            raise TypeError("image should be a PIL.Image or a list of PIL.Image instances.")

        processed_images = [self.transform(img) for img in image]
        tensor_images = torch.stack(processed_images)

        if return_tensors == "pt":
            tensor_instances = torch.zeros((tensor_images.size(0), 1, tensor_images.size(2), tensor_images.size(3)), dtype=torch.float32)
            return {'img': tensor_images, 'instance': tensor_instances}
        else:
            return {'img': tensor_images, 'instance': tensor_instances}

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)

def save_catalog(image_paths, output_dir, model_keyword="dust3r", dataset_keyword="sim"):
    catalog_name = f"catalog_{model_keyword}_{dataset_keyword}.csv"
    catalog_path = output_dir / catalog_name
    with open(catalog_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Image_Path"])
        for idx, image_path in enumerate(image_paths):
            writer.writerow([idx, str(image_path)])
    logging.info(f"Catalog saved to {catalog_path}")

def save_metadata(output_dir, model_name, data_size, feature_dim, model_keyword="dust3r", dataset_keyword="sim"):
    metadata_name = f"metadata_{model_keyword}_{dataset_keyword}.json"
    metadata_path = output_dir / metadata_name
    metadata = {
        "model_name": model_name,
        "data_size": data_size,
        "feature_dim": feature_dim
    }
    with open(metadata_path, 'w') as metafile:
        json.dump(metadata, metafile, indent=4)
    logging.info(f"Metadata saved to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract features using DUSt3R model.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted features.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (e.g., 'cpu', 'cuda').")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    start_time = time.time()

    # # Initialize the custom processor and load the model
    # processor = CustomProcessor(img_size=(224, 224))
    # model_name = "DUSt3R_ViTLarge_BaseDecoder_224_linear"
    # model_keyword = "dust3r_224"
    # dataset_keyword = "exp"
    
    processor = CustomProcessor(img_size=(512, 512))
    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_linear"
    model_keyword = "dust3r_512"
    dataset_keyword = "exp"
    
    
    
    try:
        if args.model_path:
            model = load_model(args.model_path, device=args.device)
        else:
            model = AsymmetricCroCo3DStereo.from_pretrained(f"naver/{model_name}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Process images
    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")) + \
             sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.bmp"))

    if not images:
        logging.error(f"No images found in {image_dir}.")
        return

    logging.info(f"Found {len(images)} images in {image_dir}")
    
    # Sort by run number and frame number
    images = sort_images_by_filename(images)

    all_features = []
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(images), args.batch_size):
        batch_images = images[i:i + args.batch_size]
        total_batches = (len(images) + args.batch_size - 1) // args.batch_size
        logging.info(f"Processing batch {i // args.batch_size + 1} / {total_batches}")

        inputs_list = [Image.open(p).convert("RGB") for p in batch_images]
        processed_view = processor(inputs_list, return_tensors="pt")
        processed_view = {k: v.to(device) for k, v in processed_view.items()}
        true_shape = torch.tensor([img.size[::-1] for img in inputs_list], dtype=torch.float32, device=device)

        with torch.no_grad():
            features, pos, _ = model._encode_image(processed_view['img'], true_shape)
                        
            features = reshape_vector_feature_to_square_grid(features)
            
            print(f"features shape: {features.shape}")
            
            all_features.append(features.cpu().numpy())

    # Save features
    if all_features:
        all_features = np.concatenate(all_features, axis=0)
        np.save(args.output_path, all_features)
        logging.info(f"Saved features to {args.output_path}")
        save_catalog(images, output_dir, model_keyword=model_keyword, dataset_keyword=dataset_keyword)
        save_metadata(output_dir, model_name, len(images), all_features.shape[1:], model_keyword=model_keyword, dataset_keyword=dataset_keyword)
    else:
        logging.warning("No features were extracted.")

    end_time = time.time()
    total_time = end_time - start_time
    time_per_frame = total_time / len(images)
    logging.info(f"Total time: {total_time:.2f} seconds, Time per frame: {time_per_frame:.2f} seconds")

if __name__ == "__main__":
    main()

# python src/feature_extraction/extract_dust3r_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/frames_test --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/frames_test/dust3r_features.npy

# python src/feature_extraction/extract_dust3r_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/sim_frames_672_432 --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/sim_frames_672_432/dust3r_224_features.npy

# python src/feature_extraction/extract_dust3r_features.py --image_dir /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/raw/sim_frames_672_432 --output_path /Users/tiffan/Desktop/CS468/project/foundation-model-schlieren/data/features/sim_frames_672_432/dust3r_512_features.npy
