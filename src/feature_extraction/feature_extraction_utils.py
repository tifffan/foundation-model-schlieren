import csv
import json
import logging
import re
import torch

def reshape_vector_feature_to_square_grid(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Reshape a batch of patch embeddings into a square grid.

    Args:
        patch_embeddings (torch.Tensor): Input tensor of shape 
            (batch_size, num_patches, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape 
            (batch_size, grid_size, grid_size, hidden_dim), 
            where grid_size = sqrt(num_patches).
    """
    # Validate input dimensions
    if patch_embeddings.dim() != 3:
        raise ValueError(f"Input tensor must be 3-dimensional, but got {patch_embeddings.dim()} dimensions.")

    batch_size = patch_embeddings.size(0)
    num_patches = patch_embeddings.size(1)
    hidden_dim = patch_embeddings.size(2)
    
    # Compute the grid size
    grid_size = int(num_patches ** 0.5)
    if grid_size ** 2 != num_patches:
        raise ValueError(f"num_patches ({num_patches}) is not a perfect square.")

    # Reshape to grid format
    grid_embeddings = patch_embeddings.view(batch_size, grid_size, grid_size, hidden_dim)
    return grid_embeddings

def sort_images_by_filename(image_list):
    """
    Sorts a list of image filenames by run number and frame number for 'runXX' type,
    or by test number and timestamp for 'TestXXX' type. Places 'TestXXX' type files
    after 'runXX' type files.

    Args:
        image_list (list): List of image file paths.

    Returns:
        list: Sorted list of image file paths.
    """
    def extract_key(filename):
        """
        Extracts sorting keys from the filename.

        Args:
            filename (str): Filename to extract information from.

        Returns:
            tuple: Sorting key. First element determines type (0 for 'runXX', 1 for 'TestXXX').
                   Second and third elements are numbers for sorting within each type.
        """
        # Match 'runXX' type filenames
        run_frame_match = re.search(r"run(\d+)-.*_frame(\d+)", filename)
        if run_frame_match:
            run_number = int(run_frame_match.group(1))
            frame_number = int(run_frame_match.group(2))
            return (0, run_number, frame_number)

        # Match 'TestXXX' type filenames
        test_timestamp_match = re.search(r"Test(\d+)_SCH_IMG_(\d+\.\d+)", filename)
        if test_timestamp_match:
            test_number = int(test_timestamp_match.group(1))
            timestamp = float(test_timestamp_match.group(2))
            return (1, test_number, timestamp)

        # Default key for unrecognized files
        return (2, float('inf'), float('inf'))

    # Sort by extracted keys
    return sorted(image_list, key=lambda x: extract_key(str(x)))

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