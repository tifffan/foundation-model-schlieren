# src/utils/data_loader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import pandas as pd
from torchvision import transforms

# Define the run ID categories
direct_ignition = [0, 6, 9, 14, 17, 18, 19, 21, 24, 25, 28, 29, 32, 35, 37]
indirect_ignition = [2, 4, 16, 23, 27, 42]

class CompSchlierenDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and metadata.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Assign ignition labels based on run_id
        self.data['ignition'] = self.data['run_id'].apply(self.assign_ignition_label)

    def assign_ignition_label(self, run_id):
        """Assign ignition labels based on predefined run IDs."""
        if run_id in direct_ignition:
            return 2
        elif run_id in indirect_ignition:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.data.iloc[idx, 0]
        
        try:
            image = Image.open(img_name).convert('L')  # Load in grayscale
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a tensor of zeros and default labels or handle as per requirement
            dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
            return dummy_image, torch.tensor(0), torch.tensor(0), torch.tensor(0)

        # Convert grayscale to RGB by duplicating channels
        image = image.convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Define default transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # Converts to [0, 1] and shape [C, H, W]
                # You can add normalization here if required
            ])
            image = transform(image)

        ignition = torch.tensor(self.data.iloc[idx]['ignition'], dtype=torch.long)
        run_id = torch.tensor(self.data.iloc[idx]['run_id'], dtype=torch.long)
        frame_id = torch.tensor(self.data.iloc[idx]['frame_id'], dtype=torch.long)

        return image, ignition, run_id, frame_id

if __name__ == "__main__":
    # Example usage of the CompSchlierenDataset
    dataset = CompSchlierenDataset(csv_file='data/raw/test_catalog.csv')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    for images, ignitions, run_ids, frame_ids in dataloader:
        print(f"Image batch shape: {images.shape}")  # Expected: [4, 3, 224, 224]
        print(f"Ignition labels: {ignitions}")
        print(f"Run IDs: {run_ids}")
        print(f"Frame IDs: {frame_ids}")
        break  # Only display the first batch
