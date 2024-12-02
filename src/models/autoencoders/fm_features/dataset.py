# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, data_path, subsample_fraction=None, subsample_size=None):
        self.features = np.load(data_path)  # Expected shape: (num_samples, grid_size, grid_size, reduced_dim)
        self.features = self.features.astype(np.float32)

        # Convert to (num_samples, reduced_dim, grid_size, grid_size)
        self.features = np.transpose(self.features, (0, 3, 1, 2))

        # Resize if necessary
        if self.features.shape[2] != 32 or self.features.shape[3] != 32:
            self.features = torch.nn.functional.interpolate(
                torch.tensor(self.features), size=(32, 32), mode='bilinear', align_corners=False
            ).numpy()

        # Convert features to torch tensor
        self.features = torch.from_numpy(self.features)

        # Subsampling
        self.subsample(subsample_fraction, subsample_size)

    def subsample(self, subsample_fraction=None, subsample_size=None):
        num_samples = self.features.shape[0]
        indices = np.arange(num_samples)

        if subsample_fraction is not None:
            assert 0 < subsample_fraction <= 1, "subsample_fraction must be between 0 and 1"
            subsample_size = int(num_samples * subsample_fraction)
            indices = np.random.choice(indices, subsample_size, replace=False)
        elif subsample_size is not None:
            assert 0 < subsample_size <= num_samples, "subsample_size must be between 1 and the number of samples"
            indices = np.random.choice(indices, subsample_size, replace=False)
        else:
            # No subsampling
            return

        # Subsample the features
        self.features = self.features[indices]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]
        return x, x  # Since it's an autoencoder, input and target are the same
