# main.py

import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from config import get_config
from autoencoders import UNetAutoencoder, UNetVAE
from dataset import FeatureDataset
from trainer import Trainer

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For CUDA deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Get configuration
    config = get_config()

    # Set random seed
    set_seed(config.random_seed)

    # Prepare dataset and dataloaders
    dataset = FeatureDataset(
        data_path=config.data_path,
        subsample_fraction=config.subsample_fraction,
        subsample_size=config.subsample_size
    )

    # Split dataset with deterministic behavior
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.random_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    
    # def worker_init_fn(worker_id):
    #     worker_seed = config.random_seed + worker_id
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)

    # # DataLoader initialization
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size,      shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    
    # DataLoader initialization
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,      shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Define model based on model_keyword
    if config.conv_channels:
        conv_channels = config.conv_channels
    else:
        # Default conv_channels based on model_keyword
        if '3blk' in config.model_keyword:
            conv_channels = [64, 128, 256]
        elif '4blk' in config.model_keyword:
            conv_channels = [64, 128, 256, 512]
        elif '5blk' in config.model_keyword:
            conv_channels = [64, 128, 256, 512, 1024]
        else:
            raise ValueError(f"Unknown model_keyword: {config.model_keyword}")

    # Instantiate model
    if 'unet_ae' in config.model_keyword:
        model = UNetAutoencoder(
            n_channels=dataset.features.shape[1],  # reduced_dim
            conv_channels=conv_channels,
            latent_dim=config.latent_dim,
            linear_hidden_dim=256,
            bilinear=False
        )
    elif 'unet_vae' in config.model_keyword:
        model = UNetVAE(
            n_channels=dataset.features.shape[1],
            conv_channels=conv_channels,
            latent_dim=config.latent_dim,
            linear_hidden_dim=256,
            bilinear=False
        )
    else:
        raise ValueError(f"Unknown model_keyword: {config.model_keyword}")

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Start training
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

if __name__ == "__main__":
    main()
