# config.py

import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Training Configuration")

    # Model parameters
    parser.add_argument('--model_keyword', type=str, default='unet_ae_3blk',
                        choices=['unet_ae_3blk', 'unet_ae_4blk', 'unet_ae_5blk',
                                 'unet_vae_3blk', 'unet_vae_4blk', 'unet_vae_5blk'],
                        help='Model type to use.')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimensionality of the latent space.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for VAE loss function.')
    parser.add_argument('--conv_channels', type=str, default=None,
                        help='Comma-separated list of convolutional channels, e.g., "8,8,8,16,16".')

    # Dataset parameters
    parser.add_argument('--dataset_keyword', type=str, default='features_dataset',
                        help='Dataset type to use.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the reduced features file.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--save_checkpoint_every', type=int, default=5,
                        help='Save checkpoint every N epochs.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    # Result directory and checkpoint parameters
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Base directory to save results and checkpoints.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint to resume training.')
    
    # Subsampling parameters
    parser.add_argument('--subsample_fraction', type=float, default=None,
                        help='Fraction of the dataset to use (between 0 and 1).')
    parser.add_argument('--subsample_size', type=int, default=None,
                        help='Number of samples to use.')

    args = parser.parse_args()

    # Parse conv_channels if provided
    if args.conv_channels:
        args.conv_channels = [int(ch.strip()) for ch in args.conv_channels.split(',')]

    return args
