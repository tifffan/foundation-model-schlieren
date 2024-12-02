# test_autoencoders.py

import unittest
import torch
from autoencoders import UNetAutoencoder, UNetVAE

class TestModels(unittest.TestCase):
    def setUp(self):
        self.n_channels = 16  # Example reduced_dim
        self.latent_dim = 64
        self.conv_channels = [64, 128, 256]
        self.batch_size = 2
        self.img_size = 32
        self.input_tensor = torch.randn(self.batch_size, self.n_channels, self.img_size, self.img_size)

    def test_unet_autoencoder(self):
        model = UNetAutoencoder(
            n_channels=self.n_channels,
            conv_channels=self.conv_channels,
            latent_dim=self.latent_dim,
            linear_hidden_dim=256,
            bilinear=False
        )
        output = model(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape, "Output shape should match input shape for autoencoder")

    def test_unet_vae(self):
        model = UNetVAE(
            n_channels=self.n_channels,
            conv_channels=self.conv_channels,
            latent_dim=self.latent_dim,
            linear_hidden_dim=256,
            bilinear=False
        )
        output, mu, logvar = model(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape, "Output shape should match input shape for VAE")
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim), "Mu shape mismatch")
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim), "Logvar shape mismatch")

if __name__ == '__main__':
    unittest.main()
