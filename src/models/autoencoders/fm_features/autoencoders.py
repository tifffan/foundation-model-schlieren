# autoencoders.py

import torch
import torch.nn as nn
from src.models.autoencoders.unet_modules.unet_parts import DoubleConv, Down, Up, OutConv

class UNetBase(nn.Module):
    def __init__(self, n_channels, conv_channels, latent_dim, linear_hidden_dim=256, bilinear=False):
        super(UNetBase, self).__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # Initial convolution
        self.inc = DoubleConv(n_channels, conv_channels[0])

        # Downsampling layers
        self.downs = nn.ModuleList()
        for idx in range(len(conv_channels) - 1):
            self.downs.append(Down(conv_channels[idx], conv_channels[idx + 1]))

        # Calculate the size after downsampling
        self.num_downs = len(self.downs)
        self.final_channels = conv_channels[-1]
        self.final_size = 32 // (2 ** self.num_downs)  # Assuming input size is 32x32

        # Linear layers for encoder
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_input_size = self.final_channels * self.final_size * self.final_size

        # Linear layers for decoder
        self.linear_hidden_dim = linear_hidden_dim
        self.linear_decode = nn.Sequential(
            nn.Linear(latent_dim, linear_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_dim, self.linear_input_size),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.final_channels, self.final_size, self.final_size))

        # Upsampling layers
        self.ups = nn.ModuleList()
        reversed_channels = conv_channels[::-1]
        for idx in range(len(reversed_channels) - 1):
            self.ups.append(Up(reversed_channels[idx], reversed_channels[idx + 1], bilinear))

        # Final output layer
        self.outc = OutConv(reversed_channels[-1], n_channels)

    def encoder_forward(self, x):
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        x = self.flatten(x)
        return x

    def decoder_forward(self, x):
        x = self.linear_decode(x)
        x = self.unflatten(x)
        for up in self.ups:
            x = up(x)
        x = self.outc(x)
        return x

class UNetAutoencoder(UNetBase):
    def __init__(self, n_channels, conv_channels, latent_dim, linear_hidden_dim=256, bilinear=False):
        super(UNetAutoencoder, self).__init__(n_channels, conv_channels, latent_dim, linear_hidden_dim, bilinear)
        self.linear_encode = nn.Sequential(
            nn.Linear(self.linear_input_size, linear_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_dim, latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder_forward(x)
        x = self.linear_encode(x)
        x = self.decoder_forward(x)
        return x

class UNetVAE(UNetBase):
    def __init__(self, n_channels, conv_channels, latent_dim, linear_hidden_dim=256, bilinear=False):
        super(UNetVAE, self).__init__(n_channels, conv_channels, latent_dim, linear_hidden_dim, bilinear)
        self.linear_mu = nn.Linear(self.linear_input_size, latent_dim)
        self.linear_logvar = nn.Linear(self.linear_input_size, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder_forward(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder_forward(z)
        return x_recon, mu, logvar
