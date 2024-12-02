# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_vae = 'vae' in self.config.model_keyword.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss function components
        self.recon_criterion = nn.MSELoss(reduction='sum')  # Sum over elements

        # Optimizer with weight decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.start_epoch = 0

        # Checkpoint directory
        self.hyperparams_str = self.get_hyperparams_str(self.config, self.is_vae)
        self.checkpoint_dir = os.path.join(self.config.result_dir, self.config.dataset_keyword, self.config.model_keyword, self.hyperparams_str)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load checkpoint if provided
        if config.checkpoint:
            self.load_checkpoint(config.checkpoint)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')  # Initialize best validation loss
        self.num_pixels = None
        

    @staticmethod
    def get_hyperparams_str(config, is_vae):
        conv_channels_str = '_'.join(map(str, config.conv_channels)) if config.conv_channels else 'default'
        hyperparams = (
            f"r_{config.random_seed}"
            f"_ld_{config.latent_dim}"
            f"_conv_{conv_channels_str}"
            f"_lr_{config.learning_rate}"
            f"_bs_{config.batch_size}"
            f"_wd_{config.weight_decay}"
        )
        
        if is_vae:
            hyperparams += f"_beta_{config.beta}"
        return hyperparams

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved at {best_model_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from checkpoint at epoch {self.start_epoch - 1}")

    def loss_function(self, outputs, inputs, mu=None, logvar=None):
        # Reconstruction loss
        recon_loss = self.recon_criterion(outputs, inputs)
        if self.is_vae:
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # Per-sample KL divergence
            kl_loss = kl_loss.mean()  # Average over batch
            # Total loss
            loss = recon_loss + self.config.beta * kl_loss
            return loss, recon_loss, kl_loss
        else:
            loss = recon_loss
            return loss, recon_loss, torch.tensor(0.0)

    def train(self):
        # Set number of pixels per image if not already set
        if self.num_pixels is None:
            sample_input, _ = next(iter(self.train_loader))
            self.num_pixels = sample_input[0].numel()  # Number of elements in one image
        
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Divide the losses by the number of pixels
            train_loss_normalized = train_loss #/ self.num_pixels
            val_loss_normalized = val_loss #/ self.num_pixels

            self.train_losses.append(train_loss_normalized)
            self.val_losses.append(val_loss_normalized)

            # Check if current validation loss is the best
            is_best = val_loss_normalized < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint every N epochs or if it's the best model
            if (epoch + 1) % self.config.save_checkpoint_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            print(f"Epoch [{epoch+1}/{self.config.epochs}] - Train Loss: {train_loss_normalized:.6f}, Val Loss: {val_loss_normalized:.6f}")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            if self.is_vae:
                outputs, mu, logvar = self.model(inputs)
                loss, recon_loss, kl_loss = self.loss_function(outputs, inputs, mu, logvar)
            else:
                outputs = self.model(inputs)
                loss, recon_loss, kl_loss = self.loss_function(outputs, inputs)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(self.train_loader.dataset)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                inputs = inputs.to(self.device)
                if self.is_vae:
                    outputs, mu, logvar = self.model(inputs)
                    loss, recon_loss, kl_loss = self.loss_function(outputs, inputs, mu, logvar)
                else:
                    outputs = self.model(inputs)
                    loss, recon_loss, kl_loss = self.loss_function(outputs, inputs)
                running_loss += loss.item()
        avg_loss = running_loss / len(self.val_loader.dataset)
        return avg_loss

    def evaluate(self):
        # Load best model
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
        else:
            print("Best model not found. Please ensure training has been completed.")
            return

        # Plot loss history
        self.plot_loss_history()

        # Compute loss on entire training dataset
        train_loss = self.compute_full_loss(self.train_loader, dataset_name='Training')
        # Compute loss on entire validation dataset
        val_loss = self.compute_full_loss(self.val_loader, dataset_name='Validation')
        
        train_loss_normalized = train_loss #/ self.num_pixels
        val_loss_normalized = val_loss #/ self.num_pixels

        print(f"Best Model Evaluation - Training Loss: {train_loss_normalized:.6f}, Validation Loss: {val_loss_normalized:.6f}")

    def compute_full_loss(self, data_loader, dataset_name='Dataset'):
        running_loss = 0.0
        recon_running_loss = 0.0
        kl_running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                if self.is_vae:
                    outputs, mu, logvar = self.model(inputs)
                    loss, recon_loss, kl_loss = self.loss_function(outputs, inputs, mu, logvar)
                else:
                    outputs = self.model(inputs)
                    loss, recon_loss, kl_loss = self.loss_function(outputs, inputs)
                running_loss += loss.item()
                recon_running_loss += recon_loss.item()
                kl_running_loss += kl_loss.item()
        avg_loss = running_loss / len(data_loader.dataset) #/ self.num_pixels
        avg_recon_loss = recon_running_loss / len(data_loader.dataset) #/ self.num_pixels
        avg_kl_loss = kl_running_loss / len(data_loader.dataset) #/ self.num_pixels

        print(f"{dataset_name} Loss: {avg_loss:.6f} | Recon Loss: {avg_recon_loss:.6f} | KL Loss: {avg_kl_loss:.6f}")
        return avg_loss

    def plot_loss_history(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Save the plot
        plot_path = os.path.join(self.checkpoint_dir, 'loss_history.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss history plot saved at {plot_path}")
