"""
Variational Autoencoder (VAE) for synthetic patient generation

This is the core generative model that actually creates synthetic patients.
The VAE learns the probability distribution of clinically plausible patient
data and lets us sample new patients from that learned distribution.

Architecture:
- Encoder compresses patient features into a latent space (50 dimensions)
- Decoder reconstructs patients from latent codes
- Reparameterization trick makes it all differentiable

Math (from Section 3.5 of the paper):
    Encoder: q_φ(z|x) ≈ N(μ_φ(x), σ²_φ(x))
    Decoder: p_θ(x|z) ≈ N(μ_θ(z), σ²_θ(z))

    ELBO Loss = Reconstruction Loss − KL Divergence
    Where KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

Built by: Chatchai Tritham & Chakkrit Snae Namahoot
Where: Naresuan University, Thailand
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAEModel(nn.Module):
    """
    Variational Autoencoder for synthetic data generation.

    Architecture:
    - Encoder: Input → [512, 256, 128] → μ (latent_dim), log(σ²) (latent_dim)
    - Decoder: z (latent_dim) → [128, 256, 512] → Output (reconstructed input)

    Parameters match paper specifications:
    - Latent dimension: 20 (r=20 as per NMF rank)
    - Activation: ReLU for hidden layers
    - Output activation: Sigmoid for binary/normalized features
    """

    def __init__(self, input_dim: int, latent_dim: int = 20, hidden_dims: list = None):
        """
        Initialize VAE model.

        Args:
            input_dim: Dimension of input features (after one-hot encoding)
            latent_dim: Dimension of latent space (default: 20, matching paper's r=20)
            hidden_dims: List of hidden layer dimensions (default: [512, 256, 128])
        """
        super(VAEModel, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]

        logger.info(f"Initializing VAE: input_dim={input_dim}, latent_dim={latent_dim}")

        # =====================================================================
        # ENCODER: x → μ, log(σ²)
        # =====================================================================
        encoder_layers = []
        prev_dim = input_dim

        for h_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Separate heads for μ and log(σ²)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], latent_dim)

        # =====================================================================
        # DECODER: z → x̂
        # =====================================================================
        decoder_layers = []
        prev_dim = latent_dim

        for h_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer (sigmoid for normalized/binary features)
        self.fc_output = nn.Linear(self.hidden_dims[0], input_dim)

        logger.info("✓ VAE architecture initialized successfully")

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            log_var: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)

        This allows backpropagation through the sampling operation.

        Args:
            mu: Mean [batch_size, latent_dim]
            log_var: Log variance [batch_size, latent_dim]

        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * log_var)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)      # ε ~ N(0, I)
        z = mu + eps * std                # z = μ + σ * ε
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed output.

        Args:
            z: Latent vector [batch_size, latent_dim]

        Returns:
            x_recon: Reconstructed output [batch_size, input_dim]
        """
        h = self.decoder(z)
        x_recon = torch.sigmoid(self.fc_output(h))  # Sigmoid for [0, 1] range
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            x_recon: Reconstructed output [batch_size, input_dim]
            mu: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO loss = Reconstruction Loss + KL Divergence

        From paper (Section 3.7):
        L_ELBO(θ, φ; x) = E_q_φ(z|x)[log p_θ(x|z)] − D_KL(q_φ(z|x)||p(z))

        Args:
            recon_x: Reconstructed output [batch_size, input_dim]
            x: Original input [batch_size, input_dim]
            mu: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]
            beta: Weight for KL term (default: 1.0, can use β-VAE for disentanglement)

        Returns:
            total_loss: Total ELBO loss
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence component
        """
        # Reconstruction loss: Binary cross-entropy
        # -E_q_φ(z|x)[log p_θ(x|z)]
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL divergence: D_KL(q_φ(z|x) || p(z))
        # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total ELBO loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from prior p(z) = N(0, I) and decode.

        This generates new synthetic data from the learned latent distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to run on ('cpu' or 'cuda')

        Returns:
            samples: Generated samples [num_samples, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior: z ~ N(0, I)
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
        return samples


def train_vae(
    vae_model: VAEModel,
    train_data: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    convergence_threshold: float = 0.01,
    patience: int = 10,
    save_path: Optional[Path] = None
) -> Dict:
    """
    Train VAE model with early stopping.

    Args:
        vae_model: VAE model instance
        train_data: Training data tensor [n_samples, input_dim]
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        convergence_threshold: Early stopping threshold for loss improvement
        patience: Number of epochs to wait for improvement before stopping
        save_path: Path to save best model (optional)

    Returns:
        training_history: Dictionary containing loss curves and metadata
    """
    logger.info("=" * 80)
    logger.info("SUB-PHASE 2.2: VAE Model Training")
    logger.info("=" * 80)
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {device}")

    # Move model to device
    vae_model = vae_model.to(device)
    train_data = train_data.to(device)

    # Create data loader
    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'best_epoch': 0,
        'best_loss': float('inf')
    }

    # Early stopping
    best_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        vae_model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_idx, (batch_data,) in enumerate(data_loader):
            batch_data = batch_data.to(device)

            # Forward pass
            recon_batch, mu, log_var = vae_model(batch_data)

            # Compute loss
            total_loss, recon_loss, kl_loss = vae_model.loss_function(
                recon_batch, batch_data, mu, log_var
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        # Average losses
        avg_total_loss = epoch_total_loss / len(train_data)
        avg_recon_loss = epoch_recon_loss / len(train_data)
        avg_kl_loss = epoch_kl_loss / len(train_data)

        history['total_loss'].append(avg_total_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)

        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs}: "
                f"Total Loss = {avg_total_loss:.4f}, "
                f"Recon = {avg_recon_loss:.4f}, "
                f"KL = {avg_kl_loss:.4f}"
            )

        # Early stopping check
        if avg_total_loss < best_loss:
            improvement = best_loss - avg_total_loss
            if improvement > convergence_threshold:
                best_loss = avg_total_loss
                patience_counter = 0
                history['best_epoch'] = epoch
                history['best_loss'] = best_loss

                # Save best model
                if save_path:
                    torch.save(vae_model.state_dict(), save_path)
                    logger.info(f"  ✓ New best model saved (epoch {epoch})")
            else:
                patience_counter += 1
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    logger.info("=" * 80)
    logger.info(f"✓ VAE Training Completed")
    logger.info(f"  Best epoch: {history['best_epoch']}")
    logger.info(f"  Best loss: {history['best_loss']:.4f}")
    logger.info(f"  Final recon loss: {history['recon_loss'][-1]:.4f}")
    logger.info(f"  Final KL loss: {history['kl_loss'][-1]:.4f}")
    logger.info("=" * 80)

    return history


def sample_from_vae(
    vae_model: VAEModel,
    n_samples: int,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate synthetic samples from trained VAE.

    Args:
        vae_model: Trained VAE model
        n_samples: Number of samples to generate
        device: Device to run on

    Returns:
        samples: Generated samples as numpy array [n_samples, input_dim]
    """
    logger.info(f"Generating {n_samples} samples from VAE latent space...")

    vae_model = vae_model.to(device)
    vae_model.eval()

    with torch.no_grad():
        # Sample from prior
        z = torch.randn(n_samples, vae_model.latent_dim).to(device)
        samples = vae_model.decode(z)

    samples_np = samples.cpu().numpy()
    logger.info(f"✓ Generated {len(samples_np)} synthetic samples")

    return samples_np


def evaluate_vae_reconstruction(
    vae_model: VAEModel,
    test_data: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate VAE reconstruction quality.

    Args:
        vae_model: Trained VAE model
        test_data: Test data tensor
        device: Device to run on

    Returns:
        metrics: Dictionary containing reconstruction metrics
    """
    vae_model = vae_model.to(device)
    test_data = test_data.to(device)
    vae_model.eval()

    with torch.no_grad():
        recon_data, mu, log_var = vae_model(test_data)
        total_loss, recon_loss, kl_loss = vae_model.loss_function(
            recon_data, test_data, mu, log_var
        )

    # Calculate reconstruction accuracy (for binary features)
    recon_accuracy = ((recon_data > 0.5).float() == test_data).float().mean().item()

    metrics = {
        'reconstruction_accuracy': recon_accuracy,
        'total_loss': total_loss.item() / len(test_data),
        'recon_loss': recon_loss.item() / len(test_data),
        'kl_loss': kl_loss.item() / len(test_data)
    }

    logger.info("VAE Reconstruction Metrics:")
    logger.info(f"  Accuracy: {metrics['reconstruction_accuracy']:.4f}")
    logger.info(f"  Total loss: {metrics['total_loss']:.4f}")
    logger.info(f"  Recon loss: {metrics['recon_loss']:.4f}")
    logger.info(f"  KL loss: {metrics['kl_loss']:.4f}")

    return metrics
