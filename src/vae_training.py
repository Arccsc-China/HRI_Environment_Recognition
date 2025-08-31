import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

# ===== Configuration =====
class Config:
    # Dataset and output paths
    SUBJECT_IDX = 21  # 0-21
    TRIAL_SET = 'stable'  # 'all' or 'stable'
    PROCESSED_PATH = "data/processed/subject_"+f"{SUBJECT_IDX}"+"/"+f"{TRIAL_SET}"  # path to preprocessed data
    OUTPUT_DIR = "results/subject_"+f"{SUBJECT_IDX}"+"/vae_"+f"{TRIAL_SET}"    # output directory for results
    FILE_NAME = "subject_"+f"{SUBJECT_IDX}"+"_processed.npz"  # preprocessed file name

    # Feature groups: currently training one VAE per feature
    FEATURE_GROUPS = {
        "Angle": [0],
        "Target": [1],
        "Ext": [2],
        "Flex": [3],
        "KM": [4],
        "KD": [5],
        "Torque": [6]
    }
    
    # VAE hyperparameters
    Z_DIM = 32          # latent dimension
    BETA = 0.001        # KL divergence weight
    LR = 1e-4           # learning rate
    EPOCHS = 100        # number of training epochs
    BATCH_SIZE = 64     # batch size
    
    # Device selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluation / visualization
    RECON_SAMPLES = 3   # number of reconstruction samples to visualize
    SEQ_LENGTH = 250    # input sequence length


# ===== Simplified VAE model for single-feature sequences =====
class FeatureVAE(nn.Module):
    def __init__(self, z_dim):
        super(FeatureVAE, self).__init__()
        self.seq_len = Config.SEQ_LENGTH
        
        # Encoder: maps sequence -> hidden
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Latent parameters
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)
        
        # Decoder: latent -> reconstructed sequence
        self.decoder_fc = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.seq_len)
        )
    
    def encode(self, x):
        """Encode input sequence into mu and logvar."""
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        recon = self.decoder_fc(z)
        return recon.view(-1, self.seq_len)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# ===== Core utilities =====
def load_data(processed_path, feature_idx, file_name=Config.FILE_NAME):
    """Load preprocessed arrays and return DataLoaders for a single feature.

    Expects the preprocessed .npz to contain 'data' (num_samples, 250, 7) and 'labels'.
    It also expects a 'data_split_indices.npz' with train/val/test indices produced by preprocessing.
    """
    # Load processed data
    data = np.load(os.path.join(processed_path, file_name))
    all_data = data["data"]   # (num_samples, 250, 7)
    all_labels = data["labels"] # (num_samples,)

    # Load precomputed split indices (ensures consistent splits across runs)
    split_indices = np.load(os.path.join(processed_path, "data_split_indices.npz"))
    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]
    test_idx = split_indices["test_idx"]
    
    # Select single feature channel
    selected_data = all_data[:, :, feature_idx]  # (num_samples, 250)

    # Split according to saved indices
    train_data = selected_data[train_idx]
    train_labels = all_labels[train_idx]
    val_data = selected_data[val_idx]
    val_labels = all_labels[val_idx]
    test_data = selected_data[test_idx]
    test_labels = all_labels[test_idx]

    # Convert to PyTorch TensorDatasets
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    print(f"Loaded data: Total samples={len(all_labels)}")
    print(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


def vae_loss(recon_x, x, mu, logvar, beta=Config.BETA):
    """Compute VAE loss: reconstruction (MSE) + beta * KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_model(model, train_loader, val_loader, epochs, device, feature_name):
    """Train the VAE and save the best model based on validation loss."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, _ = batch
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon_x, x, mu, logvar)
                val_loss += loss.item()

        # Record epoch statistics
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_{feature_name}.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model_{feature_name}.pth"))
    return history


def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set and collect latent vectors and labels."""
    model.eval()
    test_loss = 0
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, labels = batch
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon_x, x, mu, logvar)
            test_loss += loss.item()
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    latent_vectors = np.vstack(all_mu)
    labels = np.concatenate(all_labels)
    return avg_test_loss, latent_vectors, labels


def visualize_reconstruction(model, test_loader, device, feature_name, n_samples=Config.RECON_SAMPLES):
    """Visualize a few reconstructed sequences vs original sequences."""
    model.eval()
    samples = []

    # Collect a few batches for visualization
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_samples:
                break
            x, _ = batch
            x = x.to(device)
            recon_x, _, _ = model(x)
            samples.append((x.cpu().numpy(), recon_x.cpu().numpy()))

    # Plot and save
    plt.figure(figsize=(12, 8))
    for i, (orig, recon) in enumerate(samples):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(orig[0], label="Original")
        plt.plot(recon[0], linestyle='--', label="Reconstructed")
        plt.title(f"Sample {i+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle(f"Reconstruction for {feature_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"recon_{feature_name}.png"))
    plt.close()


def visualize_latent_space(latent_vectors, labels, feature_name):
    """Project latent vectors to 2D with t-SNE and save a scatter plot colored by condition."""
    # Set a reasonable perplexity
    if len(latent_vectors) > 30:
        perplexity = 30
    else:
        perplexity = max(5, len(latent_vectors) // 3)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    latent_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    visual_noise = ["V0", "V1", "V2"]
    haptic_noise = ["H0", "H1", "H2"]
    colors = plt.cm.viridis(np.linspace(0, 1, 9))

    for cond_idx in range(9):
        mask = labels == cond_idx
        if np.sum(mask) > 0:
            v_idx = cond_idx // 3
            h_idx = cond_idx % 3
            label = f"{visual_noise[v_idx]}{haptic_noise[h_idx]}"
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                        color=colors[cond_idx], s=50, alpha=0.7,
                        label=label, edgecolor='w', linewidth=0.5)

    plt.title(f"Latent Space for {feature_name}", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)

    # Create compact legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Noise Conditions", loc='best', frameon=True, framealpha=0.8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"latent_space_{feature_name}.png"))
    plt.close()


def save_results(results, feature_name):
    """Save evaluation summary to CSV and print a short summary."""
    result_dict = {
        "feature": feature_name,
        "test_loss": results[0],
        "z_dim": Config.Z_DIM,
        "epochs": Config.EPOCHS
    }
    df = pd.DataFrame([result_dict])
    df.to_csv(os.path.join(output_dir, f"results_{feature_name}.csv"), index=False)

    print("\n" + "="*50)
    print(f"Results for {feature_name}:")
    print(f"- Test Loss: {result_dict['test_loss']:.4f}")
    print("="*50)


# ===== Main entrypoint =====
if __name__ == "__main__":
    # Create output directory
    output_dir = Config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Train a VAE per feature
    for feature_name, feature_indices in Config.FEATURE_GROUPS.items():
        feature_idx = feature_indices[0]  # single channel per feature
        
        print(f"\n{'='*50}")
        print(f"Training VAE for feature: {feature_name}")
        print(f"{'='*50}")
        
        # 1. Load data
        train_loader, val_loader, test_loader = load_data(Config.PROCESSED_PATH, feature_idx)
        
        # 2. Initialize model
        model = FeatureVAE(Config.Z_DIM)
        
        # 3. Train
        history = train_model(model, train_loader, val_loader, Config.EPOCHS, Config.DEVICE, feature_name)
        
        # 4. Load best model and evaluate
        model.load_state_dict(torch.load(os.path.join(output_dir, f"best_model_{feature_name}.pth")))
        test_loss, latent_vectors, test_labels = evaluate_model(model, test_loader, Config.DEVICE)
        
        # 5. Visualize
        visualize_reconstruction(model, test_loader, Config.DEVICE, feature_name)
        visualize_latent_space(latent_vectors, test_labels, feature_name)
        
        # 6. Save summary results
        results = (test_loss,)
        save_results(results, feature_name)
    
    print("\nTraining and evaluation completed!")
    print(f"Results saved to: {output_dir}")
