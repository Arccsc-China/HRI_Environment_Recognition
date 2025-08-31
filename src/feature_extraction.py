import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from vae_training import FeatureVAE, Config  # reuse model and configuration from vae_training

# ===== Configuration for feature extraction =====
class FeatureConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    # Feature groups map names to feature channel indices in the preprocessed data
    FEATURE_GROUPS = {
        "kinematic": [0, 1],           # Angle, Target
        "emg": [2, 3, 4, 5],           # Ext, Flex, KM, KD
        "kinematic_emg": [0, 1, 2, 3, 4, 5],  # Kinematic + EMG
        "full": [0, 1, 2, 3, 4, 5, 6]   # All features
    }
    SEQ_LENGTH = 250  # sequence length


def load_model(model_path, z_dim):
    """Load a trained VAE model from disk and return it in eval mode on the configured device."""
    model = FeatureVAE(z_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(FeatureConfig.DEVICE)
    model.eval()
    return model


def extract_features(data_dir, model_dir, output_dir, z_dim, feature_group, file_name=Config.FILE_NAME):
    """Extract latent features for a given feature group using independently trained VAEs.

    The function loads the processed data (.npz) and the saved train/val/test split indices.
    For each set (train/val/test), it loads the corresponding VAE per feature channel, encodes
    all sequences into latent means (mu) and stores concatenated latent vectors per sample.

    Returns the last processed set's latent features and labels (by design this is the test set).
    """
    # Load processed data
    data = np.load(os.path.join(data_dir, file_name))
    all_data = data["data"]   # (num_samples, 250, num_features)
    all_labels = data["labels"]

    # Load saved split indices (ensures same train/val/test split as preprocessing)
    split_indices = np.load(os.path.join(data_dir, "data_split_indices.npz"))
    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]
    test_idx = split_indices["test_idx"]

    # Get feature channel indices for the requested group
    feature_indices = FeatureConfig.FEATURE_GROUPS[feature_group]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each set separately and save per-set feature files
    last_features = None
    last_labels = None
    for set_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        set_data = all_data[indices]
        set_labels = all_labels[indices]

        # Preallocate latent feature matrix: (set_size, len(feature_indices)*z_dim)
        latent_features = np.zeros((len(indices), len(feature_indices) * z_dim))

        # For each feature channel in the group, load its VAE and encode the sequences
        for i, feature_idx in enumerate(feature_indices):
            # Extract the single-channel sequences for this set
            feature_data = set_data[:, :, feature_idx]  # (set_size, seq_len)
            dataset = TensorDataset(torch.tensor(feature_data, dtype=torch.float32))
            data_loader = DataLoader(dataset, batch_size=FeatureConfig.BATCH_SIZE, shuffle=False)

            # Model file naming: consistent with vae_training naming
            feature_name = list(Config.FEATURE_GROUPS.keys())[feature_idx]
            model_path = os.path.join(model_dir, f"best_model_{feature_name}.pth")
            model = load_model(model_path, z_dim)

            # Collect latent means (mu) for all samples in the set
            all_mu = []
            with torch.no_grad():
                for batch in data_loader:
                    x = batch[0].to(FeatureConfig.DEVICE)
                    mu, _ = model.encode(x)
                    all_mu.append(mu.cpu().numpy())

            feature_mu = np.vstack(all_mu)
            start_idx = i * z_dim
            end_idx = (i + 1) * z_dim
            latent_features[:, start_idx:end_idx] = feature_mu

        # Save the set-specific latent features and labels
        np.savez(os.path.join(output_dir, f"{feature_group}_{set_name}_features.npz"),
                 features=latent_features, labels=set_labels)

        last_features = latent_features
        last_labels = set_labels

    # Return the last processed set (test by convention)
    return last_features, last_labels


def save_features(output_dir, features, labels, feature_group):
    """Helper to save a single feature matrix and labels to disk."""
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, f"{feature_group}_features.npz"), features=features, labels=labels)
    print(f"Saved {feature_group} features: {features.shape} with {len(labels)} labels")


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction using Independent VAEs")
    parser.add_argument("--data_dir", type=str,
                        default="data/processed/subject_"+f"{Config.SUBJECT_IDX}"+"/"+f"{Config.TRIAL_SET}",
                        help="Path to processed data directory")
    parser.add_argument("--model_dir", type=str,
                        default="results/subject_"+f"{Config.SUBJECT_IDX}"+"/vae_"+f"{Config.TRIAL_SET}",
                        help="Path to trained model directory")
    parser.add_argument("--z_dim", type=int, default=Config.Z_DIM,
                        help="Dimension of latent space (must match training)")
    args = parser.parse_args()

    output_dir = os.path.join(args.model_dir, "extracted_features")
    os.makedirs(output_dir, exist_ok=True)

    feature_groups = ["kinematic", "emg", "kinematic_emg", "full"]

    for feature_group in feature_groups:
        print(f"\nExtracting features for {feature_group}...")
        features, labels = extract_features(args.data_dir, args.model_dir, output_dir, args.z_dim, feature_group)
        print(f"Extracted {feature_group} features for all sets.")

    print(f"\nFeature extraction complete for all feature groups!")
    print(f"Features saved to: {output_dir}")


if __name__ == "__main__":
    main()
