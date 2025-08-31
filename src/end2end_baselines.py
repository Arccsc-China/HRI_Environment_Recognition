import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal, integrate, stats
from scipy.fft import fft

# ===== Configuration =====
class Config:
    SUBJECT_IDX = 0
    TRIAL_SET = "stable"  # options: "all" or "stable"

    # Feature extraction methods mapping (informational)
    FEATURE_EXTRACTORS = {
        "raw": "raw",
        "pca": "pca",
        "tsne": "tsne",
        "stats": "stats",
        "advanced_stats": "advanced_stats"
    }
    
    # Classical classifiers
    CLASSICAL_CLFS = {
        "SVM": SVC(kernel="rbf", C=1.0, probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10),
        "LogisticRegression": LogisticRegression(max_iter=1000, multi_class="multinomial", solver="saga")
    }
    
    # Deep learning model names (informational)
    DL_MODELS = {
        "MLP": "MLP",
        "LSTM": "LSTM"
    }
    
    METRICS = {
        "Accuracy": accuracy_score,
        "F1_Macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
    }
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    DL_EPOCHS = 100
    DL_BATCH_SIZE = 32
    DL_LR = 0.001
    DL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Feature extraction functions =====
def extract_features(X, method="raw", n_components=50, feature_set="kinematic+emg+torque"):
    """Extract features from time-series data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, seq_len, n_features)
    method : str
        Extraction method: 'raw', 'pca', 'tsne', 'stats', 'advanced_stats'
    n_components : int
        Number of components for dimensionality reduction methods
    feature_set : str
        Which channels to include: 'kinematic', 'emg', 'kinematic+emg', 'kinematic+emg+torque'

    Returns
    -------
    np.ndarray
        Feature matrix with shape (n_samples, n_extracted_features)
    """
    n_samples, seq_len, n_features = X.shape

    # Determine which input channels to use
    feature_indices = []
    if "kinematic" in feature_set:
        feature_indices.extend([0, 1])  # Angle, Target
    if "emg" in feature_set:
        feature_indices.extend([2, 3, 4, 5])  # Ext, Flex, KM, KD
    if "torque" in feature_set:
        feature_indices.extend([6])  # Torque

    # Subselect channels
    X = X[:, :, feature_indices]
    n_features = len(feature_indices)

    if method == "raw":
        # Flatten time dimension
        return X.reshape(n_samples, -1)

    elif method == "pca":
        pca = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
        flattened = X.reshape(n_samples, -1)
        return pca.fit_transform(flattened)

    elif method == "tsne":
        # t-SNE is computationally heavy; use for small datasets only
        tsne = TSNE(n_components=2, random_state=Config.RANDOM_STATE)
        flattened = X.reshape(n_samples, -1)
        return tsne.fit_transform(flattened)

    elif method == "stats":
        # Basic per-channel statistics
        features = []
        for i in range(n_samples):
            sample_features = []
            for j in range(n_features):
                channel = X[i, :, j]
                sample_features.extend([
                    np.mean(channel),    # mean
                    np.std(channel),     # std
                    np.min(channel),     # min
                    np.max(channel),     # max
                    np.median(channel),  # median
                    np.ptp(channel)      # peak-to-peak
                ])
            features.append(sample_features)
        return np.array(features)

    elif method == "advanced_stats":
        # Compute richer per-sample features including kinematic, EMG and torque statistics
        features = []
        for i in range(n_samples):
            sample_features = []

            # Kinematic-derived features (if present)
            if "kinematic" in feature_set:
                angle = X[i, :, 0]
                target = X[i, :, 1]

                # Velocity and acceleration
                velocity = np.gradient(angle)
                acceleration = np.gradient(velocity)

                max_vel = np.max(np.abs(velocity))
                avg_vel = np.mean(np.abs(velocity))
                max_acc = np.max(np.abs(acceleration))
                avg_acc = np.mean(np.abs(acceleration))

                peak_vel_idx = np.argmax(np.abs(velocity))
                time_to_peak = peak_vel_idx / len(angle)

                sample_features.extend([max_vel, avg_vel, max_acc, avg_acc, time_to_peak])

                # Accuracy/error metrics
                error = angle - target
                abs_error = np.abs(error)
                overshoot = np.max(np.abs(error))
                arrival_acc = np.mean(abs_error[-min(100, len(abs_error)):])

                # Normalized error metrics (guard against zero range)
                target_range = (np.max(target) - np.min(target))
                if target_range < 1e-8:
                    norm_error = 0.0
                    norm_traj_error = 0.0
                else:
                    norm_error = np.mean(abs_error) / target_range
                    norm_traj_error = np.sum(abs_error) / target_range

                mae = np.mean(abs_error)
                sample_features.extend([overshoot, arrival_acc, norm_error, norm_traj_error, mae])

                # Smoothness / jerk measures
                jerk = np.gradient(acceleration)
                mean_jerk = np.mean(np.abs(jerk))
                jerk_vel_ratio = mean_jerk / avg_vel if avg_vel > 1e-8 else 0.0
                vel_shape = stats.kurtosis(velocity) if np.std(velocity) > 1e-8 else 0.0

                freqs = np.fft.fftfreq(len(velocity))
                fft_vals = np.abs(fft(velocity))
                spectral_arc = integrate.simpson(fft_vals, freqs)

                sample_features.extend([mean_jerk, jerk_vel_ratio, vel_shape, spectral_arc])

            # EMG-derived features (if present)
            if "emg" in feature_set:
                # Map indices within the selected channels
                # If EMG channels were selected they are present in X as contiguous indices
                # Here we assume channel ordering preserved: [Angle, Target, Ext, Flex, KM, KD, Torque?]
                # Find local indices for Ext and Flex if present
                # Note: feature_indices variable in outer scope defines the mapping used above
                local_indices = {feat: idx for idx, feat in enumerate(feature_indices)} if 'feature_indices' in locals() else None

                # Safely get channel arrays if present
                def safe_get(idx):
                    return X[i, :, idx] if idx is not None and idx >= 0 and idx < X.shape[2] else None

                # Determine presence by checking if absolute positions exist
                ext = None
                flex = None
                km = None
                kd = None
                try:
                    # attempt to map by original feature ids
                    if 2 in feature_indices:
                        ext = X[i, :, feature_indices.index(2)]
                    if 3 in feature_indices:
                        flex = X[i, :, feature_indices.index(3)]
                    if 4 in feature_indices:
                        km = X[i, :, feature_indices.index(4)]
                    if 5 in feature_indices:
                        kd = X[i, :, feature_indices.index(5)]
                except Exception:
                    # fallback: skip EMG features if mapping fails
                    ext = flex = km = kd = None

                # For raw EMG channels (ext and flex), compute stats
                for emg_signal in [ext, flex]:
                    if emg_signal is None:
                        continue
                    abs_signal = np.abs(emg_signal)
                    max_amp = np.max(emg_signal)
                    min_amp = np.min(emg_signal)
                    mean_amp = np.mean(abs_signal)
                    median_amp = np.median(abs_signal)
                    auc = integrate.simpson(abs_signal, dx=1)
                    rms = np.sqrt(np.mean(np.square(emg_signal)))
                    mean_power = np.mean(np.square(emg_signal))
                    mav = np.mean(abs_signal)
                    wl = np.sum(np.abs(np.diff(emg_signal)))
                    skew = stats.skew(emg_signal) if np.std(emg_signal) > 1e-8 else 0.0
                    kurt = stats.kurtosis(emg_signal) if np.std(emg_signal) > 1e-8 else 0.0
                    fractal_len = (np.sum(np.abs(np.diff(emg_signal))) / (np.max(emg_signal) - np.min(emg_signal) + 1e-8)) if (np.max(emg_signal) - np.min(emg_signal) + 1e-8) > 1e-8 else 0.0
                    ssi = np.sum(np.square(emg_signal))
                    aac = np.mean(np.abs(np.diff(emg_signal)))
                    emg_features = [max_amp, min_amp, mean_amp, median_amp, auc, rms, mean_power, mav, wl, skew, kurt, fractal_len, ssi, aac]
                    sample_features.extend(emg_features)

                # KM features
                if km is not None:
                    km_features = [
                        np.max(km), np.min(km), np.mean(km), np.std(km),
                        np.median(km), np.ptp(km), np.sqrt(np.mean(np.square(km)))
                    ]
                    sample_features.extend(km_features)

                # KD features
                if kd is not None:
                    kd_features = [
                        np.max(kd), np.min(kd), np.mean(kd), np.std(kd),
                        np.median(kd), np.ptp(kd), np.sqrt(np.mean(np.square(kd)))
                    ]
                    sample_features.extend(kd_features)

            # Torque features
            if "torque" in feature_set:
                try:
                    torque = X[i, :, feature_indices.index(6)]
                except Exception:
                    torque = None

                if torque is not None:
                    abs_channel = np.abs(torque)
                    max_val = np.max(torque)
                    min_val = np.min(torque)
                    mean_val = np.mean(torque)
                    std_val = np.std(torque)
                    median_val = np.median(torque)
                    ptp_val = np.ptp(torque)
                    rms = np.sqrt(np.mean(np.square(torque)))
                    auc = integrate.simpson(abs_channel, dx=1)
                    skew = stats.skew(torque) if np.std(torque) > 1e-8 else 0.0
                    kurt = stats.kurtosis(torque) if np.std(torque) > 1e-8 else 0.0
                    channel_features = [max_val, min_val, mean_val, std_val, median_val, ptp_val, rms, auc, skew, kurt]
                    sample_features.extend(channel_features)

            features.append(sample_features)

        return np.array(features)

    else:
        raise ValueError(f"Unknown feature extraction method: {method}")


# ===== Deep learning models =====
class MLP(nn.Module):
    """Simple multilayer perceptron for classification."""
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier that uses the last timestep output."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


# ===== Training & evaluation helpers =====
def train_classical_clf(X_train, y_train, clf_name):
    clf = Config.CLASSICAL_CLFS[clf_name]
    clf.fit(X_train, y_train)
    return clf


def train_dl_model(model, train_loader, val_loader, output_dim):
    """Train a PyTorch model and save best weights by validation accuracy."""
    model.to(Config.DL_DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.DL_LR)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(Config.DL_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(Config.DL_DEVICE)
            labels = labels.to(Config.DL_DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        val_acc = evaluate_dl_model(model, val_loader)
        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{Config.DL_EPOCHS}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    return model, history


def evaluate_dl_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(Config.DL_DEVICE)
            labels = labels.to(Config.DL_DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def evaluate_model(model, X_test, y_test, is_dl=False, test_loader=None):
    if is_dl:
        if test_loader is None:
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
            test_loader = DataLoader(test_dataset, batch_size=Config.DL_BATCH_SIZE)
        accuracy = evaluate_dl_model(model, test_loader)
        y_pred = []
        model.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(Config.DL_DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
    else:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {"accuracy": accuracy, "f1_macro": f1, "confusion_matrix": cm, "report": report}


def visualize_results(results, output_dir, feature_group, extractor, model_name):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=range(9), yticklabels=range(9))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (9 Conditions)")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    result_data = {"feature_group": [feature_group], "feature_extractor": [extractor], "model": [model_name], "accuracy": [results["accuracy"]], "f1_macro": [results["f1_macro"]], "timestamp": [pd.Timestamp.now().isoformat()]}
    df = pd.DataFrame(result_data)

    master_results_path = os.path.join(output_dir, "..", "all_results.csv")
    if os.path.exists(master_results_path):
        df.to_csv(master_results_path, mode="a", header=False, index=False)
    else:
        df.to_csv(master_results_path, index=False)

    exp_results_path = os.path.join(output_dir, "results.csv")
    df.to_csv(exp_results_path, index=False)

    report_df = pd.DataFrame(results["report"]).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    return df


# ===== Main program =====
def main():
    parser = argparse.ArgumentParser(description="End-to-end baseline experiments")
    parser.add_argument("--data_path", type=str,
                        default="data/processed/subject_"+f"{Config.SUBJECT_IDX}"+"/"+f"{Config.TRIAL_SET}"+"/subject_"+f"{Config.SUBJECT_IDX}"+"_processed.npz",
                        help="Path to preprocessed data (.npz)")
    parser.add_argument("--feature_set", type=str, default="kinematic",
                        choices=["kinematic", "emg", "kinematic+emg", "kinematic+emg+torque"],
                        help="Which feature channels to use")
    parser.add_argument("--feature_extractor", type=str, default="advanced_stats",
                        choices=["raw", "pca", "tsne", "stats", "advanced_stats"],
                        help="Feature extraction method")
    parser.add_argument("--model_type", type=str, default="all",
                        choices=["classical", "dl", "all"],
                        help="Type of models to run")
    args = parser.parse_args()

    # Prepare output directory
    if args.feature_set == "kinematic+emg+torque":
        exp_name = f"full_{args.feature_extractor}_{Config.TRIAL_SET}"
    else:
        exp_name = f"{args.feature_set}_{args.feature_extractor}_{Config.TRIAL_SET}"
    global output_dir
    output_dir = os.path.join("results", "subject_"+f"{Config.SUBJECT_IDX}", "end2end_baselines", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load preprocessed data and saved split indices
    data = np.load(args.data_path)
    X = data["data"]  # (n_samples, 250, 7)
    y = data["labels"]

    data_dir = os.path.dirname(args.data_path)
    split_indices = np.load(os.path.join(data_dir, "data_split_indices.npz"))
    train_idx = split_indices["train_idx"]
    val_idx = split_indices["val_idx"]
    test_idx = split_indices["test_idx"]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Feature extraction
    X_train_feat = extract_features(X_train, args.feature_extractor, feature_set=args.feature_set)
    X_val_feat = extract_features(X_val, args.feature_extractor, feature_set=args.feature_set)
    X_test_feat = extract_features(X_test, args.feature_extractor, feature_set=args.feature_set)

    # Standardize using training set statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    all_results = []

    # Choose models to run
    models_to_train = []
    if args.model_type in ["classical", "all"]:
        models_to_train.extend(Config.CLASSICAL_CLFS.keys())
    if args.model_type in ["dl", "all"]:
        models_to_train.extend(Config.DL_MODELS.keys())

    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training model: {model_name} ({args.feature_set}, {args.feature_extractor})")
        print(f"{'='*50}")

        if model_name in Config.CLASSICAL_CLFS:
            model = train_classical_clf(X_train_scaled, y_train, model_name)
            results = evaluate_model(model, X_test_scaled, y_test)

        elif model_name in Config.DL_MODELS:
            # LSTM uses raw time-series; MLP uses extracted features
            if model_name == "LSTM":
                feature_indices = []
                if "kinematic" in args.feature_set:
                    feature_indices.extend([0, 1])
                if "emg" in args.feature_set:
                    feature_indices.extend([2, 3, 4, 5])
                if "torque" in args.feature_set:
                    feature_indices.extend([6])

                X_train_model = X_train[:, :, feature_indices]
                X_val_model = X_val[:, :, feature_indices]
                X_test_model = X_test[:, :, feature_indices]

                print(f"LSTM input shape: {X_train_model.shape}")
            else:
                X_train_model = X_train_scaled
                X_val_model = X_val_scaled
                X_test_model = X_test_scaled

            # Build DataLoaders
            train_dataset = TensorDataset(torch.tensor(X_train_model, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            val_dataset = TensorDataset(torch.tensor(X_val_model, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
            test_dataset = TensorDataset(torch.tensor(X_test_model, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

            train_loader = DataLoader(train_dataset, batch_size=Config.DL_BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=Config.DL_BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=Config.DL_BATCH_SIZE)

            # Initialize model
            if model_name == "MLP":
                model = MLP(X_train_scaled.shape[1], 9)
            elif model_name == "LSTM":
                model = LSTMClassifier(input_dim=X_train_model.shape[2], hidden_dim=128, output_dim=9)

            # Train and evaluate
            model, history = train_dl_model(model, train_loader, val_loader, 9)
            results = evaluate_model(model, None, y_test, is_dl=True, test_loader=test_loader)

        # Print and save results
        print(f"Results:")
        print(f"- Accuracy: {results['accuracy']:.4f}")
        print(f"- F1 Macro: {results['f1_macro']:.4f}")

        model_output_dir = os.path.join(output_dir, model_name)
        result_df = visualize_results(results, model_output_dir, args.feature_set, args.feature_extractor, model_name)
        all_results.append(result_df)

    # Aggregate results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
        print("\nAll experiments completed! Results saved to:", output_dir)
    else:
        print("No models were trained.")


if __name__ == "__main__":
    main()
