import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===== Configuration =====
class Config:
    SUBJECT_IDX = 21  # 0-21
    TRIAL_SET = 'stable'  # 'all' or 'stable'

    # Classical classifiers to evaluate
    CLASSIFIERS = {
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, multi_class="multinomial", solver="saga", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    }
    
    # Evaluation metrics mapping
    METRICS = {
        "Accuracy": accuracy_score,
        "F1_Macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
    }
    
    # Training parameters for deep learning classifier
    RANDOM_STATE = 42
    DL_EPOCHS = 100
    DL_BATCH_SIZE = 64
    DL_LR = 0.001
    DL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class labels (noise conditions)
    CLASS_LABELS = ["V0H0", "V0H1", "V0H2", "V1H0", "V1H1", "V1H2", "V2H0", "V2H1", "V2H2"]
    
    # Feature group names (consistent with feature_extraction outputs)
    FEATURE_GROUPS = ["kinematic", "emg", "kinematic_emg", "full"]


# ===== Deep learning MLP classifier for latent features =====
class FeatureMLP(nn.Module):
    """MLP classifier used to classify concatenated VAE latent features."""
    def __init__(self, input_dim, output_dim):
        super(FeatureMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)


# ===== Training and evaluation utilities =====
def load_features(features_dir, feature_group):
    """Load per-set (train/val/test) feature files for the requested feature group.

    Expects files named: <feature_group>_train_features.npz, <feature_group>_val_features.npz, <feature_group>_test_features.npz
    Each file should contain arrays 'features' and 'labels'.
    """
    train_data = np.load(os.path.join(features_dir, f"{feature_group}_train_features.npz"))
    val_data = np.load(os.path.join(features_dir, f"{feature_group}_val_features.npz"))
    test_data = np.load(os.path.join(features_dir, f"{feature_group}_test_features.npz"))
    
    X_train = train_data["features"]
    y_train = train_data["labels"]
    X_val = val_data["features"]
    y_val = val_data["labels"]
    X_test = test_data["features"]
    y_test = test_data["labels"]
    
    print(f"Loaded feature group: {feature_group}, Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_classical_clf(X_train, y_train, clf_name):
    """Train a classical ML classifier (scikit-learn style)."""
    clf = Config.CLASSIFIERS[clf_name]
    clf.fit(X_train, y_train)
    return clf


def train_dl_model(model, X_train, y_train, X_val, y_val, feature_group):
    """Train the deep learning MLP classifier and save best model by validation accuracy."""
    model.to(Config.DL_DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.DL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Create PyTorch datasets and loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.DL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.DL_BATCH_SIZE)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}
    
    for epoch in range(Config.DL_EPOCHS):
        # Training loop
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
        
        # Validation accuracy
        val_acc = evaluate_dl_model(model, val_loader)
        scheduler.step(val_acc)
        
        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{Config.DL_EPOCHS}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_dl_model_{feature_group}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_dl_model_{feature_group}.pth"))
    return model, history


def evaluate_dl_model(model, data_loader):
    """Compute accuracy for a PyTorch model on a DataLoader."""
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
    """Evaluate either a classical sklearn model or a trained PyTorch model and return metrics."""
    if is_dl:
        # If not provided, build a test DataLoader
        if test_loader is None:
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
            test_loader = DataLoader(test_dataset, batch_size=Config.DL_BATCH_SIZE)
        accuracy = evaluate_dl_model(model, test_loader)
        
        # Gather predictions
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
    
    # Additional metrics and report
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "report": report,
        "y_true": y_test,
        "y_pred": y_pred
    }


def visualize_results(results, output_dir, model_name, feature_group):
    """Save confusion matrix, CSV reports and training history (if available)."""
    # Create per-feature-group output folder
    group_output_dir = os.path.join(output_dir, feature_group)
    os.makedirs(group_output_dir, exist_ok=True)
    
    # Per-model folder
    model_output_dir = os.path.join(group_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 1) Confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=Config.CLASS_LABELS,
                yticklabels=Config.CLASS_LABELS)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name} ({feature_group})")
    plt.savefig(os.path.join(model_output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()
    
    # 2) Save summary CSV row
    result_data = {
        "feature_group": [feature_group],
        "model": [model_name],
        "accuracy": [results["accuracy"]],
        "f1_macro": [results["f1_macro"]],
        "timestamp": [pd.Timestamp.now().isoformat()]
    }
    df = pd.DataFrame(result_data)
    master_results_path = os.path.join(output_dir, "all_results.csv")
    if os.path.exists(master_results_path):
        df.to_csv(master_results_path, mode="a", header=False, index=False)
    else:
        df.to_csv(master_results_path, index=False)
    
    # 3) Save per-model CSV
    model_results_path = os.path.join(model_output_dir, "results.csv")
    df.to_csv(model_results_path, index=False)
    
    # 4) Save detailed classification report
    report_df = pd.DataFrame(results["report"]).transpose()
    report_df.to_csv(os.path.join(model_output_dir, "classification_report.csv"))
    
    # 5) Training history plots (if present)
    if "history" in results:
        history = results["history"]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history["val_acc"], 'r-', label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(f"Training History - {model_name} ({feature_group})")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(model_output_dir, "training_history.png"))
        plt.close()
    
    return df


# ===== Main program =====
def main():
    parser = argparse.ArgumentParser(description="Train classifiers on VAE-extracted features")
    parser.add_argument("--features_dir", type=str,
                        default="results/subject_"+f"{Config.SUBJECT_IDX}"+"/vae_"+f"{Config.TRIAL_SET}"+"/extracted_features",
                        help="Directory with extracted VAE features (per-set .npz files)")
    parser.add_argument("--model_type", type=str, default="all",
                        choices=["classical", "dl", "all"],
                        help="Type of models to train: classical (scikit-learn), dl (PyTorch), or all")
    args = parser.parse_args()
    
    # Create output directory
    global output_dir
    output_dir = os.path.join("results", "subject_"+f"{Config.SUBJECT_IDX}", "vae_"+f"{Config.TRIAL_SET}", "classification_result")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []

    # Iterate over feature groups
    for feature_group in Config.FEATURE_GROUPS:
        print(f"\n{'='*50}")
        print(f"Processing feature group: {feature_group}")
        print(f"{'='*50}")
        
        # 1) Load per-set features
        train_set, val_set, test_set = load_features(args.features_dir, feature_group)
        X_train, y_train = train_set
        X_val, y_val = val_set
        X_test, y_test = test_set
        
        # 2) Standardize using training set stats
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 3) Build list of models to train
        models_to_train = []
        if args.model_type in ["classical", "all"]:
            models_to_train.extend(Config.CLASSIFIERS.keys())
        if args.model_type in ["dl", "all"]:
            models_to_train.append("FeatureMLP")
        
        for model_name in models_to_train:
            print(f"\n{'='*30}")
            print(f"Training model: {model_name} ({feature_group})")
            print(f"{'='*30}")
            
            results = {}
            
            # Classical ML models
            if model_name in Config.CLASSIFIERS:
                print(f"Training classical classifier: {model_name}")
                model = train_classical_clf(X_train_scaled, y_train, model_name)
                eval_results = evaluate_model(model, X_test_scaled, y_test)
                results.update(eval_results)
            
            # Deep learning model
            elif model_name == "FeatureMLP":
                print(f"Training deep learning model: {model_name}")
                input_dim = X_train_scaled.shape[1]
                model = FeatureMLP(input_dim, 9)
                model, history = train_dl_model(model, X_train_scaled, y_train, X_val_scaled, y_val, feature_group)
                
                test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
                test_loader = DataLoader(test_dataset, batch_size=Config.DL_BATCH_SIZE)
                
                eval_results = evaluate_model(model, None, y_test, is_dl=True, test_loader=test_loader)
                results["history"] = history
                results.update(eval_results)
            
            # Save results and visualizations
            result_df = visualize_results(results, output_dir, model_name, feature_group)
            all_results.append(result_df)
    
    # 4) Aggregate and save summary results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        summary_path = os.path.join(output_dir, "summary_results.csv")
        final_df.to_csv(summary_path, index=False)
        
        # Accuracy comparison plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x="model", y="accuracy", hue="feature_group", data=final_df, palette="viridis")
        plt.title("Comparison of classifier accuracy for different feature groups")
        plt.ylabel("accuracy")
        plt.xlabel("model")
        plt.xticks(rotation=45)
        plt.legend(title="feature group", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
        plt.close()
        
        # Feature-group performance summary (max F1 macro)
        plt.figure(figsize=(12, 6))
        group_perf = final_df.groupby("feature_group")["f1_macro"].max().reset_index()
        sns.barplot(x="feature_group", y="f1_macro", data=group_perf, palette="rocket")
        plt.title("Max F1 Macro for each Feature Group")
        plt.ylabel("max F1 Macro")
        plt.xlabel("feature group")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_group_performance.png"))
        plt.close()
        
        # Print best model summary
        best_result = final_df.loc[final_df['accuracy'].idxmax()]
        print("\n" + "="*50)
        print(f"Best model: {best_result['model']} ({best_result['feature_group']})")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print("="*50)
        
        print("\nAll experiments completed! Results saved to:", output_dir)
        print(f"Summary CSV: {summary_path}")
    else:
        print("No models were trained.")


if __name__ == "__main__":
    main()
