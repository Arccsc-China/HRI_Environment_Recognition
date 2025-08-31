# Operational Environment Recognition for Human–Robot Collaboration

**(Human physiological sensory feedback → environmental-noise recognition)**

---

## Overview

This repository contains the codebase used to preprocess a wrist impedance dataset, extract features (raw, VAE, handcrafted statistics), train VAEs, and benchmark classifiers for inferring environmental **visual/haptic noise** from human physiological signals (kinematics, EMG, torque).
The README focuses on repository usage and reproduction of experiments (no research narrative).

---

## Quick pipeline summary

1. **Preprocess** raw CSV sensor data → segmented, interpolated, standardized `.npz` per subject.
2. **Train VAEs** (one VAE per feature channel) and save best models.
3. **Extract features** from VAEs (latent means) for train/val/test splits.
4. **Train classifiers** (classical ML + a small DL MLP) on extracted features.
5. Optionally run **end-to-end baselines** (raw / PCA / t-SNE / handcrafted stats + classical/DL classifiers).

Order to run for a single subject:

```bash
# 1. preprocess (select subject index and trial set)
python data_preprocessing_cleaned.py --subject_idx 0 --trial_set stable

# 2. train VAEs (edit Config.SUBJECT_IDX / TRIAL_SET in vae_training_cleaned.py as needed)
python vae_training_cleaned.py

# 3. extract features using the trained VAEs
python feature_extraction_cleaned.py --data_dir data/processed/subject_0/stable --model_dir results/subject_0/vae_stable --z_dim 32

# 4. train classifiers on extracted features
python classification_cleaned.py --features_dir results/subject_0/vae_stable/extracted_features --model_type all

# 5. run end-to-end baselines (alternative workflows)
python end2end_baselines_cleaned.py --data_path data/processed/subject_0/stable/subject_0_processed.npz \
    --feature_set kinematic --feature_extractor advanced_stats --model_type all
```

> NOTE: Some scripts use command-line args; others rely on `Config` constants inside the script (edit `Config` where noted). All scripts are included in cleaned form in the repository.

---

## Repository structure

```
/<repo-root>
│   README.md
│   requirements.txt
│   LICENSE
│
├── src/
│   ├── data_preprocessing_cleaned.py
│   ├── vae_training_cleaned.py
│   ├── feature_extraction_cleaned.py
│   ├── classification_cleaned.py
│   └── end2end_baselines_cleaned.py
│
└── data/
    ├── raw/
    │   ├── Kinematic/
    │   │   ├── Angle.csv
    │   │   └── Target.csv
    │   ├── EMG/
    │   │   ├── Ext.csv
    │   │   ├── Flex.csv
    │   │   ├── KM.csv
    │   │   └── KD.csv
    │   └── Torque/
    │       └── Torque_motor.csv
    │
    └── processed/
        ├── subject_0/
        │   ├── all/
        │   │   ├── subject_0_processed.npz
        │   │   ├── data_split_indices.npz
        │   │   ├── label_map.npy
        │   │   ├── means.npy
        │   │   └── stds.npy
        │   └── stable/
        │       ├── subject_0_processed.npz
        │       ├── data_split_indices.npz
        │       ├── label_map.npy
        │       ├── means.npy
        │       └── stds.npy
        └── subject_1/
            └── ...

```

---

## Data format & expected shapes

* Raw CSVs are reshaped to: **(2000, 22, 9, 9)** → `(time, subjects, conditions, trials)`.
* After preprocessing each sample is a segmented/interpolated sequence of length **250**. The main processed `.npz` contains:
  * `data`: shape `(num_samples, 250, n_channels)` — typically `n_channels == 7` (Angle, Target, Ext, Flex, KM, KD, Torque)
  * `labels`: shape `(num_samples,)` — integer condition labels 0..8
  * `data_split_indices.npz`: `train_idx`, `val_idx`, `test_idx` (saved indices used throughout the pipeline)
* Feature extraction scripts expect those `.npz` outputs and the split indices.

---

## Feature extraction methods implemented

* **Raw**: flattened preprocessed time-series (baseline).
* **VAE**: independent VAE per feature channel; latent mean `µ` extracted and concatenated across channels.
* **Advanced\_stats**: handcrafted motion & EMG statistics (velocity/acceleration/jerk, EMG RMS/AUC/mean-power/skew/kurtosis, torque stats, etc.).
* **PCA / t-SNE / stats**: additional baseline extractors supported by `end2end_baselines_cleaned.py`.

---

## Models & classifiers

* Classical: **SVM, RandomForest, LogisticRegression, KNN, XGBoost**, etc. (defined in `classification_cleaned.py`).
* Deep: small MLP classifier (used on latent or handcrafted features) and a bi-directional LSTM option (used on raw time-series in `end2end_baselines_cleaned.py`).

---

## Outputs

* Preprocessed `.npz` and normalization parameters (`means.npy`, `stds.npy`) saved in `data/processed/subject_<idx>/<trial_set>/`.
* VAE models & visualizations saved in `results/subject_<idx>/vae_<trial_set>/`.
* Extracted features saved to `results/.../extracted_features/` as `<feature_group>_{train,val,test}_features.npz`.
* Classification results, confusion matrices, training histories and aggregated CSVs in `results/.../classification_result/` or `results/.../end2end_baselines/`.

---

## Reproducibility notes

* Fixed random seeds are used where relevant (`RANDOM_STATE = 42`).
* Preprocessing saves split indices (`data_split_indices.npz`) — downstream steps load those indices to keep consistent train/val/test splits.
* Standardization uses **training-set** mean/std; the same parameters are applied to validation/test to avoid leakage.

---

## Requirements

Minimal (install via `pip`):

```
numpy scipy pandas scikit-learn matplotlib seaborn
torch torchvision    # appropriate CUDA/CuDNN if using GPU
xgboost
```

Recommended: Python 3.8+ and a CUDA-capable GPU for faster VAE / DL training (PyTorch optional for classical experiments).

---

## Tips & common options

* To run stable-trials-only experiments, either set `--trial_set stable` in preprocessing or use the corresponding `Config.TRIAL_SET` in scripts that rely on `Config`. Preprocessing can produce both full and stable datasets.
* If you change `Config.SUBJECT_IDX` across scripts, ensure consistency or run scripts in the order above using the same subject ID.
* Check `results/` for saved models (`best_model_*.pth`) and debugging visualizations (reconstructions, t-SNE / latent plots, confusion matrices).

---

## Contact

* For questions about code usage or to request scripts adapted to your dataset, open an issue or contact the repository owner.
