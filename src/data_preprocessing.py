import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
import argparse
from sklearn.model_selection import train_test_split

# ===== Configuration =====
parser = argparse.ArgumentParser(description='Data Preprocessing')
parser.add_argument('--subject_idx', type=int, 
                    default=1, 
                    help='Subject index (0-based)')
parser.add_argument('--trial_set', type=str, 
                    default='all', 
                    choices=['all', 'stable'], help='Which trials to use: all or stable (last 4)')
args = parser.parse_args()

SUBJECT_IDX = args.subject_idx
TRIAL_SET = args.trial_set
RAW_DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/subject_"+f"{SUBJECT_IDX}"+"/"+f"{TRIAL_SET}"
FILE_NAME = "subject_"+f"{SUBJECT_IDX}"+"_processed.npz"
L_SEGMENT = 250  # length after interpolation for each segment
MIN_PEAKS = 10  # minimum required number of peaks per trial
MIN_DISTANCE = 150  # minimum distance between peaks (in samples)
RANDOM_STATE = 42  # random seed for reproducibility

# Mapping of feature file relative paths
FEATURE_FILES = {
    "Angle": "Kinematic/Angle.csv",
    "Target": "Kinematic/Target.csv",
    "Ext": "EMG/Ext.csv",
    "Flex": "EMG/Flex.csv",
    "KM": "EMG/KM.csv",
    "KD": "EMG/KD.csv",
    "Torque": "Torque/Torque_motor.csv"
}

# Mapping from condition index to label (order matters)
CONDITION_LABELS = {
    0: 'V0H0', 1: 'V0H1', 2: 'V0H2',
    3: 'V1H0', 4: 'V1H1', 5: 'V1H2',
    6: 'V2H0', 7: 'V2H1', 8: 'V2H2'
}


def load_and_combine_data():
    """Load all feature files, reshape to the expected shape and extract the selected subject.

    Returns
    -------
    combined : dict
        Dictionary mapping feature name to an array of shape (2000, 9, n_trials)
        where n_trials is either 9 (all) or 4 (stable last 4 trials).
    """
    combined = {}
    for feat_name, rel_path in FEATURE_FILES.items():
        path = os.path.join(RAW_DATA_PATH, rel_path)
        data = pd.read_csv(path, header=None).values

        # Reshape to (time, subjects, conditions, trials)
        data = data.reshape((2000, 22, 9, 9))

        # Extract the selected subject: (2000, 9, 9)
        subject_data = data[:, SUBJECT_IDX, :, :]

        # If using only stable trials, select the last 4 trials (indices 5-8)
        if TRIAL_SET == 'stable':
            subject_data = subject_data[:, :, 5:9]

        combined[feat_name] = subject_data

    return combined


def find_target_peaks(target_data):
    """Detect peaks in the target signal for each condition and trial.

    Strategy:
    1. Use first derivative zero-crossings and check second-derivative sign to detect maxima.
    2. If too few peaks are found, fallback to scipy.find_peaks with distance and prominence.
    3. If still too few, use a theoretical waveform to predict peaks as a last resort.

    Returns a list of dicts with keys: 'condition', 'trial', 'peaks'.
    """
    all_peaks = []
    n_conditions = target_data.shape[1]
    n_trials = target_data.shape[2]

    for cond_idx in range(n_conditions):
        for trial_idx in range(n_trials):
            curve = target_data[:, cond_idx, trial_idx]

            # 1. First derivative
            dy = np.gradient(curve)

            # 2. Find zero crossings of derivative (candidate extrema)
            zero_crossings = np.where(np.diff(np.sign(dy)))[0]

            # 3. Keep local maxima by checking approximate second derivative
            peaks = []
            for idx in zero_crossings:
                if idx > 0 and idx < len(curve) - 1:
                    d2y = curve[idx-1] - 2*curve[idx] + curve[idx+1]
                    if d2y < 0:  # local maximum
                        peaks.append(idx)

            # 4. Fallback: use scipy.find_peaks if not enough peaks
            if len(peaks) < MIN_PEAKS:
                alt_peaks, _ = find_peaks(
                    curve,
                    distance=MIN_DISTANCE,
                    prominence=0.1 * (np.max(curve) - np.min(curve))
                )
                peaks = alt_peaks.tolist()

            # 5. Sort peaks in time order
            peaks.sort()

            # 6. Boundary handling: ensure start and end are represented
            if len(peaks) > 0:
                # If first peak is far from the start, include the start index
                if peaks[0] > 100:
                    peaks.insert(0, 0)

                # If last peak is far from the end, include the last index
                if peaks[-1] < len(curve) - 100:
                    peaks.append(len(curve) - 1)

            # 7. If still insufficient peaks, use a theoretical prediction as last resort
            if len(peaks) < MIN_PEAKS:
                print(f"Warning: Condition {CONDITION_LABELS[cond_idx]}, "
                      f"Trial {trial_idx} only has {len(peaks)} peaks. Using theoretical prediction.")

                t = np.arange(len(curve)) / 100.0  # assume 100 Hz sampling
                t0 = 0  # baseline shift (left as 0; adjust externally if needed)
                t_star = t + t0

                # Theoretical function used for prediction (kept as in original code)
                theoretical = 18.5 * np.sin(2.031 * t_star) * np.sin(1.093 * t_star)

                peaks, _ = find_peaks(
                    theoretical,
                    distance=MIN_DISTANCE,
                    prominence=0.1 * (np.max(theoretical) - np.min(theoretical))
                )

            # 8. Store peak indices and metadata (limit to MIN_PEAKS)
            all_peaks.append({
                'condition': cond_idx,
                'trial': trial_idx,
                'peaks': peaks[:MIN_PEAKS]
            })
    return all_peaks


def segment_and_interpolate(data_dict, peak_info):
    """Segment the signals between consecutive peaks and interpolate each segment to a fixed length.

    Returns
    -------
    segments : np.ndarray
        Array of shape (num_segments, L_SEGMENT, num_features)
    labels : np.ndarray
        Integer condition labels for each segment
    """
    segments = []
    labels = []

    for info in peak_info:
        cond_idx = info['condition']
        trial_idx = info['trial']
        peaks = info['peaks']

        if len(peaks) < 2:
            print(f"Skipping condition {CONDITION_LABELS[cond_idx]}, "
                  f"trial {trial_idx} - insufficient peaks ({len(peaks)})")
            continue

        # Create segments between consecutive peaks
        for seg_idx in range(len(peaks) - 1):
            start_idx = peaks[seg_idx]
            end_idx = peaks[seg_idx + 1]

            seg_length = end_idx - start_idx
            if seg_length < 10:  # skip segments that are too short
                continue

            seg_data = []
            for feat in FEATURE_FILES.keys():
                # Extract raw segment for this feature
                raw_segment = data_dict[feat][start_idx:end_idx+1, cond_idx, trial_idx]

                # Linearly interpolate to fixed length L_SEGMENT
                x_old = np.linspace(0, 1, len(raw_segment))
                x_new = np.linspace(0, 1, L_SEGMENT)
                interpolator = interp1d(x_old, raw_segment, kind='linear')
                seg_data.append(interpolator(x_new))

            # Stack features -> (L_SEGMENT, num_features)
            seg_array = np.vstack(seg_data).T
            segments.append(seg_array)
            labels.append(cond_idx)

    return np.array(segments), np.array(labels)


def standardize_data(data, train_indices):
    """Standardize data per-feature using statistics computed on the training set.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (num_samples, L_SEGMENT, num_features)
    train_indices : array-like
        Indices of samples belonging to the training set

    Returns
    -------
    standardized : np.ndarray
        Standardized data with same shape as input
    means : np.ndarray
        Per-feature means computed on the training data
    stds : np.ndarray
        Per-feature standard deviations computed on the training data
    """
    # Flatten training set across time to compute per-feature stats
    flattened = data[train_indices].reshape(-1, data.shape[-1])

    means = np.mean(flattened, axis=0)
    stds = np.std(flattened, axis=0)

    # Standardize all data (add small epsilon to avoid division by zero)
    standardized = (data - means) / (stds + 1e-8)

    return standardized, means, stds


def save_processed_data(data, labels, means, stds):
    """Utility helper to save processed arrays to disk (not used in main to preserve original structure).

    Creates the processed directory for the subject/trial set and writes compressed data and metadata.
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    np.savez_compressed(
        os.path.join(PROCESSED_PATH, FILE_NAME),
        data=data,
        labels=labels
    )
    np.save(os.path.join(PROCESSED_PATH, "means.npy"), means)
    np.save(os.path.join(PROCESSED_PATH, "stds.npy"), stds)

    label_map = {i: CONDITION_LABELS[i] for i in range(9)}
    np.save(os.path.join(PROCESSED_PATH, "label_map.npy"), label_map)


def main():
    print("===== Starting data preprocessing =====")
    print(f"Trial set used: {TRIAL_SET}")

    # 1. Load and combine raw data for the selected subject
    print("Loading raw data...")
    raw_data = load_and_combine_data()

    # 2. Detect peaks in the target signal per condition/trial
    print("Detecting peaks in target signal...")
    peak_info = find_target_peaks(raw_data['Target'])

    # 3. Segment signals between peaks and interpolate each segment
    print("Segmenting signals and interpolating to fixed length...")
    segmented_data, labels = segment_and_interpolate(raw_data, peak_info)

    # 4. Split dataset into train/val/test (70% / 15% / 15%) with stratification
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=RANDOM_STATE, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_STATE, stratify=labels[temp_idx])

    # 5. Standardize using training set statistics
    print("Standardizing data using training-set statistics...")
    std_data, means, stds = standardize_data(segmented_data, train_idx)

    # 6. Save processed arrays and metadata
    print("Saving processed data to disk...")
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    np.savez_compressed(
        os.path.join(PROCESSED_PATH, FILE_NAME),
        data=std_data,
        labels=labels
    )

    # Save split indices and normalization parameters for reproducibility
    np.savez(os.path.join(PROCESSED_PATH, "data_split_indices.npz"),
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    np.save(os.path.join(PROCESSED_PATH, "means.npy"), means)
    np.save(os.path.join(PROCESSED_PATH, "stds.npy"), stds)

    # Save label mapping
    label_map = {i: CONDITION_LABELS[i] for i in range(9)}
    np.save(os.path.join(PROCESSED_PATH, "label_map.npy"), label_map)

    print(f"Processing finished! Number of samples: {len(labels)}")
    print(f"Data shape: {std_data.shape}")


if __name__ == "__main__":
    main()
