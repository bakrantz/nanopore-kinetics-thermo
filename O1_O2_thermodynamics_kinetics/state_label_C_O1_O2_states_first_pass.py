import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# --- 1. ATF File Loading ---
def load_atf(filepath: str, header_row_index: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Loads data from an .atf file, preserving the multi-line header."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at '{filepath}'")

    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    if len(all_lines) < header_row_index + 2:
        raise ValueError(f"File '{filepath}' is too short.")

    header_lines = [line.strip('\n') for line in all_lines[:header_row_index + 1]]

    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=header_row_index)
        df.columns = df.columns.str.strip().str.replace(' #', '').str.replace(' ', '_').str.replace('[()]', '', regex=True)

        required_cols = {"Time_s": "Time (s)", "Trace1_pA": "Trace #1 (pA)", "Trace1_mV": "Trace #1 (mV)"}
        for cleaned_name, original_name in required_cols.items():
            if cleaned_name not in df.columns:
                raise KeyError(f"Expected column '{original_name}' not found. Available: {df.columns.tolist()}")

        times = df["Time_s"].to_numpy()
        current = df["Trace1_pA"].to_numpy()
        voltage = df["Trace1_mV"].to_numpy()

        return times, current, voltage, header_lines

    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading ATF '{filepath}': {e}")


# --- 2. Baseline Drift Correction ---
def dynamic_baseline_correction(current_trace_data: np.ndarray, window_size: int = 50, threshold_std_dev: float = 3.0, n_clusters_for_baseline_detection: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Corrects baseline drift by zeroing the dominant open state."""
    if current_trace_data.size == 0:
        return np.array([]), np.array([])
    
    kmeans_baseline = KMeans(n_clusters=n_clusters_for_baseline_detection, n_init='auto', random_state=42)
    kmeans_baseline.fit(current_trace_data.reshape(-1, 1))
    
    # Identify the dominant centroid (baseline)
    dominant_centroid = np.min(kmeans_baseline.cluster_centers_).flatten()[0]
    std_dev = np.std(current_trace_data)
    is_baseline = np.abs(current_trace_data - dominant_centroid) < threshold_std_dev * std_dev
    
    drift_points, time_points = [], []
    for i in range(0, len(current_trace_data), window_size):
        window_end = i + window_size
        window_slice = is_baseline[i:window_end]
        
        if np.any(window_slice):
            drift_points.append(np.mean(current_trace_data[i:window_end][window_slice]))
            time_points.append(i + window_size / 2)
        elif len(drift_points) > 0:
            drift_points.append(drift_points[-1])
            time_points.append(i + window_size / 2)
        else:
            drift_points.append(dominant_centroid)
            time_points.append(i + window_size / 2)

    drift_points = np.array(drift_points)
    time_points = np.array(time_points)

    drift_interpolator = interp1d(time_points, drift_points, kind='linear', fill_value='extrapolate')
    drift_vector = drift_interpolator(np.arange(len(current_trace_data)))
    
    corrected_current = current_trace_data - drift_vector
    return corrected_current, drift_vector


# --- 3. State Identification ---
def identify_conductance_states(current_trace_data: np.ndarray, n_states: int, initial_centroids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Classifies points based on initial centroid proximity."""
    X = current_trace_data.reshape(-1, 1)
    distances = np.abs(X - initial_centroids)
    labels = np.argmin(distances, axis=1)
    
    final_centroids = np.zeros(n_states)
    for i in range(n_states):
        points = current_trace_data[labels == i]
        final_centroids[i] = np.mean(points) if points.size > 0 else initial_centroids[i]
            
    sorted_indices = np.argsort(final_centroids)
    sorted_centroids = final_centroids[sorted_indices]
    
    old_to_new_map = {old: new for new, old in enumerate(sorted_indices)}
    mapped_labels = np.array([old_to_new_map[label] for label in labels])
    
    return sorted_centroids, mapped_labels


# --- 4. CSV Export ---
def export_labeled_csv(filepath: str, times: np.ndarray, filtered_current: np.ndarray, labels: np.ndarray, output_dir: str):
    """Exports Time, Filtered Current, and States to CSV with Windows/Excel compatibility."""
    file_basename = os.path.splitext(os.path.basename(filepath))[0]
    csv_filepath = os.path.join(output_dir, f"{file_basename}_labeled.csv")

    df = pd.DataFrame({
        'Time': times,
        'Filtered_Current': filtered_current,
        'State': labels
    })

    os.makedirs(output_dir, exist_ok=True)
    
    # Use utf-8-sig to ensure Excel on Windows 11 reads special characters correctly
    df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    
    return csv_filepath

# --- 5. Visualizations ---
def visualize_results(times, raw_current, drift_vector, filtered_current, centroids, title_suffix):
    """Plots Time-Series trace and Log-Scale Histogram."""
    # Plot 1: Time Series (Raw vs Filtered)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(times, raw_current, 'b-', label='Raw Current', alpha=0.5)
    ax1.plot(times, drift_vector, 'r-', label='Drift Curve', linewidth=2)
    ax1.set_title(f'Raw Data & Baseline Drift ({title_suffix})')
    ax1.set_ylabel('Current (pA)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(times, filtered_current, 'g-', label='Filtered & Corrected', alpha=0.8)
    ax2.set_title('Drift-Corrected & Gaussian-Filtered Trace')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (pA)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot 2: Log-Scale Histogram with Centroids
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(filtered_current, bins=200, color='skyblue', alpha=0.7, log=True, label='Current Dist.')
    
    colors = ['red', 'green', 'orange', 'purple']
    for i, c in enumerate(centroids):
        plt.axvline(c, color=colors[i % len(colors)], linestyle='--', linewidth=2, 
                    label=f'State {i} Centroid: {c:.2f} pA')
    
    plt.title(f'Log-Scale Histogram with {len(centroids)} States ({title_suffix})')
    plt.xlabel('Filtered Current (pA) [Open=0]')
    plt.ylabel('Log(Count)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()


# --- 6. Batch Processor ---
def batch_processor(filepaths, initial_centroids_map, output_dir, window_size=4000, 
                    threshold_std_dev=1.0, n_clusters_baseline=3, fs=600, filter_cutoff_hz=30, visualize=True):
    """Processes a batch of ATF files with Gaussian filtering and baseline correction."""
    os.makedirs(output_dir, exist_ok=True)
    log_data = []

    # Calculate Gaussian Sigma based on sampling rate and desired cutoff
    sigma = fs / (2 * np.pi * filter_cutoff_hz)
    print(f"Applying Gaussian Filter: Cutoff={filter_cutoff_hz}Hz, Sigma={sigma:.2f} (fs={fs}Hz)")

    for filepath in filepaths:
        file_basename = os.path.basename(filepath)
        log_entry = {'atf_filename': file_basename, 'csv_filename': 'N/A', 'success': False, 'error': '', 'centroids': 'N/A'}

        try:
            initial_centroids = initial_centroids_map.get(file_basename)
            if initial_centroids is None:
                raise ValueError("No initial centroids defined for file.")

            print(f"\n--- Processing {file_basename} ---")
            times, current, _, _ = load_atf(filepath)
            
            # 1. Baseline Correction
            corrected_current, drift_vector = dynamic_baseline_correction(
                current, window_size=window_size, threshold_std_dev=threshold_std_dev, n_clusters_for_baseline_detection=n_clusters_baseline
            )
            
            # 2. Gaussian Filtering
            filtered_current = gaussian_filter1d(corrected_current, sigma=sigma)
            
            # 3. State Identification
            n_states = len(initial_centroids)
            centroids, labels = identify_conductance_states(filtered_current, n_states, initial_centroids)

            # Optional: Reverse labels if you want 0=Closed, N=Open
            # reversed_labels = (n_states - 1) - labels

            # 4. Export
            csv_filepath = export_labeled_csv(filepath, times, filtered_current, labels, output_dir)
            
            log_entry.update({'success': True, 'csv_filename': os.path.basename(csv_filepath), 
                              'centroids': ', '.join([f"{c:.2f}" for c in centroids])})
            print(f"Identified Centroids (pA): {centroids}")

            # 5. Visualize
            if visualize:
                visualize_results(times, current, drift_vector, filtered_current, centroids, title_suffix=file_basename)

        except Exception as e:
            log_entry.update({'success': False, 'error': str(e)})
            print(f"Error processing {file_basename}: {e}")
        
        log_data.append(log_entry)

    pd.DataFrame(log_data).to_csv(os.path.join(output_dir, 'processing_log.csv'), index=False)
    print("\nBatch processing complete.")


# --- 7. Main Execution ---
if __name__ == "__main__":
    output_dir = "./PA/guesthost_Tyr_processed/"
    base_data_dir = "/Users/bakrantz/Documents/python/database/raw_data/PA/guesthost_Tyr/" 
    
    # Truncated list for example (add your full list back here)
    atf_filepaths_text = """
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf
    """
    atf_filepaths = [os.path.join(base_data_dir, line.strip()) for line in atf_filepaths_text.splitlines() if line.strip()]

    # Define centroids assuming Open State is zeroed by the drift correction
    # Adjust these based on your specific DOC O1/O2/C separation
    centroids_3state = np.array([-4.7, -0.8, 0.0]) # Example: C, O1, O2
    
    initial_centroids_map = {os.path.basename(f): centroids_3state for f in atf_filepaths}

    batch_processor(
        filepaths=atf_filepaths,
        initial_centroids_map=initial_centroids_map,
        output_dir=output_dir,
        fs=600,                 # Current acquisition rate
        filter_cutoff_hz=30,    # Target low-pass cutoff (10-50Hz)
        window_size=4000,
        n_clusters_baseline=3,
        visualize=True
    )
