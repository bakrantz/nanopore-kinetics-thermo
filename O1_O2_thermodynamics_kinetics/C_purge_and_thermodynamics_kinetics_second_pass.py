import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants for Eyring Equation
T = 298.15 # Room temperature in Kelvin
R = 1.987e-3 # Gas constant in kcal / (mol K)
k_B = 1.380649e-23 # Boltzmann constant J/K
h = 6.62607015e-34 # Planck constant J*s
PREFACTOR = (k_B * T) / h # approx 6.2e12 s^-1

def purge_c_states(states, c_label, o1_label, o2_label):
    """
    Purges C states by merging them into adjacent states.
    Handles the four edge cases: O1-C-O1, O2-C-O2, O1-C-O2, O2-C-O1.
    """
    purged = states.copy()
    n = len(purged)
    
    # Find contiguous runs of states
    changes = np.concatenate(([0], np.where(purged[:-1] != purged[1:])[0] + 1, [n]))
    
    for i in range(len(changes) - 1):
        start = changes[i]
        end = changes[i+1]
        val = purged[start]
        
        if val == c_label:
            length = end - start
            
            # Find previous and next valid states (ignore boundary limits)
            prev_val = purged[start - 1] if start > 0 else None
            next_val = purged[end] if end < n else None
            
            # Handle boundary C-states
            if prev_val is None and next_val is not None:
                purged[start:end] = next_val
                continue
            if next_val is None and prev_val is not None:
                purged[start:end] = prev_val
                continue
            if prev_val is None and next_val is None:
                continue # Entire trace is C state (unlikely)
                
            # Case a & b: Flanked by the same state
            if prev_val == next_val:
                purged[start:end] = prev_val
                
            # Case c & d: Flanked by different states
            else:
                half = length // 2
                purged[start:start+half] = prev_val
                purged[start+half:end] = next_val
                
    return purged

def extract_dwell_times(times, states, target_state):
    """Calculates the dwell times (in seconds) spent in a target state."""
    dt = times[1] - times[0]
    is_target = (states == target_state)
    
    # Find runs of the target state
    changes = np.concatenate(([0], np.where(is_target[:-1] != is_target[1:])[0] + 1, [len(is_target)]))
    
    dwells = []
    for i in range(len(changes) - 1):
        start = changes[i]
        end = changes[i+1]
        if is_target[start]:
            dwell_time = (end - start) * dt
            dwells.append(dwell_time)
            
    return np.array(dwells)

def fit_exponential_cdf(dwells):
    """Fits a 1 - exp(-k*t) cumulative distribution to the dwell times."""
    if len(dwells) < 5:
        return np.nan, np.nan, np.nan, np.nan # Not enough data
        
    sorted_dwells = np.sort(dwells)
    y_empirical = np.arange(1, len(sorted_dwells) + 1) / len(sorted_dwells)
    
    # Model: P(t <= T) = 1 - exp(-k * t)
    def cdf_model(t, k):
        return 1 - np.exp(-k * t)
        
    # Initial guess for k is 1 / mean(dwell time)
    k_guess = 1.0 / np.mean(sorted_dwells)
    
    try:
        popt, pcov = curve_fit(cdf_model, sorted_dwells, y_empirical, p0=[k_guess])
        k_fit = popt[0]
        # Eyring Activation Energy (kcal/mol)
        dg_dagger = -R * T * np.log(k_fit / PREFACTOR)
        return k_fit, dg_dagger, sorted_dwells, y_empirical
    except Exception as e:
        print(f"Fit failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

# --- FIXED FUNCTION ---
def process_beautification_and_kinetics(metadata_row, input_dir, output_dir, c_label=2, o1_label=1, o2_label=0):
    """Executes Steps 1-4 for a single file and returns data for Step 5."""
    filename = metadata_row['filename']
    basename = os.path.splitext(filename)[0]
    
    # Expected labeled file from Pass 1
    input_csv = os.path.join(input_dir, f"{basename}_labeled.csv")
    if not os.path.exists(input_csv):
        print(f"Warning: {input_csv} not found. Skipping.")
        return None
        
    # Step 1: Open Data
    df = pd.read_csv(input_csv)
    times = df['Time'].to_numpy()
    current = df['Filtered_Current'].to_numpy()
    raw_states = df['State'].to_numpy() 
    
    # Step 2: Beautify / Purge C-states
    purged_states = purge_c_states(raw_states, c_label=c_label, o1_label=o1_label, o2_label=o2_label)
    
    # Save Purged CSV
    purged_filename = f"{basename}_C_purged.csv"
    purged_filepath = os.path.join(output_dir, purged_filename)
    pd.DataFrame({'Time': times, 'Current': current, 'State': purged_states}).to_csv(purged_filepath, index=False)
    
    # Plotting Step 2 (Fixed y-axis mapping)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(times, current, 'b-', alpha=0.7)
    ax1.set_ylabel('Current (pA)')
    ax1.set_title(f"C-State Purged Trace: {basename}")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(times, purged_states, 'g-', alpha=0.8)
    ax2.set_yticks([o2_label, o1_label]) 
    ax2.set_yticklabels(['O2', 'O1'])
    ax2.set_ylabel('State')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_purged_trace.png"))
    plt.close()
    
    # Step 3: Thermodynamics (Equilibrium) - FIXED
    count_o1 = np.sum(purged_states == o1_label)
    count_o2 = np.sum(purged_states == o2_label)
    
    K_eq = count_o2 / count_o1 if count_o1 > 0 else np.nan
    DG_eq = -R * T * np.log(K_eq) if K_eq > 0 else np.nan
    
    # Step 4: Kinetics & Dwell Times - FIXED
    dwells_o1 = extract_dwell_times(times, purged_states, o1_label) # Time in O1 before O2
    dwells_o2 = extract_dwell_times(times, purged_states, o2_label) # Time in O2 before O1
    
    k_12, dg_dag_12, t_12, cdf_12 = fit_exponential_cdf(dwells_o1)
    k_21, dg_dag_21, t_21, cdf_21 = fit_exponential_cdf(dwells_o2)
    
    # Export and Plot CDFs
    if not np.isnan(k_12) and not np.isnan(k_21):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # O1 -> O2 Plot
        axes[0].plot(t_12, cdf_12, 'bo', alpha=0.5, label='Empirical CDF')
        axes[0].plot(t_12, 1 - np.exp(-k_12 * t_12), 'r-', lw=2, label=f'Fit: k={k_12:.2f} s⁻¹')
        axes[0].set_title(f"O1 -> O2 Kinetics\nΔG‡ = {dg_dag_12:.2f} kcal/mol")
        axes[0].set_xlabel('Dwell Time in O1 (s)')
        axes[0].set_ylabel('Probability')
        axes[0].legend()
        axes[0].grid(True, linestyle='--')
        
        # Save O1->O2 Data
        pd.DataFrame({'Dwell_Time_s': t_12, 'Empirical_CDF': cdf_12, 'Fit_CDF': 1 - np.exp(-k_12 * t_12)}).to_csv(
            os.path.join(output_dir, f"{basename}_O1_to_O2_CDF_fit.csv"), index=False)

        # O2 -> O1 Plot
        axes[1].plot(t_21, cdf_21, 'go', alpha=0.5, label='Empirical CDF')
        axes[1].plot(t_21, 1 - np.exp(-k_21 * t_21), 'r-', lw=2, label=f'Fit: k={k_21:.2f} s⁻¹')
        axes[1].set_title(f"O2 -> O1 Kinetics\nΔG‡ = {dg_dag_21:.2f} kcal/mol")
        axes[1].set_xlabel('Dwell Time in O2 (s)')
        axes[1].legend()
        axes[1].grid(True, linestyle='--')
        
        # Save O2->O1 Data
        pd.DataFrame({'Dwell_Time_s': t_21, 'Empirical_CDF': cdf_21, 'Fit_CDF': 1 - np.exp(-k_21 * t_21)}).to_csv(
            os.path.join(output_dir, f"{basename}_O2_to_O1_CDF_fit.csv"), index=False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{basename}_CDF_fits.png"))
        plt.close()

    # Step 5 Dictionary Row Creation
    summary_row = metadata_row.to_dict()
    summary_row.update({
        'First_Pass_CSV': f"{basename}_labeled.csv",
        'C_Purged_CSV': purged_filename,
        'K_eq': K_eq,
        'DG_eq_kcal_mol': DG_eq,
        'k_O1_to_O2_s-1': k_12,
        'k_O2_to_O1_s-1': k_21,
        'DG_dagger_O1_to_O2_kcal_mol': dg_dag_12,
        'DG_dagger_O2_to_O1_kcal_mol': dg_dag_21
    })
    
    return summary_row

# --- Main Execution Block ---
if __name__ == "__main__":
    # Paths for Mac environment (Update to Jaylen's paths if needed)
    base_dir = "./"
    input_dir = os.path.join(base_dir, "Super_O2_processed")
    output_dir = os.path.join(base_dir, "Super_O2_kinetics")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_csv_path = os.path.join(base_dir, "experiment_metadata.csv")
    df_meta = pd.read_csv(metadata_csv_path)
    summary_results = []
    
    for index, row in df_meta.iterrows():
        print(f"Analyzing kinetics for: {row['filename']}")
        # Pass the labels explicitly to the function here
        result = process_beautification_and_kinetics(row, input_dir, output_dir, c_label=2, o1_label=1, o2_label=0)
        if result:
            summary_results.append(result)
            
    # Step 5: Export Summary DataFrame
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        
        # Reorder columns to put metadata first, then the calculated metrics
        meta_cols = list(df_meta.columns)
        calc_cols = ['First_Pass_CSV', 'C_Purged_CSV', 'K_eq', 'DG_eq_kcal_mol', 
                     'k_O1_to_O2_s-1', 'k_O2_to_O1_s-1', 'DG_dagger_O1_to_O2_kcal_mol', 'DG_dagger_O2_to_O1_kcal_mol']
        df_summary = df_summary[meta_cols + calc_cols]
        
        summary_out_path = os.path.join(output_dir, "Thermodynamic_Kinetic_Summary.csv")
        df_summary.to_csv(summary_out_path, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully generated Kinetic Summary Dataframe: {summary_out_path}")
