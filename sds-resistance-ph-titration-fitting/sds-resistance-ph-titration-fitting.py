import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cooperative_ph_assembly(pH, pKa, n):
    """
    Standard Hill equation for pH-dependent fractional assembly.
    Fraction = 1 / (1 + 10^(n * (pH - pKa)))
    For PA: high assembly at low pH, low assembly at high pH.
    """
    return 1.0 / (1.0 + 10**(n * (pH - pKa)))

def fit_monomer_densitometry(pH_values, pa63_intensities, title_suffix=""):
    """
    Takes PA-63 monomer band intensities, inverts them to calculate 
    pore assembly, fits the data, and plots the result.
    """
    # 1. Normalize the PA-63 monomer data (0 to 1 scale)
    base_monomer = np.min(pa63_intensities)
    max_monomer = np.max(pa63_intensities)
    monomer_norm = (pa63_intensities - base_monomer) / (max_monomer - base_monomer)
    
    # 2. Invert the math: Fractional Assembly = 1.0 - Fractional Monomer
    assembly_norm = 1.0 - monomer_norm
    
    # 3. Initial parameter guesses
    # Guess pKa is roughly where assembly is 0.5
    idx_closest_to_half = np.argmin(np.abs(assembly_norm - 0.5))
    pKa_guess = pH_values[idx_closest_to_half]
    n_guess = 2.0 # Assume positive cooperativity
    
    # 4. Perform the non-linear curve fit on the Assembly data
    try:
        popt, pcov = curve_fit(
            cooperative_ph_assembly, 
            pH_values, 
            assembly_norm, 
            p0=[pKa_guess, n_guess],
            bounds=([4.0, 0.1], [10.0, 10.0]) # Logical bounds for pKa and n
        )
        pKa_fit, n_fit = popt
        
        # Standard errors
        perr = np.sqrt(np.diag(pcov))
        pKa_err, n_err = perr[0], perr[1]
        
    except Exception as e:
        print(f"Fit failed: {e}")
        return
    
    # 5. Generate high-resolution curve for plotting
    pH_smooth = np.linspace(min(pH_values) - 0.5, max(pH_values) + 0.5, 200)
    fit_curve_norm = cooperative_ph_assembly(pH_smooth, *popt)
    
    # 6. Visualization
    plt.figure(figsize=(9, 6))
    
    # Plot the inverted data points (Pore Assembly)
    plt.plot(pH_values, assembly_norm, 'bo', markersize=9, markeredgecolor='black', 
             label='Calculated Pore Assembly\n(from PA-63 depletion)', zorder=3)
    
    # Plot the fit curve
    plt.plot(pH_smooth, fit_curve_norm, 'r-', linewidth=2.5, 
             label=f'Hill Fit\n$pK_a$ = {pKa_fit:.2f} ± {pKa_err:.2f}\n$n$ = {n_fit:.2f} ± {n_err:.2f}')
    
    # Guides
    plt.axvline(pKa_fit, color='gray', linestyle='--', alpha=0.6)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.6)
    
    plt.title(f"pH-Dependent PA Pore Assembly (0.6% DOC)\n{title_suffix}")
    plt.xlabel('pH')
    plt.ylabel('Fractional Pore Assembly')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.show()
    
    # 7. Output the results
    print("-" * 40)
    print(f"FIT RESULTS: {title_suffix}")
    print(f"Midpoint (pKa):     {pKa_fit:.2f} ± {pKa_err:.2f}")
    print(f"Hill Coeff (n):     {n_fit:.2f} ± {n_err:.2f}")
    print("-" * 40)

# --- Main Block ---
if __name__ == "__main__":
    # REPLACE these arrays with scanned PA-63 band volumes
    
    # Example pH range 5.0 to 9.0
    pH_gradient = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
    
    # Example PA-63 Densitometry: Monomer is absent at low pH (pore formed), 
    # and abundant at high pH (prepore trapped).
    # These dummy numbers reflect a shift where pore forms below pH ~7.8
    pa63_band_volume = np.array([500, 800, 1200, 3000, 8000, 15000, 38000, 44000, 45000])

    fit_monomer_densitometry(pH_gradient, pa63_band_volume, title_suffix="SDS-Resistance PA-63 Monomer Analysis")
