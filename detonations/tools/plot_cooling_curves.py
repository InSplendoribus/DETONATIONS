# tools/plot_cooling.py
import sys
import numpy as np
import matplotlib.pyplot as plt

# Use your new helpers
from detonations.cinder.cinder_cool import graphite_cooling_curve
from detonations.cinder import cinder_massfrac as cm

# --- 1) Load gas (Schure) cooling table and build an interpolator ---
def load_schure_lambda(table_path):
    # expected 2 columns: log10(T [K])  log10(Lambda [erg cm^3 s^-1])
    dat = np.loadtxt(table_path, comments="#")
    logT_tab, logLam_tab = dat[:,0], dat[:,1]
    def schure_lambda(T):
        logT = np.log10(np.clip(T, 1e-2, None))
        logLam = np.interp(logT, logT_tab, logLam_tab, left=logLam_tab[0], right=logLam_tab[-1])
        return 10.0**logLam
    return schure_lambda

def main(parfile="detonations.par"):
    # Read the parfile to find the Schure table path
    from detonations.read_data import load_config, write_settings_module
    cfg = load_config(parfile)
    # Make sure detonations/settings.py exists (same as in your main.py)
    from pathlib import Path
    settings_py = Path(__file__).resolve().parents[1] / "detonations" / "settings.py"
    write_settings_module(cfg, settings_py)
    # Now import settings
    from detonations import settings as S

    schure = load_schure_lambda(S.LAMBDA_TABLE_FILE)

    # --- 2) Build dust bins (lognormal as default) ---
    # You can switch to 'mrn' and tweak parameters if you like.
    edges_um, mass_fracs = cm.make_bins(
        kind      = getattr(S, "DUST_BINS", "lognormal"),   # 'lognormal' or 'mrn'
        n_bins    = int(getattr(S, "DUST_NBINS", 10)),
        a_min_um  = float(getattr(S, "DUST_A_MIN_UM", 0.005)),
        a_max_um  = float(getattr(S, "DUST_A_MAX_UM", 0.5)),
        a0_um     = float(getattr(S, "LOGNORM_A0_UM", 0.1)),
        sigma_ln  = float(getattr(S, "LOGNORM_SIGMA", 0.5)),
        q_mrn     = float(getattr(S, "MRN_Q", -3.5)),
    )
    # Representative sizes (geometric means) in cm
    a_m_cm = np.sqrt(edges_um[:-1]*edges_um[1:]) * 1e-4

    # 1/m_grain for graphite (ρ = 2.26 g/cm^3)
    rho_g = 2.26
    inv_m = 1.0 / ((4.0/3.0) * np.pi * rho_g * (a_m_cm**3))

    # Normalize mass fractions (safety)
    w = mass_fracs / max(mass_fracs.sum(), 1e-300)

    # Choose a global dust-to-gas ratio for the plot (e.g. ISM-like 1e-2)
    D_tot = 1.0e-2

    # --- 3) Build temperature array and evaluate both curves ---
    T = np.logspace(1, 9, 500)  # 10 K ... 1e9 K
    lam_gas  = schure(T)
    lam_dust = graphite_cooling_curve(T, a_m_cm, w, inv_m, Ye=1.0, D_total=D_tot)

    # --- 4) Plot ---
    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(T, lam_gas,  label="Schure (gas) cooling")
    plt.loglog(T, lam_dust, "--", label="Dust cooling (graphite, lognormal)")
    plt.xlabel("Temperature [K]")
    plt.ylabel(r"$\Lambda\ \mathrm{[erg\ cm^{3}\ s^{-1}]}$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "detonations.par")
