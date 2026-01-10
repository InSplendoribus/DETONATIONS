# detonations/tools/plot_cell_dustcooling.py
# Plot Λ_dust(T) for a chosen total dust ratio and a sampled cloud1 cell.

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from detonations.read_data import load_config, write_settings_module
from detonations.cinder import cinder_massfrac as cm
from detonations.cinder import cinder_cool as cc

def main(par_path: str, D_override: float | None):
    # Load parfile and regenerate detonations/settings.py so bin params are visible.
    cfg = load_config(par_path)
    settings_path = Path(__file__).resolve().parents[1] / "settings.py"
    write_settings_module(cfg, settings_path)

    # Choose total D to use (override > parfile’s D_INIT_CLOUD1 if provided)
    D_plot = float(D_override if D_override is not None else cfg.dust.D_INIT_CLOUD1)

    # Build bins from settings and get representative sizes
    edges_um, mass_fracs = cm.make_bins()
    a_cm, inv_m = cm.representative_sizes(edges_um)

    # Temperature grid
    logT = np.linspace(4.0, 8.0, 400)
    T    = 10.0**logT

    # Total curve at D_plot
    fracs_bins = D_plot * mass_fracs                  # per-bin D_i
    K = cc.per_bin_kernel_graphite(T, a_cm, inv_m, Ye=1.0)   # shape (NBINS, NT)
    Lambda_tot = np.dot(fracs_bins, K)                        # NT

    # Sample “cloud1 cell” at its parfile temperature, scaled to same D_plot
    T_cell  = float(cfg.cloud1.T)
    K_cell  = cc.per_bin_kernel_graphite(T_cell, a_cm, inv_m, Ye=1.0)  # (NBINS,)
    Lam_cell = np.dot(fracs_bins, K_cell)   # scalar

    # Plot
    plt.figure(figsize=(7.6, 5.2))
    plt.plot(logT, np.log10(Lambda_tot), color='k', lw=3,
             label=fr"Full curve, $\mathcal{{D}}={D_plot:.0e}$ (bins)")
    plt.scatter([np.log10(T_cell)], [np.log10(Lam_cell)],
                s=40, color='#1f77b4',
                label=fr"Sampled cloud1 cells (scaled to $\mathcal{{D}}={D_plot:.0e}$)")
    plt.xlabel(r'$\log_{10}\,T\ \mathrm{[K]}$')
    plt.ylabel(r'$\log_{10}\ \Lambda_{\rm dust}\ \mathrm{[erg\,cm^{3}\,s^{-1}]}$')
    plt.grid(True, ls=':', alpha=0.35)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot dust cooling curve and sampled cloud1 point.")
    ap.add_argument("parfile", help="Path to detonations.par")
    ap.add_argument("--D", type=float, default=None,
                    help="Override total dust-to-gas ratio used in the plot (e.g. 1e-2).")
    args = ap.parse_args()
    main(args.parfile, args.D)
