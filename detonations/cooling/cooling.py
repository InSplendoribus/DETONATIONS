# -*- coding: utf-8 -*-
"""
cooling.py — Schure+09 table cooling + optional dust cooling.

This version implements:
  (2) Temperature-dependent mean molecular weight μ(T) for n and T.
  (3) Stable explicit cooling via subcycling with a temperature-based energy floor.

And applies Schure et al. (2009) electron-to-hydrogen ratio:
  q_gas = n_e * n_H * Λ(T),
  with n_e / n_H taken from the provided Schure table snippet.

Notes:
- If S.LAMBDA_TABLE_FILE is missing, Λ(T) returns zeros (cooling off).
- Hydrogen mass fraction X_H is taken from S.X_H if present, else 0.71.
- Dust cooling is included if S.DUST_GAS_COOLING_ON is True and cinder is available.
"""

from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import interp1d

from ..hydro.solvers import prim_from_cons
from .. import settings as S

# Optional dust cooling helper
try:
    from ..cinder import cinder_cool
except Exception:
    cinder_cool = None


# ------------------ Schure+ table loader ------------------------------------
def load_schure_table(path=None):
    """
    Load Λ(T) from table (assumed erg cm^3 s^-1 vs. log10 T).
    Table format expected: col0 = log10(T[K]), col1 = log10(Λ).
    """
    path = path or getattr(S, 'LAMBDA_TABLE_FILE', 'Schure_total.dat')
    if not os.path.exists(path):
        # Safe fallback: no gas cooling if file missing.
        return lambda T: np.zeros_like(T)
    arr = np.loadtxt(path)
    T_vals, Lam_vals = 10.0**arr[:, 0], 10.0**arr[:, 1]
    f = interp1d(np.log10(T_vals), np.log10(Lam_vals),
                 kind="linear", fill_value="extrapolate", bounds_error=False)

    def Lambda(Tin):
        Tclip = np.clip(Tin, 1e2, 1e8)   # validity band
        return 10.0**f(np.log10(Tclip))  # erg cm^3 s^-1

    return Lambda


Lambda_T = load_schure_table()


# ------------------ Thermo helpers (μ(T) and n_e/n_H) -----------------------
def mu_piecewise(T):
    '''
    Temperature-dependent mean molecular weight μ(T) in grams, piecewise:
      - 2.33 m_H   for T < 5e3 K   (molecular gas)
      - 1.27 m_H   for 5e3 <= T < 1e4 K (neutral atomic)
      - 0.61 m_H   for T >= 1e4 K  (fully ionized, solar comp.)
    '''
    T = np.asarray(T)
    mu_low  = 2.33 * S.mH
    mu_mid  = 1.27 * S.mH
    mu_high = 0.61 * S.mH
    return np.where(
        T < 5.0e3, mu_low,
        np.where(T < 1.0e4, mu_mid, mu_high)
    )


def ne_over_nH_schure(T):
    """
    Interpolate (n_e / n_H)(T) from Schure+2009 (solar metallicity) using
    the provided log10(T)–ratio table snippet.
    """
    _logT = np.array([
        3.80,3.84,3.88,3.92,3.96,4.00,4.04,4.08,4.12,4.16,4.20,4.24,4.28,4.32,4.36,4.40,4.44,4.48,4.52,4.56,4.60,4.64,4.68,4.72,4.76,4.80,4.84,4.88,4.92,4.96,5.00,5.04,5.08,5.12,5.16,5.20,5.24,5.28,5.32,5.36,5.40,5.44,5.48,5.52,5.56,5.60,5.64,5.68,5.72,5.76,5.80,5.84,5.88,5.92,5.96,
        6.00,6.04,6.08,6.12,6.16,6.20,6.24,6.28,6.32,6.36,6.40,6.44,6.48,6.52,6.56,6.60,6.64,6.68,6.72,6.76,6.80,6.84,6.88,6.92,6.96,7.00,7.04,7.08,7.12,7.16,7.20,7.24,7.28,7.32,7.36,7.40,7.44,7.48,7.52,7.56,7.60,7.64,7.68,7.72,7.76,7.80,7.84,7.88,7.92,7.96,8.00,8.04,8.08,8.12,8.16
    ])
    _ratio = np.array([
        1.3264e-05,4.2428e-05,8.8276e-05,1.7967e-04,8.4362e-04,3.4295e-03,1.3283e-02,4.2008e-02,1.2138e-01,3.0481e-01,5.3386e-01,7.6622e-01,8.9459e-01,9.5414e-01,9.8342e-01,1.0046,1.0291,1.0547,1.0767,1.0888,1.0945,1.0972,1.0988,1.1004,1.1034,1.1102,1.1233,1.1433,1.1638,1.1791,1.1885,1.1937,1.1966,1.1983,1.1993,1.1999,1.2004,1.2008,1.2012,1.2015,1.2020,1.2025,1.2030,1.2035,1.2037,1.2039,1.2040,1.2041,1.2042,1.2044,1.2045,1.2046,1.2047,1.2049,1.2050,
        1.2051,1.2053,1.2055,1.2056,1.2058,1.2060,1.2062,1.2065,1.2067,1.2070,1.2072,1.2075,1.2077,1.2078,1.2079,1.2080,1.2081,1.2082,1.2083,1.2083,1.2084,1.2084,1.2085,1.2085,1.2086,1.2086,1.2087,1.2087,1.2088,1.2088,1.2089,1.2089,1.2089,1.2089,1.2089,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2091,1.2091,1.2091
    ])
    f = interp1d(_logT, _ratio, kind='linear',
                 fill_value='extrapolate', bounds_error=False)
    T = np.asarray(T)
    logT = np.log10(np.clip(T, 1e2, 1e8))
    return f(logT)


# ------------------ Apply cooling -------------------------------------------
def apply_cooling(U, dt, bins, D_bins):
    if not getattr(S, 'COOLING_ON', True):
        return U

    rho, vx, vy, vz, P = prim_from_cons(U)
    # internal (thermal) energy density
    e_th = P / (S.gamma - 1.0)

    # Initial estimate with constant mu; then 3 fixed-point iterations for μ(T)
    n = rho / np.maximum(S.mu_mol, 1e-30)
    T = np.maximum(P / (n * S.kB), 10.0)
    for _ in range(3):
        mu_eff = mu_piecewise(T)
        n = rho / np.maximum(mu_eff, 1e-30)
        T = np.maximum(P / (n * S.kB), 10.0)

    # Hydrogen mass fraction (fallback to 0.71 if not provided)
    X_H = getattr(S, "X_H", 0.71)
    # Hydrogen number density (cm^-3) from mass density
    nH = (X_H * rho) / np.maximum(S.mH, 1e-30)
    # Electron-to-hydrogen ratio from Schure table
    ne_over_nH = ne_over_nH_schure(T)
    ne = ne_over_nH * nH

    # Gas cooling (Schure): q_gas = n_e n_H Λ(T)  — erg cm^-3 s^-1
    q_gas = ne * nH * Lambda_T(T)
    q = q_gas

    # Optional dust cooling (adds to q)
    if getattr(S, 'DUST_GAS_COOLING_ON', False) and (cinder_cool is not None):
        if (D_bins is not None) and (bins is not None):
            try:
                # cinder returns n^2 * Λ_dust; we pass n (total number density)
                q_dust = cinder_cool.dust_cooling_n2(n, T, D_bins, bins)
                q = q + q_dust
            except Exception:
                # Minimal patch policy: ignore dust if shapes mismatch
                pass

    # Subcycling based on cooling time with safety factor
    T_floor = getattr(S, 'T_FLOOR', 10.0)
    mu_floor = mu_piecewise(T_floor)
    n_floor = rho / np.maximum(mu_floor, 1e-30)
    e_floor = (n_floor * S.kB * T_floor) / (S.gamma - 1.0)

    t_cool = np.maximum(e_th / np.maximum(q, 1e-30), 1e-30)
    try:
        t_cool_min = np.min(t_cool)
    except Exception:
        t_cool_min = t_cool
    safety = 0.2
    nsub = int(np.ceil(dt / max(safety * t_cool_min, 1e-30)))
    nsub = max(1, min(nsub, 10000))  # cap to avoid pathological loops

    dt_sub = dt / nsub
    for _ in range(nsub):
        # Update state-dependent quantities at current e_th
        P_curr = (S.gamma - 1.0) * e_th

        # Recompute T via μ(T) self-consistently (one iteration is typically enough per substep)
        mu_eff = mu_piecewise(T)
        n = rho / np.maximum(mu_eff, 1e-30)
        T = np.maximum(P_curr / (n * S.kB), T_floor)

        # Update species for gas cooling
        nH = (X_H * rho) / np.maximum(S.mH, 1e-30)
        ne_over_nH = ne_over_nH_schure(T)
        ne = ne_over_nH * nH

        # Recompute q
        q = ne * nH * Lambda_T(T)

        # Add dust again if enabled
        if getattr(S, 'DUST_GAS_COOLING_ON', False) and (cinder_cool is not None):
            if (D_bins is not None) and (bins is not None):
                try:
                    q += cinder_cool.dust_cooling_n2(n, T, D_bins, bins)
                except Exception:
                    pass

        # Advance internal energy with floor
        e_th = np.maximum(e_th - q * dt_sub, e_floor)

    # Reassemble total energy (keep momenta unchanged)
    ek = 0.5 * U[0] * (
        (U[1] / np.maximum(U[0], 1e-30))**2 +
        (U[2] / np.maximum(U[0], 1e-30))**2 +
        (U[3] / np.maximum(U[0], 1e-30))**2
    )
    U[4] = e_th + ek
    return U


# ------------------ Testing helper (optional) --------------------------------
def write_cooling_comparison_png(out_png, bins=None):
    """
    Writes a simple PNG comparing Λ_gas (Schure) and Λ_dust_per_D=1 curves.
    Not called automatically; invoke manually for tests.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = np.logspace(2, 8, 400)
    L_gas = Lambda_T(T)

    if cinder_cool is not None:
        if bins is None:
            bins = cinder_cool.build_bins_from_settings()
        L_dust_per_D = cinder_cool.dust_lambda_per_D(T, bins=bins)
    else:
        L_dust_per_D = np.zeros_like(T)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(T, L_gas, label="Λ_gas (Schure)")
    ax.loglog(T, L_dust_per_D, label="Λ_dust per unit 𝓓")
    ax.set_xlabel("T [K]"); ax.set_ylabel("Λ [erg cm$^3$ s$^{-1}$]")
    ax.grid(True, which='both', ls=':')
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
