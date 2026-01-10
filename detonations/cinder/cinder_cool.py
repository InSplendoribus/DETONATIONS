# detonations/cinder/cinder_cool.py
# -*- coding: utf-8 -*-
"""
Dust-induced cooling helpers (graphite only), faithful to the standalone/Fortran:
  Λ_dust(T) = F1 * Σ_bins [ D_bin * ( T^(3/2) * ( ∫ he dYe + F2 * hn ) * (1/m_grain) ) ]

Interface:
  - he_graphite, hn_graphite
  - per_bin_kernel_graphite(T, a_cm, inv_m)  -> (NBINS, *T.shape)
  - graphite_cooling_bins(D_bins, T, bins)   -> total Λ_dust(T) using current D_bins
  - dust_cooling_n2(n, T, D_bins, bins, D_min_clip=0.0) -> n^2 * Λ_dust
  - dust_lambda_per_D(T, bins=None)          -> Λ_dust per unit 𝓓 (for plotting)
"""
import numpy as np

# Use representative sizes from cinder_massfrac
from . import cinder_massfrac as cm

# ---------------- constants ----------------
EV2ERG  = 1.60217657e-12
KEV2ERG = 1.60217657e-9
ERG2EV  = 6.24150913e11

KB  = 1.380648e-16
ME  = 9.10938291e-28
mH  = 1.6726219e-24
PI  = np.pi

# Dwek (1987)
_K1 = 0.1466
_K2 = 0.502
_K3 = -8.150
_ZETA_MAX = 0.875

# Graphite bulk density (g cm^-3)
RHO_G = 2.26

# Prefactors as per Dwek
F1 = (1.4 * mH) * np.sqrt(32.0 / (PI * ME)) * PI * (KB ** 1.5)     # overall
F2 = (14.0 / 23.0) * np.sqrt(ME / mH)                               # nuclei

# Numericals
_EPS_POS = 1e-300
_EPS_RP  = 1e-300

# Ye integration setup (0..20)
_YE_MIN = 0.0
_YE_MAX = 20.0
_YE_N   = 400  # even number preferred for Simpson

# --- cheap skip thresholds ---
_T_SKIP = 10.0**5.5     # K
_D_SKIP = 1.0e-4        # total dust-to-gas across bins

# ---------------- small helpers ----------------
def _safe_log10(x):
    return np.log10(np.maximum(x, _EPS_POS))

def _safe_disc(x):
    return np.maximum(x, 0.0)

# Safer 10**x via exp(x*ln10) with clamped exponent
def _pow10(x):
    return np.exp(np.clip(x * np.log(10.0), -700.0, 700.0))

# Safer x**y via exp(y*ln x) with clamped exponent
def _pow_xy(x, y):
    x = np.maximum(x, _EPS_POS)
    return np.exp(np.clip(y * np.log(x), -700.0, 700.0))

# Safer a/b via exp(log a - log b) with clamping (prevents overflow warnings)
def _safe_ratio(a, b):
    a = np.maximum(a, _EPS_POS)
    b = np.maximum(b, _EPS_POS)
    return np.exp(np.clip(np.log(a) - np.log(b), -700.0, 700.0))

# ---------------- single-size efficiencies ----------------
def he_graphite(Ye, a_cm, T):
    """Electron heating efficiency for graphite grains (Dwek 1987 Eq. 2)."""
    Rs = (4.0/3.0) * a_cm * RHO_G
    E  = Ye * KB * np.maximum(T, _EPS_POS) * ERG2EV
    E  = np.maximum(E, _EPS_POS)

    inv_2K1 = 0.5 / _K1
    tenK3   = _pow10(_K3)

    # Ee
    disc_Ee = _K2**2 - 4.0*_K1*(_K3 - _safe_log10(Rs))
    Ee = _pow10(( -_K2 + np.sqrt(_safe_disc(disc_Ee)) ) * inv_2K1)

    # Re and Rp
    # Original: Re = tenK3 * (E ** (_K1 * _safe_log10(E) + _K2))
    y  = (_K1 * _safe_log10(E) + _K2)
    Re = tenK3 * _pow_xy(E, y)
    Rp = np.maximum(Re - Rs, _EPS_RP)

    # Ep
    disc_Ep = _K2**2 - 4.0*_K1*(_K3 - _safe_log10(Rp))
    Ep = _pow10(( -_K2 + np.sqrt(_safe_disc(disc_Ep)) ) * inv_2K1)

    Ef = np.maximum(Ep, 0.125 * Ee)

    # Use safe ratio to avoid "overflow encountered in divide"
    ratio = _safe_ratio(Ef, E)
    zeta = np.where(E <= Ee, _ZETA_MAX, 1.0 - ratio)
    # Numerical clipping to [0,1] (prevents tiny negatives from round-off)
    zeta = np.clip(zeta, 0.0, 1.0)

    return 0.5 * Ye**2 * np.exp(-np.minimum(Ye, 700.0)) * (a_cm**2.0) * zeta

def hn_graphite(a_cm, T):
    """Atomic-nuclei heating efficiency for graphite, Dwek (1987) Eq. 8."""
    XH  = (133.0 * (a_cm*1e4) * KEV2ERG) / (KB * np.maximum(T, _EPS_POS))
    XHe = (222.0 * (a_cm*1e4) * KEV2ERG) / (KB * np.maximum(T, _EPS_POS))
    termH  = 1.0 - (1.0 + 0.5*XH ) * np.exp(-np.minimum(XH, 700.0))
    termHe = 1.0 - (1.0 + 0.5*XHe) * np.exp(-np.minimum(XHe, 700.0))
    return (termH + 0.5*termHe) * (a_cm**2.0)

# ---------------- Ye integral of he (vectorized) ----------------
def _integrated_he(a_cm, T):
    """∫_0^20 he_graphite(Ye, a, T) dYe  (shape follows T)."""
    Ye = np.linspace(_YE_MIN, _YE_MAX, _YE_N + 1)  # edges for trapezoid
    Ye2D = Ye[(slice(None),) + (None,) * np.ndim(T)]
    he = he_graphite(Ye2D, a_cm, T)  # (NYe, *T.shape)
    dYe = (Ye[-1] - Ye[0]) / _YE_N
    return dYe * (0.5*he[0] + he[1:-1].sum(axis=0) + 0.5*he[-1])

# ---------------- per-bin kernel (NBINS, *T.shape) ----------------
def per_bin_kernel_graphite(T, a_cm, inv_m, Ye=1.0):
    """
    K_i(T) = F1 * T^(3/2) * [ ∫ he dYe + F2 * hn ] * (1/m_grain,i)
    Shapes:
      T: array or scalar
      a_cm: (NBINS,)
      inv_m: (NBINS,)
    Returns:
      K: (NBINS, *T.shape)
    """
    T = np.asarray(T)
    a_cm = np.asarray(a_cm)
    inv_m = np.asarray(inv_m)

    T32 = np.maximum(T, _EPS_POS) ** 1.5

    K_list = []
    for ai, imi in zip(a_cm, inv_m):
        he_int = _integrated_he(ai, T)
        hn_val = hn_graphite(ai, T)
        kernel = F1 * T32 * (he_int + F2 * hn_val) * imi
        K_list.append(kernel)

    return np.stack(K_list, axis=0)


# ---------------- convenience: total Λ_dust for a given cell/bin set ----------------
def graphite_cooling_bins(D_bins, T, bins):
    """
    Sum over bins for the current cell:
      D_bins: (..., NBINS) dust-to-gas per bin in the cell
      T: same leading shape as the grid cell (can be scalar)
      bins: object with bin definition (only used to know NBINS)
    Returns:
      Λ_dust(T) with shape of T
    """
    # Get representative radii and 1/masses directly from cinder_massfrac
    a, inv_m = cm.representative_sizes()           # (NBINS,), (NBINS,)

    # Build per-bin kernel and weight by current D_bins
    K = per_bin_kernel_graphite(T, a, inv_m)       # (NBINS, *T.shape)

    # Move bins axis of D_bins to front to match K and sum over bins
    D = np.moveaxis(np.asarray(D_bins), -1, 0)     # (NBINS, *T.shape)
    return np.sum(D * K, axis=0)                   # (*T.shape)


# ---------------- wrappers used by cooling.py / plotting helpers -------------
def dust_cooling_n2(n, T, D_bins, bins, D_min_clip=0.0):
    """
    Return volumetric dust cooling q_dust = n^2 * Λ_dust(T) using the current
    per-cell per-bin dust-to-gas ratios D_bins.

    Fast path: if T < 10^5.5 K OR total 𝓓 < 1e-4, return 0 without
    evaluating the (expensive) kernels.
    """
    T_arr = np.asarray(T)
    n_arr = np.asarray(n)

    # total dust-to-gas ratio across bins
    Dsum = np.sum(np.maximum(D_bins, 0.0), axis=-1)

    # Build mask of cells where dust cooling matters, so to speed up when applied
    mask = (T_arr >= _T_SKIP) & (Dsum >= _D_SKIP)

    # Nothing qualifies → zero cooling quickly
    if not np.any(mask):
        return np.zeros_like(T_arr, dtype=float)

    # Compute only on the qualifying subset
    if D_min_clip is not None:
        D_bins = np.maximum(D_bins, D_min_clip)

    q = np.zeros_like(T_arr, dtype=float)

    # Scalar case
    if T_arr.ndim == 0:
        Lam = graphite_cooling_bins(D_bins, T_arr, bins)
        q = (n_arr * n_arr) * Lam
        return q

    # Array case: slice the active cells, compute, and scatter back
    Lam_sub = graphite_cooling_bins(D_bins[mask], T_arr[mask], bins)
    q[mask] = (n_arr[mask] * n_arr[mask]) * Lam_sub
    return q


def dust_lambda_per_D(T, bins=None):
    """
    Λ_dust(T) for a *unit* total dust-to-gas ratio (∑_bins D_i = 1).
    Useful for plotting/diagnostics; not used in the main loop.
    """
    edges_um, mass_fracs = cm.make_bins()          # honors settings.py
    a_cm, inv_m = cm.representative_sizes(edges_um)
    K = per_bin_kernel_graphite(T, a_cm, inv_m)    # (NBINS, *T.shape)
    mass_fracs = np.asarray(mass_fracs)
    # Sum over bins only
    return np.sum(mass_fracs[:, None] * K, axis=0)
