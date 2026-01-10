# detonations/cinder/cinder_massfrac.py
# Bin edges + mass fractions (MRN or lognormal) driven by settings.py,
# plus representative grain sizes per bin.

import numpy as np
from ..settings import (
    DUST_BINS, DUST_NBINS, DUST_A_MIN_UM, DUST_A_MAX_UM,
    LOGNORM_A0_UM, LOGNORM_SIGMA, MRN_Q,
)

def _edges_um(nbins, a_min_um, a_max_um):
    return np.logspace(np.log10(a_min_um), np.log10(a_max_um), nbins + 1)

def _lognormal_mass_fracs(edges_um, a0_um, sigma_ln):
    nbins = len(edges_um) - 1
    m = np.zeros(nbins)
    for i in range(nbins):
        a0, a1 = edges_um[i], edges_um[i+1]
        a = np.linspace(a0, a1, 256)
        g = np.exp(-0.5 * (np.log(a / max(a0_um, 1e-99)) / max(sigma_ln, 1e-12))**2)
        # mass ∝ ∫ a^3 * (1/a) g(a) da = ∫ a^2 g(a) da
        w = a**2 * g
        m[i] = np.trapz(w, a)
    m_sum = np.maximum(m.sum(), 1e-30)
    return m / m_sum

def _mrn_mass_fracs(edges_um, q):
    s = 3.0 + q
    nbins = len(edges_um) - 1
    m = np.zeros(nbins)
    if abs(s + 1.0) < 1e-12:
        # fallback numerical
        for i in range(nbins):
            a0, a1 = edges_um[i], edges_um[i+1]
            a = np.linspace(a0, a1, 256)
            m[i] = np.trapz(a**(3.0 + q), a)
    else:
        for i in range(nbins):
            a0, a1 = edges_um[i], edges_um[i+1]
            m[i] = (a1**(s+1) - a0**(s+1)) / (s+1)
    m_sum = np.maximum(m.sum(), 1e-30)
    return m / m_sum

def make_bins(kind: str | None = None):
    """
    Returns (edges_um, mass_fracs) based on settings.
    edges_um has length NBINS+1; mass_fracs has length NBINS and sums to 1.
    """
    nb = int(DUST_NBINS)
    edges = _edges_um(nb, float(DUST_A_MIN_UM), float(DUST_A_MAX_UM))
    k = (kind or DUST_BINS or "lognormal").lower()
    if k == "mrn":
        fr = _mrn_mass_fracs(edges, float(MRN_Q))
    else:
        fr = _lognormal_mass_fracs(edges, float(LOGNORM_A0_UM), float(LOGNORM_SIGMA))
    return edges, fr

def representative_sizes(edges_um: np.ndarray | None = None, rho_gr: float = 2.26):
    """
    Representative grain size per bin (geometric mean) and 1/m_grain.
      a_m  : cm, shape (NBINS,)
      inv_m: 1/g, shape (NBINS,)
    If edges_um is None, they are built from settings via make_bins().
    """
    if edges_um is None:
        edges_um, _ = make_bins()
    a_min = edges_um[:-1] * 1e-4  # cm
    a_max = edges_um[1:]  * 1e-4  # cm
    a_m   = np.sqrt(a_min * a_max)
    m_gr  = (4.0/3.0) * np.pi * rho_gr * (a_m**3)
    inv_m = 1.0 / np.maximum(m_gr, 1e-99)
    return a_m, inv_m
