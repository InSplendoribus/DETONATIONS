# -*- coding: utf-8 -*-
"""
cinder.py — Dust module (bins, growth, sputtering, dust–gas cooling)

Minimal changes:
  * Restores setup_lognormal_bins() (no hardcoded tables).
  * Bin edges/fractions come from settings.py (parfile-controlled).
  * Growth: Eq. (3) in MG+22, using current-bin fraction f_i^curr.
  * Sputtering: Tsai & Mathews (1995) form.
"""

import numpy as np
from ..settings import (
    # bin controls (from par file via read_data.write_settings_module)
    DUST_BINS, DUST_NBINS, DUST_A_MIN_UM, DUST_A_MAX_UM,
    LOGNORM_A0_UM, LOGNORM_SIGMA, MRN_Q,
)

# ---------- Constants and thresholds ----------
DUST_GAS_COOLING_ON = False

N_H_THR          = 1.0e1    # cm^-3 (n_H) for growth_mask (if used).
T_GROW_MAX       = 1.0e3    # K.
DIV_EPS          = 0.0      # s^-1.
T_SP_MIN         = 5.0e5    # K (activates sputtering).
D_CAP_SOLAR      = 1.0e-2   # D_max = D_CAP_SOLAR * (Z/Z_sun).
Z_OVER_ZSUN      = 1.0

rho_s = 3.0                 # g cm^-3 (solid density).
s_eff = 0.2222              # Sticking coefficient.

mH = 1.6726219e-24          # g.

# ---------- Bins container ----------
class Bins:
    def __init__(self, a_edges_um, mass_fracs):
        self.NBINS = len(mass_fracs)
        self.a_min = a_edges_um[:-1] * 1e-4  # cm
        self.a_max = a_edges_um[1:]  * 1e-4  # cm
        self.a_m   = np.sqrt(self.a_min * self.a_max)  # cm (geometric mean)
        self.mass_fracs = mass_fracs / np.maximum(mass_fracs.sum(), 1e-30)

def print_bins_table(bins: Bins):
    print("[Dust bins]  (sizes in μm)")
    print(" i   a_min   a_max   a_m   mass_frac")
    for i in range(bins.NBINS):
        print(f" {i:>1}   {bins.a_min[i]*1e4:0.4f}   {bins.a_max[i]*1e4:0.4f}   {bins.a_m[i]*1e4:0.4f}   {bins.mass_fracs[i]:0.5f}")
    print(" sum mass_frac = 1")
    print("---------------------")

# ---------- Bin builders ----------
def _edges_um(nbins, a_min_um, a_max_um):
    return np.logspace(np.log10(a_min_um), np.log10(a_max_um), nbins + 1)

def _lognormal_mass_fracs(edges_um, a0_um, sigma_ln):
    # mass ∝ ∫ a^3 * (1/a) exp(-0.5(ln(a/a0)/sigma)^2) da = ∫ a^2 * exp(...) da
    # integrate numerically in linear-a within each bin
    nbins = len(edges_um) - 1
    m = np.zeros(nbins)
    for i in range(nbins):
        a0, a1 = edges_um[i], edges_um[i+1]
        a = np.linspace(a0, a1, 256)
        ln_term = np.log(a / max(a0_um, 1e-99)) / max(sigma_ln, 1e-12)
        w = a**2 * np.exp(-0.5 * ln_term**2)
        m[i] = np.trapz(w, a)
    m_sum = np.maximum(m.sum(), 1e-30)
    return m / m_sum

def _mrn_mass_fracs(edges_um, q):
    # MRN: n(a) ∝ a^q; mass ∝ ∫ a^3 n(a) da ∝ ∫ a^(3+q) da
    s = 3.0 + q
    if abs(s + 1.0) < 1e-12:
        # problematic q = -4 -> log; fallback to numerical
        return _mrn_mass_fracs_numeric(edges_um, q)
    nbins = len(edges_um) - 1
    m = np.zeros(nbins)
    for i in range(nbins):
        a0, a1 = edges_um[i], edges_um[i+1]
        m[i] = (a1**(s+1) - a0**(s+1)) / (s+1)
    m_sum = np.maximum(m.sum(), 1e-30)
    return m / m_sum

def _mrn_mass_fracs_numeric(edges_um, q):
    nbins = len(edges_um) - 1
    m = np.zeros(nbins)
    for i in range(nbins):
        a0, a1 = edges_um[i], edges_um[i+1]
        a = np.linspace(a0, a1, 256)
        m[i] = np.trapz(a**(3.0 + q), a)
    m_sum = np.maximum(m.sum(), 1e-30)
    return m / m_sum

def setup_bins():
    """General bin setup honoring settings.DUST_BINS."""
    nb = int(DUST_NBINS)
    edges = _edges_um(nb, float(DUST_A_MIN_UM), float(DUST_A_MAX_UM))
    kind = (DUST_BINS or "lognormal").lower()
    if kind == "mrn":
        fr = _mrn_mass_fracs(edges, float(MRN_Q))
    else:
        fr = _lognormal_mass_fracs(edges, float(LOGNORM_A0_UM), float(LOGNORM_SIGMA))
    return Bins(edges, fr)

def setup_lognormal_bins():
    """Compat shim used by clouds.init_state(); uses lognormal params from settings."""
    nb = int(DUST_NBINS)
    edges = _edges_um(nb, float(DUST_A_MIN_UM), float(DUST_A_MAX_UM))
    fr = _lognormal_mass_fracs(edges, float(LOGNORM_A0_UM), float(LOGNORM_SIGMA))
    return Bins(edges, fr)

# ---------- Dust–gas cooling (optional placeholder) ----------
def dust_gas_cooling_rate(n, T, Dtot):
    return np.zeros_like(n)

# ---------- Growth mask (if used externally) ----------
def growth_mask(U, mu_mol, kB):
    rho = U[0]
    vx  = U[1]/np.maximum(rho,1e-30)
    vy  = U[2]/np.maximum(rho,1e-30)
    vz  = U[3]/np.maximum(rho,1e-30)
    ek  = 0.5*rho*(vx*vx+vy*vy+vz*vz)
    gamma = 5.0/3.0
    P    = (gamma-1.0)*np.maximum(U[4]-ek, 0.0)
    n    = rho/np.maximum(mu_mol,1e-30)
    T    = np.maximum(P/(np.maximum(n,1e-30)*kB), 10.0)  

    nH   = n
    mask = (nH >= N_H_THR) & (T <= T_GROW_MAX)
    if DIV_EPS > 0.0:
        dvx_dx = np.zeros_like(vx); dvy_dy = np.zeros_like(vy); dvz_dz = np.zeros_like(vz)
        dvx_dx[1:-1,:,:] = 0.5*(vx[2:,:,:] - vx[:-2,:,:])
        dvy_dy[:,1:-1,:] = 0.5*(vy[:,2:,:] - vy[:,:-2,:])
        dvz_dz[:,:,1:-1] = 0.5*(vz[:,:,2:] - vz[:,:,:-2])
        divv = dvx_dx + dvy_dy + dvz_dz
        mask = mask & (divv < -DIV_EPS)
    return mask

# ---------- Growth + Sputtering ----------
def apply_growth_and_sputtering(D_bins, U, dt, bins: Bins, mu_mol, kB, dx, dy, dz):
    rho = U[0]
    vx  = U[1]/np.maximum(rho,1e-30)
    vy  = U[2]/np.maximum(rho,1e-30)
    vz  = U[3]/np.maximum(rho,1e-30)
    ek  = 0.5*rho*(vx*vx+vy*vy+vz*vz)
    gamma = 5.0/3.0
    P    = (gamma-1.0)*np.maximum(U[4]-ek, 0.0)
    n    = rho/np.maximum(mu_mol,1e-30)                           # cm^-3.
    T    = np.maximum(P/(np.maximum(n,1e-30)*kB), 10.0)

    NB = bins.NBINS
    a_m = bins.a_m.copy()          # cm.
    D   = np.clip(D_bins, 0.0, None)

    m_ref = 12.0*mH
    v_th  = np.sqrt(8.0*kB*np.clip(T,1.0,None)/(np.pi*m_ref))
    rho_r = rho
    pref  = s_eff * (rho_r/(4.0*rho_s)) * v_th

    Dtot = np.sum(D, axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac_curr = np.where(Dtot>0.0, D/np.maximum(Dtot,1e-30), bins.mass_fracs)

    adot  = pref[...,None] * frac_curr
    grow_fac = 1.0 + np.clip((3.0*adot*dt)/np.maximum(a_m,1e-30), -0.9, 5.0)
    Dg = D * grow_fac

    # Sputtering
    mask_sp = (T >= T_SP_MIN)
    if np.any(mask_sp):
        h = 3.2e-18
        adot_sp = -1.4 * n * h / ( (1.0e6/T)**2.5 + 1.0 )
        shrink = 1.0 + (3.0*adot_sp[...,None]*dt)/a_m
        Ds = np.where(mask_sp[...,None], np.maximum(Dg*shrink, 0.0), Dg)
    else:
        Ds = Dg

    D_new = np.clip(Ds, 0.0, None)

    # Cap to D_cap for safety
    D_cap = D_CAP_SOLAR * Z_OVER_ZSUN
    Dtot_new = np.sum(D_new, axis=-1, keepdims=True)
    over = Dtot_new > (D_cap + 1e-12)
    if np.any(over):
        scale = (D_cap / np.maximum(Dtot_new,1e-30))
        D_new = np.where(over, D_new*scale, D_new)

    return D_new
