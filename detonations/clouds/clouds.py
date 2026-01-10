from ..cinder import cinder
from ..settings import *
import numpy as np

def coordinates():
    x = (np.arange(Nx)+0.5)*dx + XMIN_PC*PC_CM
    y = (np.arange(Ny)+0.5)*dy + YMIN_PC*PC_CM
    z0 = (np.arange(Nz)+0.5)*dz + ZMIN_PC*PC_CM if DIM3D else np.array([0.0])
    return np.meshgrid(x, y, z0, indexing="ij")

# --------------------------- TURBULENCIA (Kolmogorov) -----------------------


def kolmogorov_velocity_field(X, Y, Z, sigma_kms, seed, nmodes=32):
    """Superposición de modos solenoidales ~ k^{-11/6}; normalizado a sigma_kms."""
    if sigma_kms <= 0.0:
        return np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    rng = np.random.default_rng(seed)
    Lx = (XMAX_PC - XMIN_PC)*PC_CM
    Ly = (YMAX_PC - YMIN_PC)*PC_CM
    Lz = (ZMAX_PC - ZMIN_PC)*PC_CM
    kmin = 2.0*np.pi/min(Lx,Ly,Lz)
    kmax = 2.0*np.pi/max(dx,dy,dz)
    vx = np.zeros_like(X, dtype=np.float64)
    vy = np.zeros_like(X, dtype=np.float64)
    vz = np.zeros_like(X, dtype=np.float64)
    for _ in range(nmodes):
        # k en distribución log-uniforme
        u = rng.random()
        k = np.exp(np.log10(kmin) + u*(np.log10(kmax)-np.log10(kmin)))
        # dirección aleatoria de k
        th = np.arccos(2*rng.random()-1.0); ph = 2*np.pi*rng.random()
        kx, ky, kz = k*np.sin(th)*np.cos(ph), k*np.sin(th)*np.sin(ph), k*np.cos(th)
        # vector amplitud perpendicular a k (solenoidal)
        a_th = 2*np.pi*rng.random()
        ex, ey, ez = np.cos(a_th), np.sin(a_th), 0.0
        # proyecta e en plano perpendicular a k
        kk = np.array([kx,ky,kz]); e = np.array([ex,ey,ez])
        e -= (np.dot(e,kk)/np.maximum(np.dot(kk,kk),1e-30))*kk
        e /= np.sqrt(np.maximum(np.dot(e,e),1e-30))
        # amplitud ~ k^{-11/6}
        A = k**(-11.0/6.0)
        phase = 2*np.pi*rng.random()
        arg = kx*X + ky*Y + kz*Z + phase
        s = np.sin(arg)
        vx += A*e[0]*s; vy += A*e[1]*s; vz += A*e[2]*s
    # normaliza a sigma
    vmag_rms = np.sqrt(np.mean(vx**2 + vy**2 + vz**2))
    target = sigma_kms*1.0e5
    if vmag_rms > 0:
        scale = target/vmag_rms
        vx *= scale; vy *= scale; vz *= scale
    return vx, vy, vz



def add_turbulence_by_region(vx, vy, vz, X, Y, Z, mask, sigma_kms, seed):
    if sigma_kms <= 0.0: 
        return vx, vy, vz
    vx_t, vy_t, vz_t = kolmogorov_velocity_field(X, Y, Z, sigma_kms, seed)
    vx[mask] += vx_t[mask]; vy[mask] += vy_t[mask]; vz[mask] += vz_t[mask]
    return vx, vy, vz

# --------------------------- INITIAL CONDITIONS -----------------------------


def init_state():
    X,Y,Z = coordinates()

    rho = np.full((Nx,Ny,Nz), RHO_AMB)
    vx  = np.zeros_like(rho)
    vy  = np.zeros_like(rho)
    vz  = np.zeros_like(rho)
    P   = np.full((Nx,Ny,Nz), P_AMB)

    # --- Nube 1 ---
    c1 = (C1_CENTER_PC[0]*PC_CM, C1_CENTER_PC[1]*PC_CM, C1_CENTER_PC[2]*PC_CM)
    r1 = C1_RADIUS_PC*PC_CM
    mask1 = ( (X-c1[0])**2 + (Y-c1[1])**2 + (Z-c1[2])**2 ) <= r1*r1
    rho[mask1] = C1_RHO
    P  [mask1] = C1_RHO * kB*C1_T / mu_mol
    vx [mask1] = C1_VX; vy[mask1] = C1_VY; vz[mask1] = C1_VZ

    # --- Nube 2 ---
    c2 = (C2_CENTER_PC[0]*PC_CM, C2_CENTER_PC[1]*PC_CM, C2_CENTER_PC[2]*PC_CM)
    r2 = C2_RADIUS_PC*PC_CM
    mask2 = ( (X-c2[0])**2 + (Y-c2[1])**2 + (Z-c2[2])**2 ) <= r2*r2
    rho[mask2] = C2_RHO
    P  [mask2] = C2_RHO * kB*C2_T / mu_mol
    vx [mask2] = C2_VX; vy[mask2] = C2_VY; vz[mask2] = C2_VZ

    # --- Turbulencia (opcional) ---
    if TURB_ON:
        # máscara ambiente = fuera de nubes
        mask_amb = ~(mask1 | mask2)
        # ambiente
        vx,vy,vz = add_turbulence_by_region(vx,vy,vz, X,Y,Z, mask_amb, SIGMA_T_AMB_KMS, TURB_SEED+1)
        # nube 1
        vx,vy,vz = add_turbulence_by_region(vx,vy,vz, X,Y,Z, mask1, SIGMA_T_C1_KMS, TURB_SEED+2)
        # nube 2
        vx,vy,vz = add_turbulence_by_region(vx,vy,vz, X,Y,Z, mask2, SIGMA_T_C2_KMS, TURB_SEED+3)

    # Trazadores: nube1 y nube2
    fcld1 = np.zeros((Nx,Ny,Nz), dtype=np.float64); fcld1[mask1] = 1.0
    fcld2 = np.zeros((Nx,Ny,Nz), dtype=np.float64); fcld2[mask2] = 1.0

    # Polvo
    if DUST_ON:
        bins = cinder.setup_lognormal_bins()
        cinder.print_bins_table(bins)
        D_bins = np.zeros((Nx,Ny,Nz,bins.NBINS), dtype=np.float64)
        for i in range(bins.NBINS):
            D_bins[...,i]  = D_INIT_AMBIENT * bins.mass_fracs[i]
            D_bins[...,i][mask1] = D_INIT_CLOUD1 * bins.mass_fracs[i]
            D_bins[...,i][mask2] = D_INIT_CLOUD2 * bins.mass_fracs[i]
        with open(GROWZONE_FILE, "w") as f:
            f.write("# t[kyr]  " + "  ".join([f"frac{i}" for i in range(bins.NBINS)]) + "\n")
    else:
        bins = None
        D_bins = None

    # header de mass history
    with open(MASS_FILE, "w") as f:
        f.write("# t[kyr]   M_gas[Msun]   M_dust[Msun]\n")

    E = P/(gamma-1.0) + 0.5*rho*(vx*vx+vy*vy+vz*vz)
    U = np.stack([rho, rho*vx, rho*vy, rho*vz, E], axis=0)
    return U, D_bins, fcld1, fcld2, bins

# ------------------------- PRIMITIVES / FLUXES ------------------------------


def masses(U, D_bins):
    M_gas = U[0].sum()*VOL_CELL/MSUN
    if (D_bins is None):
        M_dust = 0.0
    else:
        Dtot = np.sum(np.clip(D_bins,0,None), axis=-1)
        M_dust = (U[0]*Dtot).sum()*VOL_CELL/MSUN
    return M_gas, M_dust

# --------------------------------- MAIN -------------------------------------

