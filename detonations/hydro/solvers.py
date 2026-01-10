from ..settings import *
import numpy as np
try:
    from ..cinder import cinder
except Exception:
    cinder = None

# per-timestep scalar-call counter for wind bin indexing (reset in step_euler)
_WIND_SCALAR_CALL_COUNT = 0


# ------------------ Wind tunnel helpers ------------------
def _mu_piecewise(T):
    """Piecewise μ(T) in grams (consistent with cooling.py)."""
    mu_low  = 2.33 * mH
    mu_mid  = 1.27 * mH
    mu_high = 0.61 * mH
    if np.isscalar(T):
        if T < 5.0e3:   return mu_low
        elif T < 1.0e4: return mu_mid
        else:           return mu_high
    T = np.asarray(T)
    return np.where(T < 5.0e3, mu_low, np.where(T < 1.0e4, mu_mid, mu_high))


def _wind_inflow_state():
    """Build conserved U for the wind ghost cells from settings."""
    if not globals().get('WIND_ON', False):
        return None
    # velocities (prefer cgs if present, else derive from *_KMS)
    vx = globals().get('WIND_VX', None)
    vy = globals().get('WIND_VY', None)
    vz = globals().get('WIND_VZ', None)
    if vx is None: vx = globals().get('WIND_VX_KMS', 0.0) * 1.0e5
    if vy is None: vy = globals().get('WIND_VY_KMS', 0.0) * 1.0e5
    if vz is None: vz = globals().get('WIND_VZ_KMS', 0.0) * 1.0e5

    T  = globals().get('WIND_T', 1.0e7)
    n  = globals().get('WIND_N_CM3', 1.0e-2)  # total number density [cm^-3]
    mu = _mu_piecewise(T)
    rho = np.maximum(mu * n, 1e-30)
    P   = np.maximum(n * kB * T, 1e-30)
    ek  = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
    E   = P/(gamma-1.0) + ek

    U_in = np.zeros((5,), dtype=float)
    U_in[0] = rho
    U_in[1] = rho*vx
    U_in[2] = rho*vy
    U_in[3] = rho*vz
    U_in[4] = E
    return U_in


# --------------------------- PRIMITIVES / FLUXES ----------------------------
def prim_from_cons(U):
    """Return (rho, vx, vy, vz, P) from conserved U=[rho,mx,my,mz,E]."""
    rho = np.maximum(U[0], 1e-30)
    vx  = U[1]/rho
    vy  = U[2]/rho
    vz  = U[3]/rho
    ek  = 0.5*rho*(vx*vx+vy*vy+vz*vz)
    P   = (gamma-1.0)*np.maximum(U[4]-ek, 0.0)
    return rho, vx, vy, vz, P


def cons_from_prim(rho, vx, vy, vz, P):
    U = np.empty((5,)+rho.shape, dtype=np.float64)
    U[0] = rho
    U[1] = rho*vx
    U[2] = rho*vy
    U[3] = rho*vz
    U[4] = P/(gamma-1.0) + 0.5*rho*(vx*vx+vy*vy+vz*vz)
    return U


def sound_speed(P, rho):
    """a = sqrt(gamma*P/rho). (Order kept as (P,rho) to match our calls.)"""
    return np.sqrt(gamma*np.maximum(P,1e-30)/np.maximum(rho,1e-30))


def _flux_components(U, axis):
    """Return physical flux F(U) along given axis 'x'|'y'|'z'. Shape: (5, ...)."""
    rho, vx, vy, vz, P = prim_from_cons(U)
    E = U[4]
    if axis == 'x':
        vn = vx
        F = np.empty_like(U)
        F[0] = rho*vn
        F[1] = rho*vn*vx + P
        F[2] = rho*vn*vy
        F[3] = rho*vn*vz
        F[4] = (E + P)*vn
        return F
    if axis == 'y':
        vn = vy
        F = np.empty_like(U)
        F[0] = rho*vn
        F[1] = rho*vn*vx
        F[2] = rho*vn*vy + P
        F[3] = rho*vn*vz
        F[4] = (E + P)*vn
        return F
    if axis == 'z':
        vn = vz
        F = np.empty_like(U)
        F[0] = rho*vn
        F[1] = rho*vn*vx
        F[2] = rho*vn*vy
        F[3] = rho*vn*vz + P
        F[4] = (E + P)*vn
        return F
    raise ValueError("axis must be 'x','y' or 'z'")


def _hlle(UL, UR, axis):
    """
    HLLE flux between left/right conserved states along given axis.
    UL, UR shapes: (5, Nx-1, Ny, Nz) for x; similar for y/z.
    Returns: F_hlle (5, faces...), mass_flux (faces...).
    """
    # primitives
    rhoL, vxL, vyL, vzL, PL = prim_from_cons(UL)
    rhoR, vxR, vyR, vzR, PR = prim_from_cons(UR)

    if axis == 'x':
        vnL, vnR = vxL, vxR
    elif axis == 'y':
        vnL, vnR = vyL, vyR
    else:
        vnL, vnR = vzL, vzR

    aL = sound_speed(PL, rhoL)
    aR = sound_speed(PR, rhoR)

    sL = np.minimum(vnL - aL, vnR - aR)
    sR = np.maximum(vnL + aL, vnR + aR)

    FL = _flux_components(UL, axis)
    FR = _flux_components(UR, axis)

    # upwind cases
    F = np.where(sL >= 0.0, FL, np.where(sR <= 0.0, FR,
         (sR*FL - sL*FR + (sL*sR)*(UR - UL)) / np.maximum(sR - sL, 1e-30)))
    mass_flux = F[0]
    return F, mass_flux


# --------------------------- RIEMANN / FACE FLUXES --------------------------
def _face_fluxes(U):
    """
    Compute face fluxes (vector) and mass fluxes in x/y/z using HLLE.
    Returns:
      Fx_vec (5,Nx+1,Ny,Nz), Fy_vec (5,Nx,Ny+1,Nz), Fz_vec (5,Nx,Ny,Nz+1),
      Mx (Nx+1,Ny,Nz), My (Nx,Ny+1,Nz), Mz (Nx,Ny,Nz+1)
    """
    Nx_, Ny_, Nz_ = U.shape[1:4]

    # -- X faces --
    UL = U[:, 0:Nx_-1, :, :]
    UR = U[:, 1:Nx_  , :, :]
    Fx_mid, Mx_mid = _hlle(UL, UR, 'x')                # (5,Nx-1,Ny,Nz)
    Fx_vec = np.zeros((5, Nx_+1, Ny_, Nz_), dtype=U.dtype)
    Fx_vec[:, 1:Nx_, :, :] = Fx_mid
    # boundaries: use physical flux F(U) (zero-gradient), then override if wind face
    Fx_vec[:, 0, :, :]  = _flux_components(U[:, 0, :, :], 'x')
    Fx_vec[:, -1, :, :] = _flux_components(U[:, -1, :, :], 'x')
    Mx = Fx_vec[0]

    # -- Y faces --
    UL = U[:, :, 0:Ny_-1, :]
    UR = U[:, :, 1:Ny_  , :]
    Fy_mid, My_mid = _hlle(UL, UR, 'y')                # (5,Nx,Ny-1,Nz)
    Fy_vec = np.zeros((5, Nx_, Ny_+1, Nz_), dtype=U.dtype)
    Fy_vec[:, :, 1:Ny_, :] = Fy_mid
    Fy_vec[:, :, 0, :]  = _flux_components(U[:, :, 0, :], 'y')
    Fy_vec[:, :, -1, :] = _flux_components(U[:, :, -1, :], 'y')
    My = Fy_vec[0]

    # -- Z faces --
    Fz_vec = np.zeros((5, Nx_, Ny_, Nz_+1), dtype=U.dtype)
    if DIM3D:
        UL = U[:, :, :, 0:Nz_-1]
        UR = U[:, :, :, 1:Nz_  ]
        Fz_mid, Mz_mid = _hlle(UL, UR, 'z')            # (5,Nx,Ny,Nz-1)
        Fz_vec[:, :, :, 1:Nz_] = Fz_mid
        Fz_vec[:, :, :, 0]  = _flux_components(U[:, :, :, 0], 'z')
        Fz_vec[:, :, :, -1] = _flux_components(U[:, :, :, -1], 'z')
    Mz = Fz_vec[0]

    # ---- wind inflow: override the corresponding boundary face flux with HLLE against inflow state
    if globals().get('WIND_ON', False):
        face = globals().get('WIND_FACE', 'x-')
        U_in = _wind_inflow_state()
        if U_in is not None:
            if face == 'x-':
                UL = U_in[:, None, None]   # ghost (left)
                UR = U[:, 0, :, :]        # boundary cell
                F, M = _hlle(UL, UR, 'x')
                Fx_vec[:, 0, :, :] = F
                Mx[0, :, :] = M
            elif face == 'x+':
                UL = U[:, -1, :, :]
                UR = U_in[:, None, None]   # ghost (right)
                F, M = _hlle(UL, UR, 'x')
                Fx_vec[:, -1, :, :] = F
                Mx[-1, :, :] = M
            elif face == 'y-':
                UL = U_in[:, None, None]
                UR = U[:, :, 0, :]
                F, M = _hlle(UL, UR, 'y')
                Fy_vec[:, :, 0, :] = F
                My[:, 0, :] = M
            elif face == 'y+':
                UL = U[:, :, -1, :]
                UR = U_in[:, None, None]
                F, M = _hlle(UL, UR, 'y')
                Fy_vec[:, :, -1, :] = F
                My[:, -1, :] = M
            elif face == 'z-' and DIM3D:
                UL = U_in[:, None, None]
                UR = U[:, :, :, 0]
                F, M = _hlle(UL, UR, 'z')
                Fz_vec[:, :, :, 0] = F
                Mz[:, :, 0] = M
            elif face == 'z+' and DIM3D:
                UL = U[:, :, :, -1]
                UR = U_in[:, None, None]
                F, M = _hlle(UL, UR, 'z')
                Fz_vec[:, :, :, -1] = F
                Mz[:, :, -1] = M

    return Fx_vec, Fy_vec, Fz_vec, Mx, My, Mz


# --------------------------- UPDATE / BOUNDARIES ----------------------------
def apply_bc(U):
    # default copy BCs
    U[:, 0, :, :]  = U[:, 1, :, :]
    U[:, -1, :, :] = U[:, -2, :, :]
    U[:, :, 0, :]  = U[:, :, 1, :]
    U[:, :, -1, :] = U[:, :, -2, :]
    if DIM3D:
        U[:, :, :, 0]  = U[:, :, :, 1]
        U[:, :, :, -1] = U[:, :, :, -2]

    # wind inflow (override boundary cells with inflow state)
    if globals().get('WIND_ON', False):
        face = globals().get('WIND_FACE', 'x-')
        U_in = _wind_inflow_state()
        if U_in is not None:
            if face == 'x-':
                U[:, 0, :, :] = U_in[:, None, None]
            elif face == 'x+':
                U[:, -1, :, :] = U_in[:, None, None]
            elif face == 'y-':
                U[:, :, 0, :] = U_in[:, None, None]
            elif face == 'y+':
                U[:, :, -1, :] = U_in[:, None, None]
            elif face == 'z-' and DIM3D:
                U[:, :, :, 0] = U_in[:, None, None]
            elif face == 'z+' and DIM3D:
                U[:, :, :, -1] = U_in[:, None, None]
    return U


def cfl_dt(U):
    """Global CFL time step using max(|v|+a) in each direction."""
    rho, vx, vy, vz, P = prim_from_cons(U)
    a = sound_speed(P, rho)
    vmax_x = float(np.max(np.abs(vx) + a))
    vmax_y = float(np.max(np.abs(vy) + a))
    vmax_z = float(np.max(np.abs(vz) + a)) if DIM3D else 0.0
    dt_x = dx / vmax_x if vmax_x > 0.0 else 1e99
    dt_y = dy / vmax_y if vmax_y > 0.0 else 1e99
    dt_z = dz / vmax_z if DIM3D and vmax_z > 0.0 else 1e99
    return CFL * min(dt_x, dt_y, dt_z)


def step_euler(U, dt, return_fluxes=False):
    """
    First-order Godunov update with HLLE fluxes; returns U (and face mass fluxes if requested).
    """
    # Enforce BCs (so boundary cells are valid for face fluxes)
    U = apply_bc(U)

    # Face fluxes
    Fx_vec, Fy_vec, Fz_vec, Mx, My, Mz = _face_fluxes(U)

    # Divergence update
    U_new = U.copy()
    # x
    U_new[:, :, :, :] -= (dt/dx) * (Fx_vec[:, 1:, :, :] - Fx_vec[:, :-1, :, :])
    # y
    U_new[:, :, :, :] -= (dt/dy) * (Fy_vec[:, :, 1:, :] - Fy_vec[:, :, :-1, :])
    # z
    if DIM3D:
        U_new[:, :, :, :] -= (dt/dz) * (Fz_vec[:, :, :, 1:] - Fz_vec[:, :, :, :-1])

    # Clip internal energy to avoid negatives
    rho, vx, vy, vz, P = prim_from_cons(U_new)
    ek  = 0.5*rho*(vx*vx+vy*vy+vz*vz)
    eint = np.maximum(U_new[4] - ek, 1e-30)
    U_new[4] = eint + ek

    # Apply BCs after update
    U_new = apply_bc(U_new)

    # Reset scalar-call counter (used by advect_scalar_consistent)
    global _WIND_SCALAR_CALL_COUNT
    _WIND_SCALAR_CALL_COUNT = 0

    if return_fluxes:
        return U_new, (Mx, My, Mz)
    return U_new


# Expose mass-face fluxes to main fallback if needed
def _face_mass_fluxes(U):
    """Return only the mass fluxes (ρv·n) at faces."""
    _, _, _, Mx, My, Mz = _face_fluxes(U)
    return Mx, My, Mz


# --------------------------- SCALAR ADVECTION -------------------------------
def advect_scalar_consistent(rho_old, rho_new, D, dt, Fx, Fy, Fz):
    """
    Conservative advection of passive scalar D consistent with gas mass fluxes.
    Fx, Fy, Fz are *mass* fluxes ρ v·n located at faces:
      Fx: (Nx+1,Ny,Nz), Fy: (Nx,Ny+1,Nz), Fz: (Nx,Ny,Nz+1)
    """
    Q = rho_old * D  # conservative variable

    # Upwinded scalar on faces
    # X faces
    Nx_, Ny_, Nz_ = Q.shape
    Dfx = np.zeros((Nx_+1, Ny_, Nz_), dtype=Q.dtype)
    # interior faces use upwind based on Fx sign
    sgn = Fx[1:Nx_, :, :] >= 0.0
    Dfx[1:Nx_, :, :] = np.where(sgn, D[0:Nx_-1, :, :], D[1:Nx_, :, :])
    # boundary faces: copy nearest cell (no upstream outside domain)
    Dfx[0, :, :]  = D[0, :, :]
    Dfx[-1, :, :] = D[-1, :, :]
    FQx = Fx * Dfx

    # Y faces
    Dfy = np.zeros((Nx_, Ny_+1, Nz_), dtype=Q.dtype)
    sgn = Fy[:, 1:Ny_, :] >= 0.0
    Dfy[:, 1:Ny_, :] = np.where(sgn, D[:, 0:Ny_-1, :], D[:, 1:Ny_, :])
    Dfy[:, 0, :]  = D[:, 0, :]
    Dfy[:, -1, :] = D[:, -1, :]
    FQy = Fy * Dfy

    # Z faces
    if DIM3D:
        Dfz = np.zeros((Nx_, Ny_, Nz_+1), dtype=Q.dtype)
        sgn = Fz[:, :, 1:Nz_] >= 0.0
        Dfz[:, :, 1:Nz_] = np.where(sgn, D[:, :, 0:Nz_-1], D[:, :, 1:Nz_])
        Dfz[:, :, 0]  = D[:, :, 0]
        Dfz[:, :, -1] = D[:, :, -1]
        FQz = Fz * Dfz
    else:
        FQz = 0.0

    # Update Q with flux divergence
    Q = Q - (dt/dx)*(FQx[1:, :, :] - FQx[:-1, :, :]) \
          - (dt/dy)*(FQy[:, 1:, :] - FQy[:, :-1, :]) \
          - (dt/dz)*(FQz[:, :, 1:] - FQz[:, :, :-1]) if DIM3D else \
        Q - (dt/dx)*(FQx[1:, :, :] - FQx[:-1, :, :]) \
          - (dt/dy)*(FQy[:, 1:, :] - FQy[:, :-1, :])

    # compute updated scalar
    D_new = np.where(rho_new > 0.0, Q / rho_new, 0.0)

    # simple copy BC like gas
    D_new[0, :, :]  = D_new[1, :, :]
    D_new[-1,:, :]  = D_new[-2,:, :]
    D_new[:, 0, :]  = D_new[:, 1, :]
    D_new[:, -1,:]  = D_new[:, -2,:]
    if DIM3D:
        D_new[:, :, 0]  = D_new[:, :, 1]
        D_new[:, :, -1] = D_new[:, :, -2]

    # ---- wind dust inflow on the chosen face, per-bin via mass fractions ----
    global _WIND_SCALAR_CALL_COUNT
    if globals().get('WIND_ON', False):
        _WIND_SCALAR_CALL_COUNT = int(_WIND_SCALAR_CALL_COUNT) + 1
        tracer_calls = 2 if globals().get('TRACERS_ON', False) else 0
        dust_idx = _WIND_SCALAR_CALL_COUNT - tracer_calls - 1  # 0..NBINS-1

        NB = int(globals().get('DUST_NBINS', 0))
        if (0 <= dust_idx < NB) and (cinder is not None):
            try:
                bins = cinder.setup_bins()  # honors DUST_BINS, DUST_NBINS, etc.
                fr = np.asarray(bins.mass_fracs)
                fr = fr / np.maximum(fr.sum(), 1e-30)
                Dtot = float(globals().get('WIND_DUST_TO_GAS', 0.0))
                Dbin = Dtot * fr[dust_idx]

                face = globals().get('WIND_FACE', 'x-')
                if face == 'x-':
                    D_new[0, :, :] = Dbin
                elif face == 'x+':
                    D_new[-1, :, :] = Dbin
                elif face == 'y-':
                    D_new[:, 0, :] = Dbin
                elif face == 'y+':
                    D_new[:, -1, :] = Dbin
                elif face == 'z-' and DIM3D:
                    D_new[:, :, 0] = Dbin
                elif face == 'z+' and DIM3D:
                    D_new[:, :, -1] = Dbin
            except Exception:
                pass

    return np.clip(D_new, 0.0, None)
