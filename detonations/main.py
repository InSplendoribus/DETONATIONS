import os, sys, logging, glob
import numpy as np
import h5py
from . import read_data

def _consume(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def _kyr(t_sec):
    return t_sec / (float(globals().get("YR_S", 3.15576e7)) * 1.0e3)

def _log_snapshot_messages(snap_id, t_sec):
    """Emit user-facing lines matching your expected format."""
    outdir = globals().get("OUTDIR", ".")
    basen  = globals().get("BASEN", "clouds")
    png   = os.path.join(outdir, f"snap_{basen}_{snap_id:04d}.png")
    asc   = os.path.join(outdir, f"asc-xy-{basen}_{snap_id:04d}")
    h5plt = os.path.join(outdir, f"3D_{basen}_hdf5_{snap_id:04d}")
    logging.info(f"[snap]  wrote {png}  t={_kyr(t_sec):6.2f} kyr")
    logging.info(f"[ascii] wrote {asc}")
    logging.info(f"[hdf5]  wrote {h5plt}")

def log_header():
    # Print domain extents in cm (not pc)
    pc2cm = float(globals().get("PC_CM", 3.0856775814913673e18))
    xmin_cm = float(globals().get("XMIN_PC", 0.0)) * pc2cm
    xmax_cm = float(globals().get("XMAX_PC", 0.0)) * pc2cm
    ymin_cm = float(globals().get("YMIN_PC", 0.0)) * pc2cm
    ymax_cm = float(globals().get("YMAX_PC", 0.0)) * pc2cm
    zmin_cm = float(globals().get("ZMIN_PC", 0.0)) * pc2cm
    zmax_cm = float(globals().get("ZMAX_PC", 0.0)) * pc2cm

    logging.info("==============================================================")
    logging.info("= HLLE hydro + tracers + dust (cinder)                      =")
    logging.info("==============================================================")
    logging.info(f"Grid: {Nx}x{Ny}x{Nz}  Domain [cm]: "
                 f"[{xmin_cm:.6e},{xmax_cm:.6e}] × "
                 f"[{ymin_cm:.6e},{ymax_cm:.6e}] × "
                 f"[{zmin_cm:.6e},{zmax_cm:.6e}] | DIM3D={DIM3D}")
    logging.info(f"Cooling={COOLING_ON}  Dust={DUST_ON}  Tracers={TRACERS_ON}  View={VIEW_PLANE}")
    logging.info(f"Turbulence: ON={TURB_ON}  sigmas[km/s]: amb={SIGMA_T_AMB_KMS}  c1={SIGMA_T_C1_KMS}  c2={SIGMA_T_C2_KMS}")

# ----------------------- tolerant checkpoint I/O -----------------------

def _load_checkpoint(path):
    """
    Load checkpoint/plotfile. Accepts 'rho' or 'dens'.
    If momentum/energy are missing (old plotfile), fills safe defaults so restart still works.
    Returns: U, D_bins, f1, f2, t_sec, step, chk_id, snap_next
    """
    with h5py.File(path, "r") as h:
        # density key: 'rho' or 'dens'
        if "rho" in h:
            k_rho = "rho"
        elif "dens" in h:
            k_rho = "dens"
        else:
            raise KeyError("Unable to open object (neither 'rho' nor 'dens' found)")

        Nx_, Ny_, Nz_ = h[k_rho].shape
        U = np.zeros((5, Nx_, Ny_, Nz_), dtype=np.float64)
        U[0] = h[k_rho][...]

        # momentum/energy keys (optional but preferred)
        k_mx = "mx" if "mx" in h else None
        k_my = "my" if "my" in h else None
        k_mz = "mz" if "mz" in h else None
        k_E  = "E"  if "E"  in h else None

        have_mom = (k_mx is not None) and (k_my is not None) and (k_mz is not None)
        have_E   = (k_E  is not None)

        if have_mom:
            U[1] = h[k_mx][...]
            U[2] = h[k_my][...]
            U[3] = h[k_mz][...]
        else:
            U[1] = 0.0; U[2] = 0.0; U[3] = 0.0

        if have_E:
            U[4] = h[k_E][...]
        else:
            # build thermal energy from ambient/default temperature
            T_fill = globals().get("RESTART_T_K", globals().get("T_AMB_K", 1.0e4))
            mu     = float(globals().get("mu_mol", 0.61)) * float(globals().get("mH", 1.6735575e-24))
            kB     = float(globals().get("kB", 1.380649e-16))
            gamma  = float(globals().get("gamma", 5.0/3.0))
            rho    = U[0]
            P      = np.maximum((rho/np.maximum(mu,1e-30)) * kB * T_fill, 1e-30)
            U[4]   = P/(gamma-1.0)

        # tracers
        f1 = h["f1"][...] if "f1" in h else (h["tr1"][...] if "tr1" in h else np.zeros((Nx_,Ny_,Nz_)))
        f2 = h["f2"][...] if "f2" in h else (h["tr2"][...] if "tr2" in h else np.zeros((Nx_,Ny_,Nz_)))

        # dust bins: dst*
        dust_keys = sorted([k for k in h.keys() if k.startswith("dst")])
        D_bins = None
        if dust_keys:
            NB = len(dust_keys)
            D_bins = np.zeros((Nx_, Ny_, Nz_, NB), dtype=np.float64)
            for i,k in enumerate(dust_keys):
                D_bins[..., i] = h[k][...]
        else:
            NB = int(h.attrs.get("NBINS", 0))

        # small helper
        def _get_scalar(name, default=None):
            if name in h:
                try: return h[name][()].item() if hasattr(h[name][()], "item") else type(default or 0)(h[name][()])
                except Exception: pass
            if name in h.attrs:
                try: 
                    v = h.attrs[name]
                    return v.item() if hasattr(v, "item") else v
                except Exception: 
                    pass
            return default

        # metadata
        t_sec     = float(_get_scalar("time", 0.0))
        step      = int(_get_scalar("nstep", 0))

        # NEW: separate ids saved in checkpoint
        chk_id    = _get_scalar("chk", None)
        snap_next = _get_scalar("snap", None)

        # Backward compatibility (older checkpoints stored only 'snap' = chk_id)
        if chk_id is None and snap_next is not None:
            try:
                chk_id = int(snap_next)
                snap_next = int(snap_next)  # best-effort; no separate next-snap info existed
            except Exception:
                chk_id = 0
                snap_next = 0
        if chk_id is None:
            # As a last resort infer from filename suffix if possible
            base = os.path.basename(path)
            try:
                chk_id = int(base.split("_")[-1])
            except Exception:
                chk_id = 0
        if snap_next is None:
            snap_next = 0

    return U, D_bins, f1, f2, t_sec, step, int(chk_id), int(snap_next)

def _find_latest_checkpoint(outdir, basen):
    """
    Prefer: 3D_<basen>_hdf5_chk_*
    Fall back: any *chk*; choose newest by mtime.
    """
    if not os.path.isdir(outdir):
        return None

    preferred = sorted(
        (os.path.join(outdir, n) for n in os.listdir(outdir)
         if n.startswith(f"3D_{basen}_hdf5_chk_") and os.path.isfile(os.path.join(outdir, n))),
        key=os.path.getmtime
    )
    if preferred:
        return preferred[-1]

    allchk = sorted(
        (os.path.join(outdir, n) for n in os.listdir(outdir)
         if ("chk" in n) and os.path.isfile(os.path.join(outdir, n))),
        key=os.path.getmtime
    )
    return allchk[-1] if allchk else None

def _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_next, outdir, basen):
    """
    Self-contained checkpoint with plotfile-equivalent content + restart fields.
    Name: 3D_<basen>_hdf5_chk_####  (matches your plotfile style).
    Saves *both*:
      - 'chk'  : checkpoint id (#### above)
      - 'snap' : NEXT snapshot/plotfile id to use after restart
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"3D_{basen}_hdf5_chk_{int(chk_id):04d}")

    # primitives for common plotfile extras
    gamma = float(globals().get("gamma", 5.0/3.0))
    kB    = float(globals().get("kB", 1.380649e-16))
    mu    = float(globals().get("mu_mol", 0.61)) * float(globals().get("mH", 1.6735575e-24))
    rho   = np.maximum(U[0], 1e-30)
    vx    = U[1]/rho
    vy    = U[2]/rho
    vz    = U[3]/rho
    ek    = 0.5*rho*(vx*vx+vy*vy+vz*vz)
    P     = (gamma-1.0)*np.maximum(U[4]-ek, 0.0)
    T     = P*mu/(np.maximum(rho,1e-30)*kB)

    with h5py.File(path, "w") as h:
        # core conserved fields
        h.create_dataset("rho", data=U[0], compression="gzip")
        h.create_dataset("mx",  data=U[1], compression="gzip")
        h.create_dataset("my",  data=U[2], compression="gzip")
        h.create_dataset("mz",  data=U[3], compression="gzip")
        h.create_dataset("E",   data=U[4], compression="gzip")

        # plotfile-like extras (so chk looks like plt):
        h.create_dataset("dens", data=U[0], compression="gzip")  # alias for rho
        h.create_dataset("vx",   data=vx,   compression="gzip")
        h.create_dataset("vy",   data=vy,   compression="gzip")
        h.create_dataset("vz",   data=vz,   compression="gzip")
        h.create_dataset("P",    data=P,    compression="gzip")
        h.create_dataset("T",    data=T,    compression="gzip")

        # tracers if present
        if f1 is not None: h.create_dataset("f1", data=f1, compression="gzip")
        if f2 is not None: h.create_dataset("f2", data=f2, compression="gzip")

        # dust bins if present
        NB = 0
        if D_bins is not None:
            NB = D_bins.shape[-1]
            for i in range(NB):
                h.create_dataset(f"dst{i}", data=D_bins[..., i], compression="gzip")
        h.attrs["NBINS"] = NB

        # run metadata (as datasets + a few attrs)
        h.create_dataset("time",  data=np.array(t, dtype=np.float64))
        h.create_dataset("nstep", data=np.array(step, dtype=np.int64))
        h.create_dataset("chk",   data=np.array(chk_id, dtype=np.int64))   # this checkpoint id
        h.create_dataset("snap",  data=np.array(snap_next, dtype=np.int64))# NEXT snapshot id

        # grid info for completeness
        h.attrs["dx"] = float(globals().get("dx", 0.0))
        h.attrs["dy"] = float(globals().get("dy", 0.0))
        h.attrs["dz"] = float(globals().get("dz", 0.0))
        h.attrs["Nx"] = int(globals().get("Nx", 0))
        h.attrs["Ny"] = int(globals().get("Ny", 0))
        h.attrs["Nz"] = int(globals().get("Nz", 0))
        h.attrs["gamma"] = float(gamma)

    return path

# ----------------------- CFL + stepping helpers -----------------------

def _cfl_dt(hydro, U):
    """
    Robust CFL timestep: prefer hydro.cfl_dt(U), else hydro.dt_cfl(U),
    else compute inline from U and grid spacings.
    """
    if hasattr(hydro, "cfl_dt"):
        return hydro.cfl_dt(U)
    if hasattr(hydro, "dt_cfl"):
        return hydro.dt_cfl(U)

    # Inline fallback
    rho = np.maximum(U[0], 1e-30)
    vx  = U[1]/rho; vy = U[2]/rho; vz = U[3]/rho
    ek  = 0.5*rho*(vx*vx+vy*vy+vz*vz)
    P   = (globals().get("gamma", 5.0/3.0)-1.0)*np.maximum(U[4]-ek, 0.0)
    a   = np.sqrt(float(globals().get("gamma", 5.0/3.0))*np.maximum(P,1e-30)/rho)
    vmax_x = float(np.max(np.abs(vx) + a))
    vmax_y = float(np.max(np.abs(vy) + a))
    vmax_z = float(np.max(np.abs(vz) + a)) if DIM3D else 0.0
    dt_x = dx / vmax_x if vmax_x > 0.0 else 1e99
    dt_y = dy / vmax_y if vmax_y > 0.0 else 1e99
    dt_z = dz / vmax_z if DIM3D and vmax_z > 0.0 else 1e99
    return CFL * min(dt_x, dt_y, dt_z)

def _step_with_fluxes(hydro, U, dt):
    """
    Advance U by dt and try to obtain face mass fluxes for consistent scalar advection.
    Returns (U_new, (Fx,Fy,Fz)) when possible; otherwise (U_new, (None,None,None)).
    """
    U_prev = U
    # Try the "rich" signature that returns fluxes
    try:
        ret = hydro.step_euler(U, dt, return_fluxes=True)
        if isinstance(ret, tuple) and len(ret) == 2:
            return ret
        elif ret is not None:
            U = ret
        else:
            U = hydro.step_euler(U, dt)
    except TypeError:
        U = hydro.step_euler(U, dt)

    # If not provided, try helpers
    if hasattr(hydro, "_face_mass_fluxes"):
        try:
            Fx, Fy, Fz = hydro._face_mass_fluxes(U_prev)
            return U, (Fx, Fy, Fz)
        except Exception:
            pass
    if hasattr(hydro, "_face_fluxes"):
        try:
            Fx, Fy, Fz = hydro._face_fluxes(U_prev)
            return U, (Fx, Fy, Fz)
        except Exception:
            pass

    return U, (None, None, None)

# ----------------------- main -----------------------

def main(par_path="detonations.par"):
    # 1) Parse config and write settings.py
    cfg = read_data.load_config(par_path)
    read_data.write_settings_module(cfg, os.path.join(os.path.dirname(__file__), "settings.py"))

    # 2) Lazy imports (after settings exist) and lift constants into globals
    from .hydro import solvers as hydro
    from .clouds import clouds as clouds_mod
    from .cooling import cooling as cooling_mod
    from .plotting import plotting as plotting
    from .cinder import cinder
    from . import settings as settings
    globals().update({k: getattr(settings, k) for k in dir(settings) if not k.startswith('_')})

    os.makedirs(OUTDIR, exist_ok=True)

    # 3) Logging (to LOGFILE + console)
    logging.getLogger("").handlers.clear()
    logging.basicConfig(
        filename=LOGFILE, filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger("").addHandler(console)

    log_header()

    # 4) Build dust bins (if enabled)
    bins = None
    if DUST_ON:
        bins = cinder.setup_bins()

    # 5) Restart or ICs (+ initial outputs)
    if RESTART:
        # explicit snap
        snapreq = int(RESTART_SNAP) if isinstance(RESTART_SNAP, (int, float, str)) else 0
        path = None
        if snapreq > 0:
            candidate = os.path.join(OUTDIR, f"3D_{BASEN}_hdf5_chk_{int(snapreq):04d}")
            if os.path.exists(candidate):
                path = candidate
            else:
                logging.warning(f"[restart] requested chk {snapreq:04d} not found: {candidate}. Falling back to latest.")
        if path is None:
            path = _find_latest_checkpoint(OUTDIR, BASEN)

        if path and os.path.exists(path):
            logging.info(f"[restart] loading checkpoint: {path}")
            U, D_bins, f1, f2, t, step, loaded_chk, saved_snap_next = _load_checkpoint(path)
            if DUST_ON and (D_bins is None) and (bins is not None):
                D_bins = np.zeros((Nx, Ny, Nz, bins.NBINS), dtype=np.float64)

            # Initialize counters so next outputs do not collide
            chk_id  = int(loaded_chk) + 1     # next checkpoint index
            snap_id = saved_snap_next # next snapshot/plotfile index (from checkpoint!)
            if (SNAP_NEXT_OVERRIDE is not None) and (SNAP_NEXT_OVERRIDE >= 0):
                snap_id = int(SNAP_NEXT_OVERRIDE) 
            logging.info(f"[restart] time={t/YR_S:.6e} yr  step={step}  next chk={chk_id:04d}  next snap={snap_id:04d}")
            # Per request: do NOT write a new chk or a t=0 snapshot at restart.
        else:
            logging.info("[restart] no checkpoint found; starting from ICs.")
            U, D_bins, f1, f2, bins = clouds_mod.init_state()
            t = 0.0; step = 0; snap_id = 0; chk_id = 0
            # initial outputs (fresh run): PNG + HDF5 plotfile + ASCII (logged)
            plotting.make_png(U, t/YR_S, snap_id, D_bins, f1, f2)
            try:
                plotting.write_h5_3d(U, D_bins, f1, f2, t, step, snap_id, kind="snap")
            except Exception:
                pass
            _log_snapshot_messages(snap_id, t)
            snap_id += 1
            # time series line
            Mg, Md = clouds_mod.masses(U, D_bins)
            with open(MASS_FILE, "a") as f:
                f.write(f"{t/YR_S/1e3:12.5f}  {Mg: .8e}  {Md: .8e}\n")
            # initial checkpoint at t=0 (save next snapshot id)
            _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_id, OUTDIR, BASEN); chk_id += 1
    else:
        U, D_bins, f1, f2, bins = clouds_mod.init_state()
        t = 0.0; step = 0; snap_id = 0; chk_id = 0
        # initial outputs (fresh run): PNG + HDF5 plotfile + ASCII (logged)
        plotting.make_png(U, t/YR_S, snap_id, D_bins, f1, f2)
        try:
            plotting.write_h5_3d(U, D_bins, f1, f2, t, step, snap_id, kind="snap")
        except Exception:
            pass
        _log_snapshot_messages(snap_id, t)
        snap_id += 1
        # time series line
        Mg, Md = clouds_mod.masses(U, D_bins)
        with open(MASS_FILE, "a") as f:
            f.write(f"{t/YR_S/1e3:12.5f}  {Mg: .8e}  {Md: .8e}\n")
        # initial checkpoint at t=0 (save next snapshot id)
        _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_id, OUTDIR, BASEN); chk_id += 1

    TEND = T_END_YR * YR_S

    # checkpoint cadence tracker
    last_chk_time = t  # seconds
    warned_no_flux = False

    # 6) Time loop
    while t < TEND:
        dt = _cfl_dt(hydro, U); dt = min(dt, TEND - t)

        # --- keep rho^n (for consistent scalar transport) ---
        rho_old = U[0].copy()

        # --- hydro step; try to get face fluxes for consistent advection ---
        U, (Fx, Fy, Fz) = _step_with_fluxes(hydro, U, dt)

        # --- tracers: consistent conservative advection with gas mass fluxes ---
        if TRACERS_ON:
            if Fx is not None:
                f1 = hydro.advect_scalar_consistent(rho_old, U[0], f1, dt, Fx, Fy, Fz)
                f2 = hydro.advect_scalar_consistent(rho_old, U[0], f2, dt, Fx, Fy, Fz)
            else:
                if not warned_no_flux:
                    logging.warning("No face fluxes available; skipping tracer advection this step.")
                    warned_no_flux = True

        # --- dust bins: consistent advection ---
        if DUST_ON and (bins is not None):
            for i in range(bins.NBINS):
                if Fx is not None:
                    D_bins[..., i] = hydro.advect_scalar_consistent(rho_old, U[0], D_bins[..., i], dt, Fx, Fy, Fz)
                else:
                    if not warned_no_flux:
                        logging.warning("No face fluxes available; skipping dust advection this step.")
                        warned_no_flux = True

        # --- radiative + dust cooling ---
        if COOLING_ON:
            U = cooling_mod.apply_cooling(U, dt, bins, D_bins)        
        
        # --- dust growth + sputtering (updates D_bins mass in-place) ---
        if DUST_ON and (bins is not None) and (D_bins is not None):
            D_bins = cinder.apply_growth_and_sputtering(D_bins, U, dt, bins, mu_mol, kB, dx, dy, dz)    

        # advance time
        t += dt
        step += 1

        # --- periodic mass & growth-zone stats ---
        if step % TSERIES_EVERY == 0:
            Mg, Md = clouds_mod.masses(U, D_bins)
            with open(MASS_FILE, "a") as f:
                f.write(f"{t/YR_S/1e3:12.5f}  {Mg: .8e}  {Md: .8e}\n")

            if DUST_ON and (bins is not None):
                mask = cinder.growth_mask(U, mu_mol, kB)
                if np.any(mask):
                    Dtot = np.sum(np.clip(D_bins, 0, None), axis=-1)
                    num  = D_bins[mask].sum(axis=0)
                    den  = np.maximum(Dtot[mask].sum(), 1e-30)
                    frac = num / den
                    with open(GROWZONE_FILE, "a") as f:
                        f.write(f"{t/YR_S/1e3:12.5f}  " + "  ".join(f"{x: .6e}" for x in frac) + "\n")

        # --- periodic plots & snapshots ---
        if step % SNAP_EVERY == 0:
            plotting.make_png(U, t/YR_S, snap_id, D_bins, f1, f2)
            try:
                plotting.write_h5_3d(U, D_bins, f1, f2, t, step, snap_id, kind="snap")
            except Exception:
                pass
            _log_snapshot_messages(snap_id, t)
            snap_id += 1

        # --- periodic checkpoints (cadence) ---
        if (CHK_EVERY_STEPS > 0) and (step % CHK_EVERY_STEPS == 0):
            _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_id, OUTDIR, BASEN); chk_id += 1
            last_chk_time = t

        if (CHK_EVERY_SECONDS > 0) and (t - last_chk_time >= CHK_EVERY_SECONDS):
            _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_id, OUTDIR, BASEN); chk_id += 1
            last_chk_time = t

# --- sentinel files (checked after each step)
        def _consume(flag):
            try:
                os.remove(flag)
            except FileNotFoundError:
                pass

        flag_restart   = os.path.join(OUTDIR, ".dump_restart")
        flag_checkpoint= os.path.join(OUTDIR, ".dump_checkpoint")
        flag_plotfile  = os.path.join(OUTDIR, ".dump_plotfile")

        if os.path.exists(flag_plotfile):
            plotting.write_h5_3d(U, D_bins, f1, f2, t, step, snap_id, kind="plt")
            _consume(flag_plotfile)

        if os.path.exists(flag_checkpoint):
            _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_id, OUTDIR, BASEN); chk_id += 1
            _consume(flag_checkpoint)

        if os.path.exists(flag_restart):
            _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_id, OUTDIR, BASEN); chk_id += 1
            _consume(flag_restart)
            logging.info("[restart] sentinel requested; checkpoint written. Exiting.")
            break

    logging.info("=== finished ===")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'detonations.par')
