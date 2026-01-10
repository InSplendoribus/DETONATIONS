# -*- coding: utf-8 -*-
"""
read_data.py — Parse detonations.par (ConfigParser/INI) into a Config dataclass
and emit a self-contained `settings.py` that the rest of the code imports.

Public API used by main.py:
  - cfg = load_config(par_path)
  - write_settings_module(cfg, path_to_settings_py)

Includes a minimal wind-tunnel section:
  [wind]
  WIND_ON = True/False
  WIND_FACE = 'x-' | 'x+' | 'y-' | 'y+' | 'z-' | 'z+'
  WIND_VX_KMS, WIND_VY_KMS, WIND_VZ_KMS
  WIND_T, WIND_N_CM3
  WIND_DUST_TO_GAS
and writes corresponding constants to settings.py.
"""

import ast
from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
from typing import Tuple, Optional


# ------------------------ dataclasses ------------------------

@dataclass
class Run:
    DIM3D: bool
    INTEGRATED_MAPS: bool
    COOLING_ON: bool
    DUST_ON: bool
    TRACERS_ON: bool
    TURB_ON: bool
    TURB_SEED: int
    VIEW_PLANE: str
    T_END_YR: float
    CFL: float
    SNAP_EVERY: int
    TSERIES_EVERY: int
    # optional / defaults
    RESTART: bool = False
    RESTART_SNAP: int = 0
    SNAP_NEXT_OVERRIDE: int = -1
    CHK_EVERY_STEPS: int = 0
    CHK_EVERY_SECONDS: float = 0.0


@dataclass
class Grid:
    Nx: int
    Ny: int
    Nz: int
    XMIN_PC: float
    XMAX_PC: float
    YMIN_PC: float
    YMAX_PC: float
    ZMIN_PC: float
    ZMAX_PC: float


@dataclass
class Ambient:
    RHO_AMB: float
    T_AMB: float
    VX_AMB_KMS: float
    VY_AMB_KMS: float
    VZ_AMB_KMS: float


@dataclass
class Cloud:
    CENTER_PC: Tuple[float, float, float]
    RADIUS_PC: float
    RHO: float
    T: float
    VX_KMS: float
    VY_KMS: float
    VZ_KMS: float


@dataclass
class Turbulence:
    SIGMA_T_AMB_KMS: float
    SIGMA_T_C1_KMS: float
    SIGMA_T_C2_KMS: float


@dataclass
class Cooling:
    COOLING_ON: bool


@dataclass
class Dust:
    # initial mass fractions for ambient/clouds (dimensionless)
    D_INIT_AMBIENT: float
    D_INIT_CLOUD1: float
    D_INIT_CLOUD2: float
    # gas-dust cooling switch
    DUST_GAS_COOLING_ON: bool
    # binning controls
    DUST_BINS: str                 # 'MRN' | 'lognormal'
    DUST_NBINS: int
    DUST_A_MIN_UM: float
    DUST_A_MAX_UM: float
    LOGNORM_A0_UM: float
    LOGNORM_SIGMA: float
    MRN_Q: float


@dataclass
class Solver:
    # kept for completeness; not explicitly used in writer presently
    ORDER: Optional[int] = None
    RIEMANN: Optional[str] = None


@dataclass
class Parallel:
    # kept for completeness; not explicitly used in writer presently
    NPROC_X: Optional[int] = None
    NPROC_Y: Optional[int] = None
    NPROC_Z: Optional[int] = None


@dataclass
class Plot:
    VIEW_PLANE: str
    OUTDIR: str
    BASEN: str
    LOG_FILE: Optional[str] = None


@dataclass
class Files:
    LAMBDA_TABLE_FILE: str


# -------- wind inflow controls --------
@dataclass
class Wind:
    ON: bool
    FACE: str               # 'x-','x+','y-','y+','z-','z+'
    VX_KMS: float
    VY_KMS: float
    VZ_KMS: float
    T: float                # K
    N_CM3: float            # total number density [cm^-3]
    DUST_TO_GAS: float      # injected dust-to-gas mass ratio (distributed by bins.mass_fracs)


@dataclass
class Config:
    run: Run
    grid: Grid
    ambient: Ambient
    cloud1: Cloud
    cloud2: Cloud
    turbulence: Turbulence
    cooling: Cooling
    dust: Dust
    solver: Solver
    parallel: Parallel
    plot: Plot
    files: Files
    wind: Wind


# ------------------------ parsing helpers ------------------------

def _lit(raw: str):
    """Literal-evaluate a string when possible, fallback to stripped string."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip()


def _get_str_or_lit(cp: ConfigParser, sec: str, key: str, default=None):
    if not cp.has_section(sec) or not cp.has_option(sec, key):
        return default
    raw = cp.get(sec, key)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip()


# ------------------------ public API: read & assemble ------------------------

def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"parameters file not found: {p}")

    cp = ConfigParser()
    cp.optionxform = str  # preserve case of keys
    cp.read(p)

    def g(sec, key, default=None):
        """Get value from cp with literal eval; error if missing and no default."""
        if not cp.has_option(sec, key):
            if default is not None:
                return default
            raise KeyError(f"Missing required key [{sec}] {key}")
        return _lit(cp.get(sec, key))

    def g_basen(default="run"):
        """Try a few common key names/sections to determine BASEN."""
        cand_keys = ("BASEN", "basen", "basenm", "base", "basename")
        cand_secs = ("plot", "run")
        for sec in cand_secs:
            if cp.has_section(sec):
                for k in cand_keys:
                    val = _get_str_or_lit(cp, sec, k, default=None)
                    if val is not None:
                        return str(val)
        for sec in cp.sections():
            for k in cand_keys:
                val = _get_str_or_lit(cp, sec, k, default=None)
                if val is not None:
                    return str(val)
        return default

    def g_logfile(default=None):
        """Discover an explicit log file name if provided; returns None if not set."""
        cand_keys = ("LOG_FILE", "LOGFILE", "log_file")
        cand_secs = ("plot", "run")
        for sec in cand_secs:
            if cp.has_section(sec):
                for k in cand_keys:
                    val = _get_str_or_lit(cp, sec, k, default=None)
                    if val:
                        return str(val)
        return default

    # ---- Sections ----

    run = Run(
        DIM3D=g("run", "DIM3D"),
        INTEGRATED_MAPS=g("run", "INTEGRATED_MAPS"),
        COOLING_ON=g("run", "COOLING_ON"),
        DUST_ON=g("run", "DUST_ON"),
        TRACERS_ON=g("run", "TRACERS_ON"),
        TURB_ON=g("run", "TURB_ON"),
        TURB_SEED=g("run", "TURB_SEED"),
        VIEW_PLANE=g("run", "VIEW_PLANE"),
        T_END_YR=g("run", "T_END_YR"),
        CFL=g("run", "CFL"),
        SNAP_EVERY=g("run", "SNAP_EVERY"),
        TSERIES_EVERY=g("run", "TSERIES_EVERY"),
        RESTART=g("run", "RESTART", False),
        RESTART_SNAP=g("run", "RESTART_SNAP", 0),
        SNAP_NEXT_OVERRIDE = g("run", "SNAP_NEXT_OVERRIDE", -1),
        CHK_EVERY_STEPS=g("run", "CHK_EVERY_STEPS", 0),
        CHK_EVERY_SECONDS=g("run", "CHK_EVERY_SECONDS", 0.0),
    )

    grid = Grid(
        Nx=g("grid", "Nx"), Ny=g("grid", "Ny"), Nz=g("grid", "Nz"),
        XMIN_PC=g("grid", "XMIN_PC"), XMAX_PC=g("grid", "XMAX_PC"),
        YMIN_PC=g("grid", "YMIN_PC"), YMAX_PC=g("grid", "YMAX_PC"),
        ZMIN_PC=g("grid", "ZMIN_PC"), ZMAX_PC=g("grid", "ZMAX_PC"),
    )

    ambient = Ambient(
        RHO_AMB=g("ambient", "RHO_AMB"),
        T_AMB=g("ambient", "T_AMB"),
        VX_AMB_KMS=g("ambient", "VX_AMB_KMS"),
        VY_AMB_KMS=g("ambient", "VY_AMB_KMS"),
        VZ_AMB_KMS=g("ambient", "VZ_AMB_KMS"),
    )

    cloud1 = Cloud(
        CENTER_PC=tuple(g("cloud1", "C1_CENTER_PC")),
        RADIUS_PC=g("cloud1", "C1_RADIUS_PC"),
        RHO=g("cloud1", "C1_RHO"),
        T=g("cloud1", "C1_T"),
        VX_KMS=g("cloud1", "C1_VX_KMS"),
        VY_KMS=g("cloud1", "C1_VY_KMS"),
        VZ_KMS=g("cloud1", "C1_VZ_KMS"),
    )

    cloud2 = Cloud(
        CENTER_PC=tuple(g("cloud2", "C2_CENTER_PC")),
        RADIUS_PC=g("cloud2", "C2_RADIUS_PC"),
        RHO=g("cloud2", "C2_RHO"),
        T=g("cloud2", "C2_T"),
        VX_KMS=g("cloud2", "C2_VX_KMS"),
        VY_KMS=g("cloud2", "C2_VY_KMS"),
        VZ_KMS=g("cloud2", "C2_VZ_KMS"),
    )

    turbulence = Turbulence(
        SIGMA_T_AMB_KMS=g("turbulence", "SIGMA_T_AMB_KMS"),
        SIGMA_T_C1_KMS=g("turbulence", "SIGMA_T_C1_KMS"),
        SIGMA_T_C2_KMS=g("turbulence", "SIGMA_T_C2_KMS"),
    )

    cooling = Cooling(COOLING_ON=g("cooling", "COOLING_ON"))

    dust = Dust(
        D_INIT_AMBIENT=g("dust", "D_INIT_AMBIENT"),
        D_INIT_CLOUD1=g("dust", "D_INIT_CLOUD1"),
        D_INIT_CLOUD2=g("dust", "D_INIT_CLOUD2"),
        DUST_GAS_COOLING_ON=g("dust", "DUST_GAS_COOLING_ON"),
        DUST_BINS=g("dust", "DUST_BINS"),
        DUST_NBINS=g("dust", "DUST_NBINS"),
        DUST_A_MIN_UM=g("dust", "DUST_A_MIN_UM"),
        DUST_A_MAX_UM=g("dust", "DUST_A_MAX_UM"),
        LOGNORM_A0_UM=g("dust", "LOGNORM_A0_UM"),
        LOGNORM_SIGMA=g("dust", "LOGNORM_SIGMA"),
        MRN_Q=g("dust", "MRN_Q"),
    )

    solver = Solver(
        ORDER=_get_str_or_lit(cp, "solver", "ORDER", None),
        RIEMANN=_get_str_or_lit(cp, "solver", "RIEMANN", None),
    )

    parallel = Parallel(
        NPROC_X=_get_str_or_lit(cp, "parallel", "NPROC_X", None),
        NPROC_Y=_get_str_or_lit(cp, "parallel", "NPROC_Y", None),
        NPROC_Z=_get_str_or_lit(cp, "parallel", "NPROC_Z", None),
    )

    plot = Plot(
        VIEW_PLANE=g("plot", "VIEW_PLANE"),
        OUTDIR=g("plot", "OUTDIR"),
        BASEN=g_basen("run"),
        LOG_FILE=g_logfile(None),
    )

    files = Files(LAMBDA_TABLE_FILE=g("files", "LAMBDA_TABLE_FILE"))

    wind = Wind(
        ON=g("wind", "WIND_ON", False),
        FACE=g("wind", "WIND_FACE", "x-"),
        VX_KMS=g("wind", "WIND_VX_KMS", 777.0),
        VY_KMS=g("wind", "WIND_VY_KMS", 0.0),
        VZ_KMS=g("wind", "WIND_VZ_KMS", 0.0),
        T=g("wind", "WIND_T", 1.0e7),
        N_CM3=g("wind", "WIND_N_CM3", 1.0e-2),
        DUST_TO_GAS=g("wind", "WIND_DUST_TO_GAS", 1.0e-3),
    )

    return Config(run, grid, ambient, cloud1, cloud2,
                  turbulence, cooling, dust, solver, parallel, plot, files, wind)


# ------------------------ writer ------------------------

def write_settings_module(cfg: Config, path: str):
    """Generate detonations/settings.py from the parsed config."""
    p = Path(path)
    lines = []

    # Physical constants (cgs)
    lines += [
        "PC_CM = 3.085677581491367e18",
        "kB    = 1.3806485e-16",
        "mH    = 1.6726219e-24",
        "gamma = 5.0/3.0",
        "mu_mol= 2.33*mH",
        "MSUN  = 1.98847e33",
        "YR_S  = 365.0*24.0*3600.0",
    ]

    # Run controls
    lines += [
        f"DIM3D = {cfg.run.DIM3D!r}",
        f"INTEGRATED_MAPS = {cfg.run.INTEGRATED_MAPS!r}",
        f"COOLING_ON = {cfg.run.COOLING_ON!r}",
        f"DUST_ON = {cfg.run.DUST_ON!r}",
        f"TRACERS_ON = {cfg.run.TRACERS_ON!r}",
        f"TURB_ON = {cfg.run.TURB_ON!r}",
        f"TURB_SEED = {cfg.run.TURB_SEED!r}",
        f"VIEW_PLANE = {cfg.run.VIEW_PLANE!r}",
        f"T_END_YR = {cfg.run.T_END_YR!r}",
        f"CFL = {cfg.run.CFL!r}",
        f"SNAP_EVERY = {cfg.run.SNAP_EVERY!r}",
        f"TSERIES_EVERY = {cfg.run.TSERIES_EVERY!r}",
        f"RESTART = {cfg.run.RESTART!r}",
        f"RESTART_SNAP = {cfg.run.RESTART_SNAP!r}",
        f"SNAP_NEXT_OVERRIDE = {cfg.run.SNAP_NEXT_OVERRIDE!r}",
        f"CHK_EVERY_STEPS = {cfg.run.CHK_EVERY_STEPS!r}",
        f"CHK_EVERY_SECONDS = {cfg.run.CHK_EVERY_SECONDS!r}",
    ]

    # Grid
    lines += [
        f"Nx = {cfg.grid.Nx!r}",
        f"Ny = {cfg.grid.Ny!r}",
        f"Nz = {cfg.grid.Nz!r}",
        f"XMIN_PC = {cfg.grid.XMIN_PC!r}",
        f"XMAX_PC = {cfg.grid.XMAX_PC!r}",
        f"YMIN_PC = {cfg.grid.YMIN_PC!r}",
        f"YMAX_PC = {cfg.grid.YMAX_PC!r}",
        f"ZMIN_PC = {cfg.grid.ZMIN_PC!r}",
        f"ZMAX_PC = {cfg.grid.ZMAX_PC!r}",
    ]

    # Derived grid geometry (cm); REQUIRED by clouds.coordinates()
    lines += [
        "LX = (XMAX_PC - XMIN_PC) * PC_CM",
        "LY = (YMAX_PC - YMIN_PC) * PC_CM",
        "LZ = (ZMAX_PC - ZMIN_PC) * PC_CM",
        "dx = LX / max(Nx, 1)",
        "dy = LY / max(Ny, 1)",
        "dz = LZ / max(Nz, 1)",
        # Volume of a computational cell (use full 3D volume; clouds.py only reads dz if DIM3D)
        "VOL_CELL = dx * dy * (dz if DIM3D else 1.0)",
    ]

    # Ambient
    lines += [
        f"RHO_AMB = {cfg.ambient.RHO_AMB!r}",
        f"T_AMB = {cfg.ambient.T_AMB!r}",
        f"VX_AMB_KMS = {cfg.ambient.VX_AMB_KMS!r}",
        f"VY_AMB_KMS = {cfg.ambient.VY_AMB_KMS!r}",
        f"VZ_AMB_KMS = {cfg.ambient.VZ_AMB_KMS!r}",
        "P_AMB = RHO_AMB * kB * T_AMB / mu_mol",
    ]

    # Clouds (km/s -> cm/s copies too)
    lines += [
        f"C1_CENTER_PC = {cfg.cloud1.CENTER_PC!r}",
        f"C1_RADIUS_PC = {cfg.cloud1.RADIUS_PC!r}",
        f"C1_RHO = {cfg.cloud1.RHO!r}",
        f"C1_T = {cfg.cloud1.T!r}",
        f"C1_VX_KMS = {cfg.cloud1.VX_KMS!r}",
        f"C1_VY_KMS = {cfg.cloud1.VY_KMS!r}",
        f"C1_VZ_KMS = {cfg.cloud1.VZ_KMS!r}",
        f"C2_CENTER_PC = {cfg.cloud2.CENTER_PC!r}",
        f"C2_RADIUS_PC = {cfg.cloud2.RADIUS_PC!r}",
        f"C2_RHO = {cfg.cloud2.RHO!r}",
        f"C2_T = {cfg.cloud2.T!r}",
        f"C2_VX_KMS = {cfg.cloud2.VX_KMS!r}",
        f"C2_VY_KMS = {cfg.cloud2.VY_KMS!r}",
        f"C2_VZ_KMS = {cfg.cloud2.VZ_KMS!r}",
        f"C1_VX = {cfg.cloud1.VX_KMS*1.0e5!r}",
        f"C1_VY = {cfg.cloud1.VY_KMS*1.0e5!r}",
        f"C1_VZ = {cfg.cloud1.VZ_KMS*1.0e5!r}",
        f"C2_VX = {cfg.cloud2.VX_KMS*1.0e5!r}",
        f"C2_VY = {cfg.cloud2.VY_KMS*1.0e5!r}",
        f"C2_VZ = {cfg.cloud2.VZ_KMS*1.0e5!r}",
    ]

    # Wind-tunnel controls (km/s & cgs velocities exposed)
    lines += [
        f"WIND_ON = {cfg.wind.ON!r}",
        f"WIND_FACE = {cfg.wind.FACE!r}",
        f"WIND_VX_KMS = {cfg.wind.VX_KMS!r}",
        f"WIND_VY_KMS = {cfg.wind.VY_KMS!r}",
        f"WIND_VZ_KMS = {cfg.wind.VZ_KMS!r}",
        f"WIND_VX = {cfg.wind.VX_KMS*1.0e5!r}",
        f"WIND_VY = {cfg.wind.VY_KMS*1.0e5!r}",
        f"WIND_VZ = {cfg.wind.VZ_KMS*1.0e5!r}",
        f"WIND_T = {cfg.wind.T!r}",
        f"WIND_N_CM3 = {cfg.wind.N_CM3!r}",
        f"WIND_DUST_TO_GAS = {cfg.wind.DUST_TO_GAS!r}",
    ]

    # Turbulence + dust controls
    lines += [
        f"SIGMA_T_AMB_KMS = {cfg.turbulence.SIGMA_T_AMB_KMS!r}",
        f"SIGMA_T_C1_KMS = {cfg.turbulence.SIGMA_T_C1_KMS!r}",
        f"SIGMA_T_C2_KMS = {cfg.turbulence.SIGMA_T_C2_KMS!r}",
        f"D_INIT_AMBIENT = {cfg.dust.D_INIT_AMBIENT!r}",
        f"D_INIT_CLOUD1 = {cfg.dust.D_INIT_CLOUD1!r}",
        f"D_INIT_CLOUD2 = {cfg.dust.D_INIT_CLOUD2!r}",
        f"DUST_GAS_COOLING_ON = {cfg.dust.DUST_GAS_COOLING_ON!r}",
        f"DUST_BINS = {cfg.dust.DUST_BINS!r}",
        f"DUST_NBINS = {cfg.dust.DUST_NBINS!r}",
        f"DUST_A_MIN_UM = {cfg.dust.DUST_A_MIN_UM!r}",
        f"DUST_A_MAX_UM = {cfg.dust.DUST_A_MAX_UM!r}",
        f"LOGNORM_A0_UM = {cfg.dust.LOGNORM_A0_UM!r}",
        f"LOGNORM_SIGMA = {cfg.dust.LOGNORM_SIGMA!r}",
        f"MRN_Q = {cfg.dust.MRN_Q!r}",
    ]

    # Files / plotting
    basen = str(cfg.plot.BASEN)
    outdir = str(cfg.plot.OUTDIR)
    log_file_name = cfg.plot.LOG_FILE if cfg.plot.LOG_FILE else f"{basen}.log"

    lines += [
        f"OUTDIR = {outdir!r}",
        f"BASEN = {basen!r}",
        "LOGFILE = " + repr(f"{outdir}/{log_file_name}"),
        "MASS_FILE = " + repr(f"{outdir}/mass_history.dat"),
        "GROWZONE_FILE = " + repr(f"{outdir}/growth_zone_fractions.dat"),
        "CB_WIDTH_REL = 0.05",
        "CB_PAD_REL = 0.02",
        "CB_ASPECT = 50",
        "CB_FONT = 10",
        "FIG_DPI = 200",
        "PLOT_LIMITS = {'logrho': (-26.5, -20.5), 'logT': (2.0, 7.5)}",
        f"LAMBDA_TABLE_FILE = {cfg.files.LAMBDA_TABLE_FILE!r}",
    ]

    text = "# Auto-generated from detonations.par. Do not edit by hand.\n" + "\n".join(lines) + "\n"
    p.write_text(text)
