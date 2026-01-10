import matplotlib
matplotlib.use("Agg")  # backend no interactivo, seguro en paralelo para guardar PNGs

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import os
from multiprocessing import Pool, cpu_count

#=========================
"""
Functions
"""
#=========================
def load_simulation_data(filepath):
    data = np.loadtxt(filepath)
    shape = (128, 128)
    return {
        'x':    data[:,0].reshape(shape),
        'z':    data[:,1].reshape(shape),
        'dens': data[:,2].reshape(shape),
        'pres': data[:,3].reshape(shape),
        'temp': data[:,4].reshape(shape),
        'velx': data[:,5].reshape(shape),
        'vely': data[:,6].reshape(shape),
        'velz': data[:,7].reshape(shape),
        'eint': data[:,8].reshape(shape),
        'ener': data[:,9].reshape(shape),
        'fcld': data[:,10].reshape(shape),
        'famb': data[:,11].reshape(shape),
        'mask': data[:,12].reshape(shape),
    }

def extract_time_in_kyr(filename):
    with open(filename, 'r') as f:
        for _ in range(15):
            line = f.readline()
            if line.startswith("#") and "time" in line:
                parts = line.split()
                try:
                    return float(parts[parts.index("time")+2]) / 3.1556926e+10
                except:
                    pass
    return np.nan

def detect_plane(filename):
    name = filename.lower()
    if "xy" in name:
        return 'xy'
    elif "xz" in name:
        return 'xz'
    else:
        return 'yz'

def compute_shock_and_sound(dens, pres, vx, vz, coordX, coordZ,
                            shock_dp_thr=0.5, sound_dp_thr=0.05):
    gamma = 5/3
    cs = np.sqrt(gamma * pres / dens)
    Mach = np.hypot(vx, vz) / cs

    # media local de presión (ventana 10×10)
    kernel = 10
    pm = np.empty_like(pres)
    for i in range(pres.shape[0]):
        for j in range(pres.shape[1]):
            i0, i1 = max(0, i-kernel//2), min(pres.shape[0], i+kernel//2+1)
            j0, j1 = max(0, j-kernel//2), min(pres.shape[1], j+kernel//2+1)
            pm[i,j] = pres[i0:i1, j0:j1].mean()
    dp = (pres - pm) / pm

    shock_mask = np.abs(dp) > shock_dp_thr
    sound_mask = np.abs(dp) > sound_dp_thr

    return Mach, shock_mask, sound_mask, np.abs(dp), dp

def compute_cloud_edge_mask(dens, fcld, kernel=10, edge_thr=0.5, min_rho_cl=1e-28):
    """
    Detecta bordes de la 'nube' buscando saltos relativos en rho_cl = dens*fcld
    respecto a su media local, pero:
      - Ignora celdas con rho_cl < min_rho_cl (no cuentan ni para la media ni para contorno).
      - Devuelve una máscara booleana para contornear (True = borde de nube).
      - También devuelve log10(rho_cl) para referencia si se quiere mostrar.
    """
    rho_cl = dens * fcld
    valid = rho_cl >= min_rho_cl
    rho_work = rho_cl.copy().astype(float)
    rho_work[~valid] = np.nan

    rm = np.empty_like(rho_work)
    for i in range(rho_work.shape[0]):
        i0 = max(0, i - kernel//2)
        i1 = min(rho_work.shape[0], i + kernel//2 + 1)
        for j in range(rho_work.shape[1]):
            j0 = max(0, j - kernel//2)
            j1 = min(rho_work.shape[1], j + kernel//2 + 1)
            window = rho_work[i0:i1, j0:j1]
            m = np.nanmean(window)
            rm[i, j] = m

    with np.errstate(invalid='ignore', divide='ignore'):
        dr = (rho_work - rm) / rm

    edge_mask = np.abs(dr) > edge_thr
    edge_mask &= valid
    edge_mask[~valid] = False

    log_rho_cl = np.log10(rho_cl + 1e-99)
    return edge_mask, log_rho_cl

#=========================
# Globals para workers (se rellenan en init_worker)
#=========================
_g = {
    "coordX": None, "coordZ": None, "xlim": None, "zlim": None,
    "nsx": 2, "nsy": 2, "method": "stream",
    "mom_times": None, "mom_vals": None,
    "cloud_times": None, "cloud_zmom": None,
    "scale": 1.0,#3.0856776e+18,
    "add_momentum_panel": False,
}

def init_worker(coordX, coordZ, xlim, zlim,
                mom_times, mom_vals, cloud_times, cloud_zmom,
                nsx, nsy, method, add_momentum_panel):
    _g["coordX"] = coordX
    _g["coordZ"] = coordZ
    _g["xlim"] = xlim
    _g["zlim"] = zlim
    _g["mom_times"] = mom_times
    _g["mom_vals"] = mom_vals
    _g["cloud_times"] = cloud_times
    _g["cloud_zmom"] = cloud_zmom
    _g["nsx"] = nsx
    _g["nsy"] = nsy
    _g["method"] = method
    _g["add_momentum_panel"] = add_momentum_panel

def process_file(filename):
    print(f"Processing {filename}")
    data = load_simulation_data(filename)
    time_kyr = extract_time_in_kyr(filename)
    plane = detect_plane(filename)  # no se usa, pero se conserva por compatibilidad

    coordX, coordZ = _g["coordX"], _g["coordZ"]
    xlim, zlim = _g["xlim"], _g["zlim"]
    nsx, nsy = _g["nsx"], _g["nsy"]
    method = _g["method"]
    add_momentum_panel = _g["add_momentum_panel"]

    dens, temp, pres = data['dens'], data['temp'], data['pres']
    vx, vy, vz = data['velx'], data['vely'], data['velz']
    fcld = data['fcld']

    # (se mantiene por compatibilidad, aunque ya no se usan sus máscaras)
    Mach, shock_mask, sound_mask, shock_indicator, dp = compute_shock_and_sound(
        dens, pres, vx, vz, coordX, coordZ
    )

    # Bordes de nube con umbral físico mínimo
    cloud_edge_mask, log_rho_cl = compute_cloud_edge_mask(
        dens, fcld, kernel=10, edge_thr=0.5, min_rho_cl=1e-28
    )

    # --- GridSpec: 2×2 mapas; si hay panel de momentum, se agrega la 3ª columna ---
    if add_momentum_panel:
        fig = plt.figure(figsize=(14,9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], wspace=0.6, hspace=0.35)
        ax11 = fig.add_subplot(gs[0,0])  # 1: densidad
        ax12 = fig.add_subplot(gs[0,1])  # 2: temperatura
        ax21 = fig.add_subplot(gs[1,0])  # 3: log10(pres)
        ax22 = fig.add_subplot(gs[1,1])  # 4: Vz
    else:
        fig = plt.figure(figsize=(12,8))
        gs = fig.add_gridspec(2, 2, wspace=0.6, hspace=0.35)
        ax11 = fig.add_subplot(gs[0,0])  # 1: densidad
        ax12 = fig.add_subplot(gs[0,1])  # 2: temperatura
        ax21 = fig.add_subplot(gs[1,0])  # 3: log10(pres)
        ax22 = fig.add_subplot(gs[1,1])  # 4: Vz

    # 1: densidad
    c0 = ax11.pcolormesh(coordX, coordZ, np.log10(dens),
                         cmap='jet', shading='gouraud',
                         vmin=-24.0, vmax=-19.0)
    fig.colorbar(c0, ax=ax11, fraction=0.046, pad=0.04,
                 label=r'$\log(\rho\,[g/cm^3])$')

    # 2: temperatura
    c1 = ax12.pcolormesh(coordX, coordZ, np.log10(temp),
                         cmap='jet', shading='gouraud',
                         vmin=2., vmax=8.0)
    fig.colorbar(c1, ax=ax12, fraction=0.046, pad=0.04,
                 label=r'$\log(T/\mathrm{K})$')

    # 3: presión (mantén los límites dados en el código 1 original: -10.5..-9.5)
    c2 = ax21.pcolormesh(coordX, coordZ, np.log10(pres),
                         cmap='jet', shading='auto',
                         vmin=-10.0, vmax=-7.0)
    fig.colorbar(c2, ax=ax21, fraction=0.046, pad=0.04,
                 label=r'$\log(p\;[\mathrm{dyn\ cm^{-2}}])$')

    # 4: Vz
    c3 = ax22.pcolormesh(coordX, coordZ, vx/1e5,
                         cmap='jet_r', shading='auto',
                         vmin=0, vmax=800)
    fig.colorbar(c3, ax=ax22, fraction=0.046, pad=0.04,
                 label=r'$V_z\;[\mathrm{km\,s^{-1}}]$')
    # ~ if method == 'quiver':
        # ~ ax22.quiver(coordX[::2,::2], coordZ[::2,::2],
                    # ~ vz[::2,::2], vx[::2,::2],
                    # ~ color='w', scale=1e6)
    # ~ else:
        # ~ ax22.streamplot(coordX[::nsy, ::nsx].T,
                        # ~ coordZ[::nsy, ::nsx].T,
                        # ~ vx[::nsy, ::nsx].T,
                        # ~ vz[::nsy, ::nsx].T,
                        # ~ density=1, color='w')

    # Contornos de bordes de nube en los 4 paneles
    for ax in (ax11, ax12, ax21, ax22):
        ax.contour(coordX, coordZ, cloud_edge_mask, levels=[0.5],
                   colors='black', linewidths=1.0)

    # ejes comunes
    for ax in (ax11, ax12, ax21, ax22):
        ax.axvline(0, color='white', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='white', linestyle='--', linewidth=0.8)
        ax.set_aspect('equal')
        ax.set_xlim(_g["xlim"])
        ax.set_ylim(_g["zlim"])
        ax.set_xlabel('X [pc]')
        ax.set_ylabel('Z [pc]')

    # 7: momentum panel a la derecha (solo si se solicita)
    if add_momentum_panel:
        ax_mom = fig.add_subplot(gs[:, 2])
        mom_times   = _g["mom_times"]
        mom_vals    = _g["mom_vals"]
        cloud_times = _g["cloud_times"]
        cloud_zmom  = _g["cloud_zmom"]

        mask = mom_times <= time_kyr
        ax_mom.plot(mom_times[mask], mom_vals[mask]    / 1e42, label='total momentum')
        cmask = cloud_times <= time_kyr
        ax_mom.plot(cloud_times[cmask], cloud_zmom[cmask] / 1e42, label='cloud z-momentum')
        ax_mom.set_xlabel('t [kyr]')
        ax_mom.set_ylabel(r'$\mathrm{momentum}\,[10^{42}\,\mathrm{g\,cm/s}]$')
        ax_mom.set_title('Momentum vs Time')
        ax_mom.set_xlim(mom_times.min(), mom_times.max())
        ax_mom.set_ylim(-0, 1.05)
        ax_mom.legend()

    fig.suptitle(f"t = {time_kyr:.2f} kyr")
    fig.tight_layout(rect=[0,0,1,0.96])
    out = f"images/quiver_{os.path.basename(filename).split('_')[-1]}.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Figure saved: {out}")
    return out

#=========================
# Main: preparar grids y lanzar pool
#=========================
if __name__ == "__main__":
    pattern = os.path.expanduser("~/AI/detonations/snaps_ccc/asc-xy-clouds_*")
    file_list = sorted(glob.glob(pattern))
    # ~ file_list = sorted(glob.glob("~/AI/detonations/snaps_ccc/asc-xy-clouds_*"))
    nsx = nsy = 2
    method = "stream"
    scale = 1.0 #3.0856776e+18

    # Ajusta esta bandera: si es False no se leen/esperan archivos de momentum y no se dibuja el panel.
    add_momentum_panel = False

    if len(file_list) == 0:
        raise SystemExit("No se encontraron archivos 'asc-xz-clouds_*'.")

    # Precalcular coordX/coordZ y límites a partir del primer archivo
    first = file_list[0]
    d0 = load_simulation_data(first)
    coordX = d0['x'] / scale
    coordZ = d0['z'] / scale
    x0, x1 = coordX.min(), coordX.max()
    z0, z1 = coordZ.min(), coordZ.max()
    cx, cz = 0.5*(x0+x1), 0.5*(z0+z1)
    dx, dz = 0.5*(x1-x0), 0.5*(z1-z0)
    xlim = (cx-dx, cx+dx)
    zlim = (cz-dz, cz+dz)

    # Cargar datos de momentum solo si se pide el panel
    if add_momentum_panel:
        # Si usas otros nombres (p.ej., 'momentum.dat' / 'clouds.log'), cámbialos aquí.
        mom_data    = np.loadtxt('momentum_magnitude.dat', comments='#')
        mom_times   = mom_data[:,0]
        mom_vals    = mom_data[:,1]
        cloud_data  = np.loadtxt('clouds.dat', comments='#')
        cloud_times = cloud_data[:,0] / 3.1556926e+10
        cloud_zmom  = -cloud_data[:,4]
    else:
        mom_times = mom_vals = cloud_times = cloud_zmom = None

    # Crear carpeta de salida si no existe
    os.makedirs("images", exist_ok=True)

    # Lanzar pool usando todos los CPUs disponibles
    nproc = cpu_count() or 1
    with Pool(processes=nproc, initializer=init_worker,
              initargs=(coordX, coordZ, xlim, zlim,
                        mom_times, mom_vals, cloud_times, cloud_zmom,
                        nsx, nsy, method, add_momentum_panel)) as pool:
        # map paralelo sobre los archivos
        for _ in pool.imap_unordered(process_file, file_list, chunksize=1):
            pass
