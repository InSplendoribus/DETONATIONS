from ..hydro.solvers import prim_from_cons
import os
from ..settings import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import h5py

# ---------------- compact colorbar ----------------
def _inset_colorbar(fig, ax, im, label):
    bbox = ax.get_position()
    width = CB_WIDTH_REL * bbox.width
    pad   = CB_PAD_REL   * bbox.width
    cax = fig.add_axes(Bbox.from_extents(bbox.x1 + pad, bbox.y0,
                                         bbox.x1 + pad + width, bbox.y1))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label)
    return cb

# --------------- helpers -----------------
def _extent_1d(a0, a1, n):
    return np.linspace(a0, a1, n, endpoint=False) + 0.5*(a1-a0)/n

def get_slice_fields(U, plane, D_bins=None, fcld1=None, fcld2=None):
    rho,vx,vy,vz,P = prim_from_cons(U)
    n = rho/np.maximum(mu_mol,1e-30)
    T = np.maximum(P/(np.maximum(n,1e-30)*kB), 10.0)
    Dtot = None
    if (D_bins is not None):
        Dtot = np.sum(np.clip(D_bins,0,None), axis=-1)

    x = _extent_1d(XMIN_PC, XMAX_PC, Nx)
    y = _extent_1d(YMIN_PC, YMAX_PC, Ny)
    z = _extent_1d(ZMIN_PC, ZMAX_PC, Nz)

    if plane.lower()=="xy":
        k = Nz//2
        F = {"rho": rho[:,:,k], "T": T[:,:,k], "P": P[:,:,k],
             "vx": vx[:,:,k], "vy": vy[:,:,k], "vz": vz[:,:,k],
             "D": (Dtot[:,:,k] if Dtot is not None else None),
             "Dbins": (D_bins[:,:,k,:] if D_bins is not None else None),
             "f1": (fcld1[:,:,k] if fcld1 is not None else None),
             "f2": (fcld2[:,:,k] if fcld2 is not None else None)}
        extent = [x.min(), x.max(), y.min(), y.max()]
        axes   = ("X [pc]","Y [pc]")
        ftype  = "xy"
    elif plane.lower()=="xz":
        j = Ny//2
        F = {"rho": rho[:,j,:], "T": T[:,j,:], "P": P[:,j,:],
             "vx": vx[:,j,:], "vy": vy[:,j,:], "vz": vz[:,j,:],
             "D": (Dtot[:,j,:] if Dtot is not None else None),
             "Dbins": (D_bins[:,j,:,:] if D_bins is not None else None),
             "f1": (fcld1[:,j,:] if fcld1 is not None else None),
             "f2": (fcld2[:,j,:] if fcld2 is not None else None)}
        extent = [x.min(), x.max(), z.min(), z.max()]
        axes   = ("X [pc]","Z [pc]")
        ftype  = "xz"
    else: # "yz"
        i = Nx//2
        F = {"rho": rho[i,:,:], "T": T[i,:,:], "P": P[i,:,:],
             "vx": vx[i,:,:], "vy": vy[i,:,:], "vz": vz[i,:,:],
             "D": (Dtot[i,:,:] if Dtot is not None else None),
             "Dbins": (D_bins[i,:,:,:] if D_bins is not None else None),
             "f1": (fcld1[i,:,:] if fcld1 is not None else None),
             "f2": (fcld2[i,:,:] if fcld2 is not None else None)}
        extent = [y.min(), y.max(), z.min(), z.max()]
        axes   = ("Y [pc]","Z [pc]")
        ftype  = "yz"

    return F, extent, axes, ftype

# ------------------- unified 3D HDF5 writer -------------------
def write_h5_3d(U, D_bins, f1, f2, t_sec, step, snap_id, kind="std"):
    """
    kind: "std" -> 3D_{BASEN}_hdf5_{id}
          "plt" -> 3D_{BASEN}_hdf5_plt_{id}
          "chk" -> 3D_{BASEN}_hdf5_chk_{id}
    Includes conservative U for restart.
    """
    tag = BASEN
    if kind == "chk":
        fname = os.path.join(OUTDIR, f"3D_{tag}_hdf5_chk_{snap_id:04d}")
    elif kind == "plt":
        fname = os.path.join(OUTDIR, f"3D_{tag}_hdf5_plt_{snap_id:04d}")
    else:
        fname = os.path.join(OUTDIR, f"3D_{tag}_hdf5_{snap_id:04d}")

    rho,vx,vy,vz,P = prim_from_cons(U)
    n = rho/np.maximum(mu_mol,1e-30)
    T = np.maximum(P/(np.maximum(n,1e-30)*kB), 10.0)

    with h5py.File(fname, "w") as h:
        # meta
        h.create_dataset("time", data=np.array(t_sec))
        h.create_dataset("nstep", data=np.array(step, dtype=np.int64))
        h.create_dataset("snap", data=np.array(snap_id, dtype=np.int64))
        h.create_dataset("dimensionality", data=np.array(3 if DIM3D else 2, dtype=np.int32))
        h.create_dataset("bbox", data=np.array([[XMIN_PC,XMAX_PC],[YMIN_PC,YMAX_PC],[ZMIN_PC,ZMAX_PC]], dtype=np.float64))
        h.create_dataset("cell_sizes", data=np.array([dx,dy,dz], dtype=np.float64))

        # conservative state for restart
        h.create_dataset("U0", data=U[0])
        h.create_dataset("U1", data=U[1])
        h.create_dataset("U2", data=U[2])
        h.create_dataset("U3", data=U[3])
        h.create_dataset("U4", data=U[4])

        # derived
        h.create_dataset("dens", data=rho)
        h.create_dataset("pres", data=P)
        h.create_dataset("temp", data=T)
        h.create_dataset("velx", data=vx)
        h.create_dataset("vely", data=vy)
        h.create_dataset("velz", data=vz)

        # tracers
        if f1 is not None: h.create_dataset("fcld1", data=f1)
        if f2 is not None: h.create_dataset("fcld2", data=f2)

        # dust bins
        if D_bins is not None:
            NB = D_bins.shape[-1]
            for i in range(NB):
                h.create_dataset(f"dst{i}", data=D_bins[..., i])

    kind_label = {"std":"[hdf5]  ", "plt":"[plt]", "chk":"[chk]"}[kind]
    print(f"{kind_label} wrote {fname}")

# ---------------------------- PNG + ASCII (unchanged except basename usage) ----------------------------
def make_png(U, t_yr, snap_id, D_bins=None, fcld1=None, fcld2=None):
    F, extent, axes, ftype = get_slice_fields(U, VIEW_PLANE, D_bins, fcld1, fcld2)
    fig = plt.figure(figsize=(13.0, 8.0))
    gs  = fig.add_gridspec(2,3, width_ratios=[1,1,0.9], height_ratios=[1,1],
                           wspace=0.5, hspace=0.0005)

    ax  = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1], sharex=ax,  sharey=ax)
    ax3 = fig.add_subplot(gs[1,0], sharex=ax,  sharey=ax)
    ax4 = fig.add_subplot(gs[1,1], sharex=ax3, sharey=ax3)

    A = np.log10(np.clip(F["rho"],1e-40,None)).T
    im = ax.imshow(A, origin="lower", extent=extent,
                   vmin=PLOT_LIMITS["logrho"][0], vmax=PLOT_LIMITS["logrho"][1], cmap="jet", aspect="equal")
    _inset_colorbar(fig, ax, im, r"$\log\rho\,[\mathrm{g\,cm^{-3}}]$")

    B = np.log10(np.clip(F["T"],10.0,None)).T
    im2 = ax2.imshow(B, origin="lower", extent=extent,
                     vmin=PLOT_LIMITS["logT"][0], vmax=PLOT_LIMITS["logT"][1], cmap="jet", aspect="equal")
    _inset_colorbar(fig, ax2, im2, r"$\log\,T\,[\mathrm{K}]$")

    if F["D"] is not None:
        Dclip = np.clip(F["D"], 1.0e-49, None)
        im3 = ax3.imshow(np.log10(Dclip).T, origin="lower", extent=extent,
                         cmap="jet", aspect="equal", vmin=-4.0, vmax=-1.0)
        _inset_colorbar(fig, ax3, im3, r"$\log\,\mathcal{D}$")
    else:
        ax3.text(0.5,0.5,"Dust OFF", ha="center", va="center", transform=ax3.transAxes)

    vmag = np.sqrt(F["vx"]**2 + F["vy"]**2 + F["vz"]**2)/1.0e5  # km/s
    im4 = ax4.imshow(vmag.T, origin="lower", extent=extent, cmap="jet",
                     aspect="equal", vmin=-10.0, vmax=40.0)
    _inset_colorbar(fig, ax4, im4, r"$|{\bf v}|\,[\mathrm{km\,s^{-1}}]$")

    for Aax in (ax, ax2):
        Aax.set_xlabel(""); Aax.tick_params(labelbottom=False)
    for Aax in (ax2, ax4):
        Aax.set_ylabel(""); Aax.tick_params(labelleft=False)
    ax.set_ylabel(axes[1])
    ax3.set_ylabel(axes[1])
    ax3.set_xlabel(axes[0])
    ax4.set_xlabel(axes[0])

    fig.suptitle(f"t = {t_yr/1e3:.2f} kyr  |  {VIEW_PLANE.upper()} cut", y=0.985, fontsize=16)
    axm = fig.add_subplot(gs[:,2]); axm.axis("off")

    png = os.path.join(OUTDIR, f"snap_{BASEN}_{snap_id:04d}.png")
    fig.savefig(png, dpi=FIG_DPI); plt.close(fig)
    print(f"[snap] wrote {png}  t={t_yr/1e3:6.2f} kyr")

    # ASCII dump
    base = f"asc-{VIEW_PLANE.lower()}-{BASEN}_{snap_id:04d}"
    fname = os.path.join(OUTDIR, base)
    write_ascii_slice(fname, F, extent, VIEW_PLANE.lower())
    print(f"[ascii]  wrote {fname}")

    # Standard 3D HDF5 (as before)
    write_h5_3d(U, D_bins, fcld1, fcld2, t_yr*YR_S, step=-1, snap_id=snap_id, kind="std")

def write_ascii_slice(fname, F, extent, ftype):
    x0, x1, y0, y1 = extent
    nx, ny = F["rho"].shape
    xs = _extent_1d(x0, x1, nx)
    ys = _extent_1d(y0, y1, ny)
    Xp, Yp = np.meshgrid(xs, ys, indexing="ij")

    rho2 = F["rho"]; P2 = F["P"]; T2 = F["T"]
    vx2, vy2, vz2 = F["vx"], F["vy"], F["vz"]

    cols = [Xp.ravel(), Yp.ravel(),
            rho2.ravel(), P2.ravel(), T2.ravel(),
            vx2.ravel(), vy2.ravel(), vz2.ravel()]

    if (F.get("Dbins", None) is not None):
        for i in range(F["Dbins"].shape[-1]):
            cols.append( F["Dbins"][:,:,i].ravel() )

    M = np.vstack(cols).T
    with open(fname, "w") as f:
        f.write(f"# plane = {ftype}   shape=({nx},{ny})\n")
        headers = ["x","y","dens","pres","temp","velx","vely","velz"]
        if (F.get("Dbins", None) is not None):
            headers += [f"dst{i}" for i in range(F['Dbins'].shape[-1])]
        f.write("# " + "  ".join([f"{i+1:<12d}" for i in range(len(headers))]) + "\n")
        f.write("# " + "  ".join([f"{h:<12s}" for h in headers]) + "\n")
        np.savetxt(f, M, fmt="%.8e")
