"""
Build script: morphologically realistic mouse M1 EEG head model
================================================================

Pipeline
--------
1. Download Allen Brain Atlas CCFv3 PLY meshes (brain, isocortex, M1/MOp)
2. Centre and orient to standard neuroimaging convention:
       X = A→P  (anterior negative, posterior positive, in mm from centroid)
       Y = D→V  (dorsal negative, ventral positive, in mm from centroid)
       Z = L→R  (left≈0, right positive, in mm from centroid)
3. Voxelise brain, morphologically dilate to synthesise skull + scalp shells
4. Extract skull / scalp surface meshes via marching cubes
5. Place 16 scalp EEG electrodes (bregma-referenced) via ray casting onto
   the real scalp mesh
6. Compute local four-sphere radii at M1 by ray-casting outward from centre
7. Build FourSphereVolumeConductor (lfpykit) with local radii → leadfield
8. Save everything to mice_M1_atlas.npz next to this script

Usage
-----
    python build_head_model.py          # full build (download + compute)
    python build_head_model.py --plot   # also show a preview figure

The generated .npz is ~30 MB and is loaded at runtime by
netpyne.analysis.plotEEG(head_model='mice_M1_atlas').
"""

import argparse
import os
import struct
import urllib.request

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.measure import marching_cubes
import trimesh
from lfpykit.eegmegcalc import FourSphereVolumeConductor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, '_cache')
OUT_FILE  = os.path.join(HERE, 'mice_M1_atlas.npz')

ALLEN_BASE = (
    'https://download.alleninstitute.org/informatics-archive/'
    'current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/ply/'
)
MESHES = {
    'brain':    ('997.ply',   'Whole brain'),
    'cortex':   ('315.ply',   'Isocortex'),
    'm1':       ('985.ply',   'Primary motor cortex (MOp)'),
}

# Allen CCFv3 bregma coordinates (µm) – widely used reference
# (anterior ~5400 µm, dorsal surface ~0 µm, midline ~5700 µm)
BREGMA_CCF_UM = np.array([5400., 0., 5700.])

# Electrode montage: (AP offset mm, ML offset mm, label)
# AP+ = rostral/anterior, ML+ = right lateral  (Paxinos convention)
ELECTRODE_GRID = [
    ( 2.5,  0.0, 'Fz' ),
    ( 2.5, -1.5, 'F1' ), ( 2.5,  1.5, 'F2' ),
    ( 1.5, -1.5, 'FC1'), ( 1.5,  1.5, 'FC2'),
    ( 0.0,  0.0, 'Cz' ),
    ( 0.0, -1.5, 'C1' ), ( 0.0,  1.5, 'C2' ),
    (-1.5, -1.5, 'CP1'), (-1.5,  1.5, 'CP2'),
    (-2.0,  0.0, 'Pz' ),
    (-2.0, -1.5, 'P1' ), (-2.0,  1.5, 'P2' ),
    (-3.0,  0.0, 'Oz' ),
    (-3.0, -1.5, 'O1' ), (-3.0,  1.5, 'O2' ),
]

# Head model geometry (mm)
SKULL_OFFSET_MM  = 0.60   # brain surface → skull outer surface
SCALP_OFFSET_MM  = 1.10   # brain surface → scalp outer surface
CORTICAL_DEPTH_MM = 0.50  # M1 dipole depth below pial surface

# Conductivities (S/m): brain, CSF, skull, scalp
SIGMAS = [0.33, 1.79, 0.008, 0.43]
CSF_THICKNESS_MM = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url, dest):
    if not os.path.exists(dest):
        print(f'  Downloading {os.path.basename(dest)} …')
        urllib.request.urlretrieve(url, dest)
    else:
        print(f'  {os.path.basename(dest)} already cached.')


def read_ply(path):
    """Read binary PLY with float32 x,y,z,(nx,ny,nz) vertices + int32 faces."""
    with open(path, 'rb') as f:
        n_verts = n_faces = 0
        n_props = 0
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            elif line.startswith('property float'):
                n_props += 1
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line == 'end_header':
                break
        verts = np.frombuffer(f.read(n_verts * n_props * 4),
                               dtype=np.float32).reshape(n_verts, n_props)[:, :3]
        faces = []
        for _ in range(n_faces):
            cnt = struct.unpack('B', f.read(1))[0]
            faces.append(struct.unpack(f'{cnt}i', f.read(cnt * 4)))
    return verts.astype(np.float64), np.array(faces, dtype=np.int32)


def voxelise_mesh(verts_mm, faces, voxel_mm=0.12):
    """
    Rasterise a closed mesh into a boolean voxel grid.

    Returns
    -------
    grid   : ndarray bool (nx, ny, nz)
    origin : ndarray (3,) mm — world coords of voxel [0,0,0] centre
    """
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)
    vox  = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_mm)
    # trimesh returns a VoxelGrid; matrix is a dense bool array
    origin = np.array(vox.bounds[0])   # corner of first voxel
    return vox.matrix.astype(bool), origin, float(voxel_mm)


def dilate_and_extract(grid, origin, voxel_mm, offset_mm):
    """Morphologically dilate a voxel grid by offset_mm, then marching-cubes."""
    radius_vox = max(1, round(offset_mm / voxel_mm))
    struct_el  = generate_binary_structure(3, 1)   # 6-connectivity ball approx
    dilated    = binary_dilation(grid,
                                  structure=struct_el,
                                  iterations=radius_vox)
    verts_vox, faces, _, _ = marching_cubes(dilated.astype(np.float32),
                                             level=0.5,
                                             spacing=(voxel_mm,) * 3)
    verts_mm = verts_vox + origin
    return verts_mm, faces.astype(np.int32)


def ray_outermost_hit(mesh_tm, origin_mm, direction):
    """
    Cast a ray from origin_mm and return the FARTHEST intersection (outer surface).

    When the origin is inside a closed mesh the farthest hit is the exit point,
    i.e. the actual outer radius in that direction.  Returns None on miss.
    """
    locs, _, _ = mesh_tm.ray.intersects_location(
        ray_origins=origin_mm.reshape(1, 3),
        ray_directions=direction.reshape(1, 3),
        multiple_hits=True,
    )
    if len(locs) == 0:
        return None
    dists = np.linalg.norm(locs - origin_mm, axis=1)
    return locs[np.argmax(dists)]   # outermost = largest distance from origin


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(plot=False):
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── 1. Download ──────────────────────────────────────────────────────────
    print('\n[1/7] Downloading Allen Brain Atlas CCFv3 PLY meshes …')
    paths = {}
    for key, (fname, _) in MESHES.items():
        dest = os.path.join(CACHE_DIR, fname)
        _download(ALLEN_BASE + fname, dest)
        paths[key] = dest

    # ── 2. Load & centre ────────────────────────────────────────────────────
    print('\n[2/7] Loading meshes and centring to brain centroid …')
    vb, fb   = read_ply(paths['brain'])    # whole brain  (µm Allen CCF)
    vc, fc   = read_ply(paths['cortex'])   # isocortex
    vm, fm   = read_ply(paths['m1'])       # primary motor cortex

    centroid_um = vb.mean(axis=0)          # brain centroid in Allen µm
    print(f'  Brain centroid (µm): {centroid_um}')

    # Convert to mm, centred at brain centroid
    # Allen CCF: dim0=AP (0=anterior), dim1=DV (0=dorsal), dim2=LR (0=left)
    to_mm = lambda v: (v - centroid_um) / 1e3

    vb_mm = to_mm(vb)
    vc_mm = to_mm(vc)
    vm_mm = to_mm(vm)

    bregma_mm = to_mm(BREGMA_CCF_UM)
    print(f'  Bregma in centred mm: {bregma_mm}')
    print(f'  Brain extent (mm):  AP {vb_mm[:,0].min():.1f}–{vb_mm[:,0].max():.1f}'
          f'  DV {vb_mm[:,1].min():.1f}–{vb_mm[:,1].max():.1f}'
          f'  LR {vb_mm[:,2].min():.1f}–{vb_mm[:,2].max():.1f}')

    # ── 3. Voxelise brain ───────────────────────────────────────────────────
    print('\n[3/7] Voxelising brain mesh (0.12 mm pitch) …')
    brain_grid, grid_origin, vox_mm = voxelise_mesh(vb_mm, fb, voxel_mm=0.12)
    print(f'  Voxel grid: {brain_grid.shape}  origin={grid_origin}')

    # ── 4. Dilate → skull + scalp surfaces ─────────────────────────────────
    print('\n[4/7] Synthesising skull and scalp via morphological dilation …')

    # Pad the voxel grid so dilation is never clipped by the grid boundary.
    # We need at least (SCALP_OFFSET_MM + 1 mm buffer) / vox_mm voxels of margin.
    n_pad = int(np.ceil((SCALP_OFFSET_MM + 1.0) / vox_mm))
    brain_grid_padded  = np.pad(brain_grid, n_pad, constant_values=False)
    grid_origin_padded = grid_origin - n_pad * vox_mm
    print(f'  Padded grid: {brain_grid_padded.shape}  (margin {n_pad} voxels = {n_pad*vox_mm:.2f} mm)')

    print('  Skull …')
    skull_verts_mm, skull_faces = dilate_and_extract(
        brain_grid_padded, grid_origin_padded, vox_mm, SKULL_OFFSET_MM)
    print(f'  Skull mesh: {len(skull_verts_mm)} verts, {len(skull_faces)} faces')

    print('  Scalp …')
    scalp_verts_mm, scalp_faces = dilate_and_extract(
        brain_grid_padded, grid_origin_padded, vox_mm, SCALP_OFFSET_MM)
    print(f'  Scalp mesh: {len(scalp_verts_mm)} verts, {len(scalp_faces)} faces')

    dv_range = (scalp_verts_mm[:, 1].min(), scalp_verts_mm[:, 1].max())
    print(f'  Scalp DV range: {dv_range[0]:.2f} to {dv_range[1]:.2f} mm')

    skull_tm  = trimesh.Trimesh(vertices=skull_verts_mm,  faces=skull_faces,  process=False)
    scalp_tm  = trimesh.Trimesh(vertices=scalp_verts_mm,  faces=scalp_faces,  process=False)
    skull_tm.fix_normals(); scalp_tm.fix_normals()

    # ── 5. Electrode placement via ray casting ──────────────────────────────
    print('\n[5/7] Placing electrodes on real scalp surface …')
    # Allen CCF axis mapping (centred mm):
    #   dim0 = AP: anterior = negative, posterior = positive
    #   dim1 = DV: dorsal = negative, ventral = positive
    #   dim2 = LR: left ≈ 0 (centred), right = positive
    #
    # Electrode in Paxinos (AP+= anterior, ML+= right):
    #   AP offset → subtract from bregma dim0 (more anterior = more negative dim0)
    #   ML offset → add to bregma dim2

    electrode_positions = []   # mm, centred
    electrode_labels    = []

    # For each electrode we find the most-dorsal (minimum dim1) scalp vertex
    # within a 0.5 mm radius of the target (AP, ML) position.
    # This is more robust than ray casting on potentially imperfect meshes.
    sv = scalp_verts_mm   # shorthand

    for ap_mm, ml_mm, lbl in ELECTRODE_GRID:
        x_ap = bregma_mm[0] - ap_mm     # anterior = subtract AP offset
        z_ml = bregma_mm[2] + ml_mm     # right    = add ML offset

        # Distance in the AP-LR plane to each scalp vertex
        dists_xz = np.sqrt((sv[:, 0] - x_ap)**2 + (sv[:, 2] - z_ml)**2)

        # Widen search if nothing found within 0.5 mm
        for tol in (0.5, 1.0, 1.5, 2.5):
            cands = np.where(dists_xz < tol)[0]
            if len(cands) > 0:
                break

        if len(cands) > 0:
            # Most dorsal = minimum dim1 (DV axis: dorsal = negative)
            best = cands[np.argmin(sv[cands, 1])]
            pos  = sv[best]
        else:
            print(f'  WARNING: {lbl} — no scalp vertex found, using projection')
            raw = np.array([x_ap, bregma_mm[1], z_ml])
            pos = raw / np.linalg.norm(raw) * np.linalg.norm(sv, axis=1).max()

        electrode_positions.append(pos)
        electrode_labels.append(lbl)
        print(f'  {lbl:4s}  ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) mm')

    r_electrodes_mm = np.array(electrode_positions).T   # (3, n_elec)

    # ── 6. M1 dipole position + local four-sphere radii ─────────────────────
    print('\n[6/7] Computing M1 dipole position and local radii …')

    # M1 centroid in centred mm (from actual Allen mesh)
    m1_centroid_mm = vm_mm.mean(axis=0)
    m1_dir         = m1_centroid_mm / np.linalg.norm(m1_centroid_mm)
    print(f'  M1 centroid: {m1_centroid_mm}  direction: {m1_dir}')

    # Build trimesh for brain surface to ray-cast local radii
    brain_tm = trimesh.Trimesh(vertices=vb_mm, faces=fb, process=False)
    brain_tm.fix_normals()

    centre = np.zeros(3)

    def local_radius(mesh_tm, direction, label):
        # Use vertex projection: max dot-product gives outermost extent
        # in the M1 direction.  This is robust regardless of mesh quality.
        r = float(np.max(mesh_tm.vertices @ direction))
        # Optionally refine with ray casting (farthest hit from centre)
        hit = ray_outermost_hit(mesh_tm, centre, direction)
        if hit is not None:
            r = max(r, float(np.linalg.norm(hit)))
        print(f'  {label}: r={r:.3f} mm')
        return r

    r_brain = local_radius(brain_tm,  m1_dir, 'r_brain')
    r_csf   = r_brain + CSF_THICKNESS_MM
    r_skull = local_radius(skull_tm,  m1_dir, 'r_skull')
    r_scalp = local_radius(scalp_tm,  m1_dir, 'r_scalp')

    dipole_pos_mm = m1_dir * (r_brain - CORTICAL_DEPTH_MM)
    print(f'  Dipole pos (mm): {dipole_pos_mm}')

    # ── 7. Leadfield via FourSphereVolumeConductor ──────────────────────────
    print('\n[7/7] Computing leadfield matrix …')

    # FourSphereVolumeConductor requires all electrodes to be outside the
    # scalp sphere AND farther from centre than the dipole.  The real scalp
    # is not a perfect sphere, so we project each electrode onto the scalp
    # sphere (preserve angular direction, normalise radius to r_scalp).
    # This is the standard approach for four-sphere EEG models.
    elec_norms  = np.linalg.norm(r_electrodes_mm, axis=0, keepdims=True)  # (1, n_elec)
    r_elec_projected_mm = r_electrodes_mm / elec_norms * r_scalp          # (3, n_elec)

    radii_um        = [r * 1e3 for r in (r_brain, r_csf, r_skull, r_scalp)]
    r_elec_um       = r_elec_projected_mm.T * 1e3   # (n_elec, 3) µm  ← (n,3) for lfpykit
    dipole_pos_um   = dipole_pos_mm * 1e3            # (3,) µm

    print(f'  Electrode norms after projection: min={np.linalg.norm(r_elec_um, axis=1).min():.1f} '
          f'max={np.linalg.norm(r_elec_um, axis=1).max():.1f} µm  (r_scalp={radii_um[3]:.1f} µm)')
    print(f'  Dipole norm: {np.linalg.norm(dipole_pos_um):.1f} µm  (r_brain={radii_um[0]:.1f} µm)')

    M_mat = FourSphereVolumeConductor(
        r_electrodes=r_elec_um,
        radii=radii_um,
        sigmas=SIGMAS,
    ).get_transformation_matrix(dipole_pos_um)  # (n_elec, 3)  [nA·µm → mV]

    print(f'  Leadfield shape: {M_mat.shape}')

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f'\nSaving model to {OUT_FILE} …')
    np.savez_compressed(
        OUT_FILE,
        # Forward model
        leadfield          = M_mat,                         # (n_elec, 3)
        dipole_pos_um      = dipole_pos_um,                 # (3,)
        radii_um           = np.array(radii_um),            # (4,)
        sigmas             = np.array(SIGMAS),              # (4,)
        # Electrodes
        electrode_positions_mm = r_electrodes_mm,           # (3, n_elec)
        electrode_labels       = np.array(electrode_labels),
        # Meshes for visualisation (centred mm)
        brain_verts_mm     = vb_mm.astype(np.float32),
        brain_faces        = fb,
        cortex_verts_mm    = vc_mm.astype(np.float32),
        cortex_faces       = fc,
        m1_verts_mm        = vm_mm.astype(np.float32),
        m1_faces           = fm,
        skull_verts_mm     = skull_verts_mm.astype(np.float32),
        skull_faces        = skull_faces,
        scalp_verts_mm     = scalp_verts_mm.astype(np.float32),
        scalp_faces        = scalp_faces,
        # Coordinate reference
        centroid_um        = centroid_um,
        bregma_mm          = bregma_mm,
    )
    print('Done.')

    if plot:
        _preview_plot(vb_mm, vc_mm, vm_mm, skull_verts_mm, scalp_verts_mm,
                      r_electrodes_mm, electrode_labels, dipole_pos_mm, bregma_mm)

    return OUT_FILE


# ---------------------------------------------------------------------------
# Preview visualisation
# ---------------------------------------------------------------------------

def _preview_plot(vb_mm, vc_mm, vm_mm, skull_mm, scalp_mm,
                  r_elec_mm, labels, dp_mm, bregma_mm):
    import matplotlib.pyplot as plt

    COL = dict(scalp='#f5cba7', skull='#f9e79f',
               brain='#aed6f1', cortex='#2e86c1', m1='#e74c3c',
               elec='gold')

    def pts_near(verts, axis, val, tol=0.3):
        return verts[np.abs(verts[:, axis] - val) < tol]

    # Slice planes through M1
    ap_slice = dp_mm[0]
    lr_slice = dp_mm[2]
    dv_slice = dp_mm[1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle('Mouse M1 atlas head model — cross-sections through M1',
                 fontsize=13, fontweight='bold')

    titles = ['Coronal (LR × DV @ M1 A-P)',
              'Sagittal (A-P × DV @ M1 L-R)',
              'Axial (A-P × LR @ M1 D-V)']
    combos = [(2, 1), (0, 1), (0, 2)]

    for ax, (xi, yi), title in zip(axes, combos, titles):
        slice_ax = {(2,1):0, (0,1):2, (0,2):1}[(xi, yi)]
        slice_val = [ap_slice, lr_slice, dv_slice][slice_ax]

        for verts, col, s, a in [
            (scalp_mm,  COL['scalp'],  1.0, 0.30),
            (skull_mm,  COL['skull'],  1.0, 0.40),
            (vb_mm,     COL['brain'],  0.8, 0.50),
            (vc_mm,     COL['cortex'], 1.5, 0.75),
            (vm_mm,     COL['m1'],     4.0, 1.00),
        ]:
            p = pts_near(verts, slice_ax, slice_val)
            if len(p):
                ax.scatter(p[:, xi], p[:, yi], c=col, s=s, alpha=a,
                            linewidths=0, rasterized=True)

        # Electrodes
        for i in range(r_elec_mm.shape[1]):
            ep = r_elec_mm[:, i]
            if abs(ep[slice_ax] - slice_val) < 0.6:
                ax.plot(ep[xi], ep[yi], 'o', ms=9, c=COL['elec'],
                         markeredgecolor='k', markeredgewidth=0.6, zorder=10)
                ax.annotate(labels[i], (ep[xi], ep[yi]),
                             fontsize=6, xytext=(0, 4),
                             textcoords='offset points', ha='center')

        # M1 dipole
        ax.plot(dp_mm[xi], dp_mm[yi], '*', ms=14, c='#e74c3c',
                 markeredgecolor='k', markeredgewidth=0.5, zorder=11)

        # Bregma (on appropriate views)
        if slice_ax != 0:
            ax.axvline(bregma_mm[xi], color='purple', lw=0.7, ls='--', alpha=0.5)

        ax.set_xlabel(['A-P', 'A-P', 'A-P', 'L-R', 'D-V', 'L-R'][xi*2] + ' (mm)')
        ax.set_ylabel(['A-P', 'A-P', 'A-P', 'L-R', 'D-V', 'D-V'][yi*2] + ' (mm)')
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal'); ax.set_facecolor('#e8f0f7')
        if yi == 1: ax.invert_yaxis()   # dorsal on top

    plt.tight_layout()
    out = os.path.join(HERE, 'mice_M1_atlas_preview.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Preview saved to {out}')
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true',
                        help='Show preview plot after building')
    args = parser.parse_args()
    build(plot=args.plot)
