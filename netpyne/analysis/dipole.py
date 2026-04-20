"""
Module for analyzing and plotting LFP-related results

"""

import math
import os

basestring = str

from netpyne import __gui__

if __gui__:
    import matplotlib.pyplot as plt
    from matplotlib import mlab
import numpy as np
from numbers import Number


def plotDipole(showCell=None, showPop=None, timeRange=None, dpi=300, figSize=(6, 6), showFig=True, saveFig=True):
    from .. import sim

    try:
        if showCell:
            p = sim.allSimData['dipoleCells'][showCell]
        elif showPop:
            p = sim.allSimData['dipolePops'][showPop]
        else:
            p = sim.allSimData['dipoleSum']

        # if list (as a result of side-effect of some of save-load operations), make sure to convert to np.array
        if isinstance(p, list):
            p = np.array(p)

        p = p / 1000.0  # convert from nA to uA
    except:
        print('Unable to collect dipole information...')

    if timeRange is None:
        timeRange = [0, sim.cfg.duration]

    timeSteps = [int(timeRange[0] / sim.cfg.recordStep), int(timeRange[1] / sim.cfg.recordStep)]

    # current dipole moment
    plt.figure(figsize=figSize)
    plt.plot(np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep), np.array(p)[timeSteps[0] : timeSteps[1]])
    # plt.legend([r'$P_x$ (mA um)', r'$P_y$ (mA um)', r'$P_z$ (mA um)'])
    plt.legend([r'$P_x$', r'$P_y$', r'$P_z$'])
    plt.ylabel(r'$\mathbf{P}(t)$ ($\mu$A $\mu$m)')
    plt.xlabel('$t$ (ms)')
    ax = plt.gca()
    ax.grid(False)

    # save figure
    if saveFig:
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = sim.cfg.filename + '_dipole.png'
        try:
            plt.savefig(filename, dpi=dpi)
        except:
            plt.savefig('dipole_fig.png', dpi=dpi)

    # display figure
    if showFig is True:
        plt.show()


# ---------------------------------------------------------------------------
# Helpers for mice_M1 head model
# ---------------------------------------------------------------------------

def _mice_m1_setup():
    """
    Four-sphere volume conductor for a mouse M1 cortical column.

    Coordinate convention
    ---------------------
    X = mediolateral  (right positive)
    Y = anteroposterior (rostral positive)
    Z = dorsoventral  (dorsal positive)
    Origin at the geometric centre of the brain sphere.
    Bregma lies at approximately (0, 0, r_brain) on the brain-sphere surface.

    All lengths are in µm to match FourSphereVolumeConductor expectations
    (returns a matrix mapping nA·µm → mV).
    Conductivities are in S/m.

    Returns
    -------
    M_mat        : ndarray (n_elec, 3)   transformation matrix [nA·µm → mV]
    dipole_pos   : ndarray (3,)          M1 source position [µm]
    r_electrodes : ndarray (3, n_elec)   scalp electrode positions [µm]
    labels       : list[str]             electrode labels
    radii        : list[float]           [r_brain, r_csf, r_skull, r_scalp] in µm
    """
    from lfpykit.eegmegcalc import FourSphereVolumeConductor

    # Adult C57BL/6 mouse head — approximate values from the literature
    radii  = [7000., 7100., 7600., 8100.]    # µm: brain, CSF, skull, scalp
    sigmas = [0.33,  1.79,  0.008, 0.43]     # S/m: brain, CSF, skull, scalp

    r_brain, _, _, r_scalp = radii

    # M1 dipole: AP +1.5 mm, ML +1.0 mm from bregma, 0.5 mm cortical depth
    raw_m1     = np.array([1000., 1500., r_brain])          # µm (ML, AP, start at bregma apex)
    depth_um   = 500.                                         # µm cortical depth
    dipole_pos = raw_m1 / np.linalg.norm(raw_m1) * (r_brain - depth_um)

    # Scalp electrode grid — bregma-relative positions (ML, AP in µm)
    # projected onto the scalp sphere surface
    bregma_grid = [
        (    0.,  2500., 'Fz' ),
        (-1500.,  2500., 'F1' ), ( 1500.,  2500., 'F2' ),
        (-1500.,  1500., 'FC1'), ( 1500.,  1500., 'FC2'),
        (    0.,     0., 'Cz' ),
        (-1500.,     0., 'C1' ), ( 1500.,     0., 'C2' ),
        (-1500., -1500., 'CP1'), ( 1500., -1500., 'CP2'),
        (    0., -2000., 'Pz' ),
        (-1500., -2000., 'P1' ), ( 1500., -2000., 'P2' ),
        (    0., -3000., 'Oz' ),
        (-1500., -3000., 'O1' ), ( 1500., -3000., 'O2' ),
    ]
    elec_pos, labels = [], []
    for ml, ap, lbl in bregma_grid:
        raw = np.array([ml, ap, r_scalp])
        elec_pos.append(raw / np.linalg.norm(raw) * r_scalp)
        labels.append(lbl)

    r_electrodes = np.array(elec_pos).T   # shape (3, n_elec) — used for plots

    # FourSphereVolumeConductor.get_transformation_matrix uses axis=1 to norm
    # electrodes, so it expects shape (n_elec, 3), not (3, n_elec).
    # Project each electrode onto the scalp sphere (preserve direction, fix radius)
    # so the constraint r_elec > r_dipole > 0 is guaranteed for all electrodes.
    r_elec_dirs = r_electrodes / np.linalg.norm(r_electrodes, axis=0, keepdims=True)
    r_elec_on_sphere = (r_elec_dirs * radii[3]).T  # (n_elec, 3) µm

    M_mat = FourSphereVolumeConductor(
        r_electrodes=r_elec_on_sphere,
        radii=radii,
        sigmas=sigmas,
    ).get_transformation_matrix(dipole_pos)  # shape (n_elec, 3)

    return M_mat, dipole_pos, r_electrodes, labels, radii


def _rotate_dipole_to_normal(p, target_normal, orig_ax_vec=None):
    """
    Rotate dipole moment time-series so that `orig_ax_vec` aligns with
    `target_normal` (the cortical surface normal at the dipole location).

    Implements the same Rodrigues-rotation approach as
    NYHeadModel.rotate_dipole_to_surface_normal.

    Parameters
    ----------
    p            : ndarray (3, n_t)  dipole time-series [nA·µm]
    target_normal: ndarray (3,)      desired orientation (will be normalised)
    orig_ax_vec  : ndarray (3,)      cortical-column axis in the simulation
                                     coordinate frame.  Defaults to [0, 0, 1]
                                     (Z axis = apical-dendrite / depth axis).

    Returns
    -------
    ndarray (3, n_t)  rotated dipole moment
    """
    if orig_ax_vec is None:
        orig_ax_vec = np.array([0., 0., 1.])

    n = np.asarray(target_normal, dtype=float)
    n /= np.linalg.norm(n)
    u = np.asarray(orig_ax_vec, dtype=float)
    u /= np.linalg.norm(u)

    phi     = math.acos(np.clip(np.dot(u, n), -1., 1.))
    rot_axis = np.cross(u, n)
    axis_len = np.linalg.norm(rot_axis)
    if axis_len < 1e-9:          # already aligned (or anti-parallel)
        return p
    rot_axis /= axis_len

    x_, y_, z_ = rot_axis
    cos_th, sin_th = np.cos(phi), np.sin(phi)
    R = np.array([
        [cos_th + x_**2*(1-cos_th),     x_*y_*(1-cos_th) - z_*sin_th,  x_*z_*(1-cos_th) + y_*sin_th],
        [y_*x_*(1-cos_th) + z_*sin_th,  cos_th + y_**2*(1-cos_th),     y_*z_*(1-cos_th) - x_*sin_th],
        [z_*x_*(1-cos_th) - y_*sin_th,  z_*y_*(1-cos_th) + x_*sin_th,  cos_th + z_**2*(1-cos_th)  ],
    ])
    return R @ p


# ---------------------------------------------------------------------------
# Main plotting functions
# ---------------------------------------------------------------------------

def plotEEG(
    showCell=None,
    showPop=None,
    timeRange=None,
    dipole_location='parietal_lobe',
    head_model='NYHead',
    orig_ax_vec=None,
    dpi=300,
    figSize=(19, 10),
    showFig=True,
    saveFig=True,
):
    """
    Plot EEG signals derived from the simulated current dipole moment.

    Parameters
    ----------
    showCell        : int or None
        GID of a specific cell whose dipole to use (requires saveDipoleCells).
    showPop         : str or None
        Population label whose aggregate dipole to use (requires saveDipolePops).
    timeRange       : list [t_start, t_end] in ms.  Defaults to full simulation.
    dipole_location : str
        Predefined dipole location for the NYHead model (e.g. 'parietal_lobe',
        'motor_cortex').  Ignored when head_model='mice_M1'.
    head_model      : {'NYHead', 'mice_M1'}
        * 'NYHead'  — human New York Head model via lfpykit (default, existing
                      behaviour).
        * 'mice_M1' — mouse primary motor cortex modelled with a four-sphere
                      volume conductor (brain/CSF/skull/scalp) scaled to adult
                      C57BL/6 dimensions.  Electrodes match a standard dorsal
                      mouse EEG montage referenced to bregma.
    orig_ax_vec     : array-like (3,) or None
        Axis in the simulation coordinate frame that corresponds to the
        apical-dendrite / cortical-depth direction.  Used to rotate the dipole
        onto the cortical surface normal before applying the forward model.
        Defaults to [0, 0, 1] (Z axis) for both head models.
    dpi, figSize, showFig, saveFig : standard figure options.
    """
    from .. import sim

    # ------------------------------------------------------------------ data
    if showCell:
        p = sim.allSimData['dipoleCells'][showCell]
    elif showPop:
        p = sim.allSimData['dipolePops'][showPop]
    else:
        p = sim.allSimData['dipoleSum']

    if isinstance(p, list):
        p = np.array(p)

    if timeRange is None:
        timeRange = [0, sim.cfg.duration]

    t0 = int(timeRange[0] / sim.cfg.recordStep)
    t1 = int(timeRange[1] / sim.cfg.recordStep)

    p = np.array(p).T[:, t0:t1]   # (3, n_steps)
    t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)

    plt.close("all")

    # ==========================================================================
    # NYHead model — existing behaviour, unchanged
    # ==========================================================================
    if head_model == 'NYHead':
        from lfpykit.eegmegcalc import NYHeadModel

        nyhead = NYHeadModel(nyhead_file=os.getenv('NP_LFPYKIT_HEAD_FILE', None))

        # dipole_location = 'parietal_lobe'  # predefined location from NYHead class
        nyhead.set_dipole_pos(dipole_location)
        M = nyhead.get_transformation_matrix()

        # We rotate current dipole moment to be oriented along the normal vector of cortex
        p = nyhead.rotate_dipole_to_surface_normal(p)
        eeg = M @ p * 1e9  # [mV] -> [pV] unit conversion

        # plot EEG data
        x_lim = [-100, 100]
        y_lim = [-130, 100]
        z_lim = [-160, 120]

        fig = plt.figure(figsize=[19, 10])
        fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.17, wspace=0.3, left=0.1, right=0.99)
        ax1 = fig.add_subplot(245, aspect=1, xlabel="x (mm)", ylabel='y (mm)', xlim=x_lim, ylim=y_lim)
        ax2 = fig.add_subplot(246, aspect=1, xlabel="x (mm)", ylabel='z (mm)', xlim=x_lim, ylim=z_lim)
        ax3 = fig.add_subplot(247, aspect=1, xlabel="y (mm)", ylabel='z (mm)', xlim=y_lim, ylim=z_lim)
        ax_eeg = fig.add_subplot(244, xlabel="Time (ms)", ylabel='pV', title='EEG at all electrodes')

        ax_cdm = fig.add_subplot(248, xlabel="Time (ms)", ylabel='nA$\cdot \mu$m', title='Current dipole moment')
        dist, closest_elec_idx = nyhead.find_closest_electrode()
        print("Closest electrode to dipole: {:1.2f} mm".format(dist))

        max_elec_idx = np.argmax(np.std(eeg, axis=1))
        time_idx = np.argmax(np.abs(eeg[max_elec_idx]))
        max_eeg = np.max(np.abs(eeg[:, time_idx]))
        max_eeg_idx = np.argmax(np.abs(eeg[:, time_idx]))

        max_eeg_pos = nyhead.elecs[:3, max_eeg_idx]
        fig.text(0.01, 0.25, "Cortex", va='center', rotation=90, fontsize=22)
        fig.text(
            0.03,
            0.25,
            "Dipole pos: {:1.1f}, {:1.1f}, {:1.1f}\nDipole moment: {:1.2f} {:1.2f} {:1.2f}".format(
                nyhead.dipole_pos[0],
                nyhead.dipole_pos[1],
                nyhead.dipole_pos[2],
                p[0, time_idx],
                p[1, time_idx],
                p[2, time_idx],
            ),
            va='center',
            rotation=90,
            fontsize=14,
        )

        fig.text(0.01, 0.75, "EEG", va='center', rotation=90, fontsize=22)
        fig.text(
            0.03,
            0.75,
            "Max: {:1.2f} pV at idx {}\n({:1.1f}, {:1.1f} {:1.1f})".format(
                max_eeg, max_eeg_idx, max_eeg_pos[0], max_eeg_pos[1], max_eeg_pos[2]
            ),
            va='center',
            rotation=90,
            fontsize=14,
        )

        ax7 = fig.add_subplot(241, aspect=1, xlabel="x (mm)", ylabel='y (mm)', xlim=x_lim, ylim=y_lim)
        ax8 = fig.add_subplot(242, aspect=1, xlabel="x (mm)", ylabel='z (mm)', xlim=x_lim, ylim=z_lim)
        ax9 = fig.add_subplot(243, aspect=1, xlabel="y (mm)", ylabel='z (mm)', xlim=y_lim, ylim=z_lim)

        ax_cdm.plot(t, p[2, :], 'k')
        [ax_eeg.plot(t, eeg[idx, :], c='gray') for idx in range(eeg.shape[0])]
        ax_eeg.plot(t, eeg[closest_elec_idx, :], c='green', lw=2)

        vmax = np.max(np.abs(eeg[:, time_idx]))
        cmap = lambda v: plt.cm.bwr((v + vmax) / (2 * vmax))

        threshold = 2

        xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] - nyhead.dipole_pos[1]) < threshold)[0]
        xy_plane_idxs = np.where(np.abs(nyhead.cortex[2, :] - nyhead.dipole_pos[2]) < threshold)[0]
        yz_plane_idxs = np.where(np.abs(nyhead.cortex[0, :] - nyhead.dipole_pos[0]) < threshold)[0]

        ax1.scatter(nyhead.cortex[0, xy_plane_idxs], nyhead.cortex[1, xy_plane_idxs], s=5)
        ax2.scatter(nyhead.cortex[0, xz_plane_idxs], nyhead.cortex[2, xz_plane_idxs], s=5)
        ax3.scatter(nyhead.cortex[1, yz_plane_idxs], nyhead.cortex[2, yz_plane_idxs], s=5)

        for idx in range(eeg.shape[0]):
            c = cmap(eeg[idx, time_idx])
            ax7.plot(nyhead.elecs[0, idx], nyhead.elecs[1, idx], 'o', ms=10, c=c, zorder=nyhead.elecs[2, idx])
            ax8.plot(nyhead.elecs[0, idx], nyhead.elecs[2, idx], 'o', ms=10, c=c, zorder=nyhead.elecs[1, idx])
            ax9.plot(nyhead.elecs[1, idx], nyhead.elecs[2, idx], 'o', ms=10, c=c, zorder=-nyhead.elecs[0, idx])

        img = ax3.imshow([[], []], origin="lower", vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
        plt.colorbar(img, ax=ax9, shrink=0.5)

        ax1.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=12, color='orange', zorder=1000)
        ax2.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[2], '*', ms=12, color='orange', zorder=1000)
        ax3.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[2], '*', ms=12, color='orange', zorder=1000)

        ax7.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=15, color='orange', zorder=1000)
        ax8.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[2], '*', ms=15, color='orange', zorder=1000)
        ax9.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[2], '*', ms=15, color='orange', zorder=1000)

    # ==========================================================================
    # Mouse M1 — four-sphere head model
    # ==========================================================================
    elif head_model == 'mice_M1':
        M_mat, dipole_pos, r_electrodes, electrode_labels, radii = _mice_m1_setup()

        # Rotate simulated dipole so it points along the M1 cortical surface normal.
        # For a spherical head model the surface normal equals the radial direction.
        surface_normal = dipole_pos / np.linalg.norm(dipole_pos)
        p = _rotate_dipole_to_normal(p, surface_normal, orig_ax_vec=orig_ax_vec)

        eeg = M_mat @ p * 1e9  # nA·µm → mV → pV

        # Closest scalp electrode to M1 (smallest solid angle)
        dipole_dir = dipole_pos / np.linalg.norm(dipole_pos)
        cos_angles  = np.array([
            np.dot(r_electrodes[:, i] / np.linalg.norm(r_electrodes[:, i]), dipole_dir)
            for i in range(r_electrodes.shape[1])
        ])
        closest_elec_idx = int(np.argmax(cos_angles))
        arc_mm = math.acos(np.clip(cos_angles[closest_elec_idx], -1., 1.)) * radii[3] / 1e3
        print("Closest electrode to M1 dipole: {} ({:.2f} mm arc distance on scalp)".format(
            electrode_labels[closest_elec_idx], arc_mm))

        # Peak EEG time index (electrode with highest variance)
        max_elec_idx = np.argmax(np.std(eeg, axis=1))
        time_idx     = np.argmax(np.abs(eeg[max_elec_idx]))
        vmax         = np.max(np.abs(eeg[:, time_idx])) or 1.
        cmap         = lambda v: plt.cm.bwr((v + vmax) / (2 * vmax))

        # Convert µm → mm for all geometry to be plotted
        dp_mm = dipole_pos   / 1e3
        re_mm = r_electrodes / 1e3
        r_mm  = [r / 1e3 for r in radii]
        lim   = r_mm[3] * 1.25

        theta = np.linspace(0, 2 * np.pi, 360)

        fig = plt.figure(figsize=figSize)
        fig.suptitle("Mouse M1 EEG — four-sphere head model", fontsize=14, y=0.98)
        fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.40, wspace=0.40,
                            left=0.08, right=0.97)

        # ------ top row: EEG amplitude maps ------
        ax_xy_eeg = fig.add_subplot(241, aspect='equal',
                                     xlabel='ML (mm)', ylabel='AP (mm)',
                                     title='EEG — axial (top-down)',
                                     xlim=(-lim, lim), ylim=(-lim, lim))
        ax_xz_eeg = fig.add_subplot(242, aspect='equal',
                                     xlabel='ML (mm)', ylabel='DV (mm)',
                                     title='EEG — coronal',
                                     xlim=(-lim, lim), ylim=(-lim, lim))
        ax_yz_eeg = fig.add_subplot(243, aspect='equal',
                                     xlabel='AP (mm)', ylabel='DV (mm)',
                                     title='EEG — sagittal',
                                     xlim=(-lim, lim), ylim=(-lim, lim))
        ax_eeg    = fig.add_subplot(244,
                                     xlabel='Time (ms)', ylabel='pV',
                                     title='EEG traces')

        # Scalp outline on EEG panels
        for ax in (ax_xy_eeg, ax_xz_eeg, ax_yz_eeg):
            ax.fill(r_mm[3] * np.cos(theta), r_mm[3] * np.sin(theta),
                    color='#ececec', zorder=0)
            ax.plot(r_mm[3] * np.cos(theta), r_mm[3] * np.sin(theta),
                    'k-', lw=0.8, zorder=1)

        # Electrode markers coloured by EEG amplitude at peak time
        for idx in range(r_electrodes.shape[1]):
            c  = cmap(eeg[idx, time_idx])
            lw = 2.0 if idx == closest_elec_idx else 0.4
            ax_xy_eeg.plot(re_mm[0, idx], re_mm[1, idx], 'o', ms=9, c=c,
                            markeredgecolor='k', markeredgewidth=lw,
                            zorder=2 + re_mm[2, idx])
            ax_xz_eeg.plot(re_mm[0, idx], re_mm[2, idx], 'o', ms=9, c=c,
                            markeredgecolor='k', markeredgewidth=lw,
                            zorder=2 + re_mm[1, idx])
            ax_yz_eeg.plot(re_mm[1, idx], re_mm[2, idx], 'o', ms=9, c=c,
                            markeredgecolor='k', markeredgewidth=lw,
                            zorder=2 - re_mm[0, idx])
            ax_xy_eeg.annotate(electrode_labels[idx],
                                xy=(re_mm[0, idx], re_mm[1, idx]),
                                fontsize=5.5, ha='center', va='bottom', zorder=10)

        # Mark M1 dipole projection on each panel
        ax_xy_eeg.plot(dp_mm[0], dp_mm[1], '*', ms=13, color='orange', zorder=100,
                        label='M1 dipole')
        ax_xz_eeg.plot(dp_mm[0], dp_mm[2], '*', ms=13, color='orange', zorder=100)
        ax_yz_eeg.plot(dp_mm[1], dp_mm[2], '*', ms=13, color='orange', zorder=100)
        ax_xy_eeg.legend(fontsize=7, loc='lower right')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.bwr,
                                    norm=plt.Normalize(vmin=-vmax, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax_yz_eeg, shrink=0.55, label='pV')

        # EEG time traces
        for idx in range(eeg.shape[0]):
            ax_eeg.plot(t, eeg[idx, :], c='gray', lw=0.6)
        ax_eeg.plot(t, eeg[closest_elec_idx, :], c='green', lw=2,
                    label=electrode_labels[closest_elec_idx] + ' (closest)')
        ax_eeg.legend(fontsize=7)

        # ------ bottom row: head geometry views ------
        ax_xy = fig.add_subplot(245, aspect='equal',
                                 xlabel='ML (mm)', ylabel='AP (mm)',
                                 title='Head geometry — axial',
                                 xlim=(-lim, lim), ylim=(-lim, lim))
        ax_xz = fig.add_subplot(246, aspect='equal',
                                 xlabel='ML (mm)', ylabel='DV (mm)',
                                 title='Head geometry — coronal',
                                 xlim=(-lim, lim), ylim=(-lim, lim))
        ax_yz = fig.add_subplot(247, aspect='equal',
                                 xlabel='AP (mm)', ylabel='DV (mm)',
                                 title='Head geometry — sagittal',
                                 xlim=(-lim, lim), ylim=(-lim, lim))
        ax_cdm = fig.add_subplot(248,
                                  xlabel='Time (ms)', ylabel='nA·µm',
                                  title='Current dipole moment')

        # Draw concentric sphere cross-sections (outermost first so inner ones show on top)
        layer_colors  = ['#aed6f1', '#a9dfbf', '#f9e79f', '#f5cba7']  # brain, CSF, skull, scalp
        layer_labels  = ['brain', 'CSF', 'skull', 'scalp']
        for r, col, lbl in zip(r_mm[::-1], layer_colors[::-1], layer_labels[::-1]):
            for ax in (ax_xy, ax_xz, ax_yz):
                ax.fill(r * np.cos(theta), r * np.sin(theta),
                        color=col, alpha=0.7,
                        label=lbl if ax is ax_xy else None, zorder=0)
                ax.plot(r * np.cos(theta), r * np.sin(theta),
                        'k-', lw=0.5, zorder=1)

        # Mark electrode positions on geometry panels (outline only)
        for idx in range(r_electrodes.shape[1]):
            ax_xy.plot(re_mm[0, idx], re_mm[1, idx], 'o', ms=5,
                        c='steelblue', markeredgecolor='k', markeredgewidth=0.4, zorder=5)
            ax_xz.plot(re_mm[0, idx], re_mm[2, idx], 'o', ms=5,
                        c='steelblue', markeredgecolor='k', markeredgewidth=0.4, zorder=5)
            ax_yz.plot(re_mm[1, idx], re_mm[2, idx], 'o', ms=5,
                        c='steelblue', markeredgecolor='k', markeredgewidth=0.4, zorder=5)

        # Mark M1 dipole source
        for ax, xv, yv in [(ax_xy, dp_mm[0], dp_mm[1]),
                            (ax_xz, dp_mm[0], dp_mm[2]),
                            (ax_yz, dp_mm[1], dp_mm[2])]:
            ax.plot(xv, yv, '*', ms=14, color='orange', zorder=10)

        ax_xy.legend(fontsize=7, loc='upper right')

        # Dipole moment components
        ax_cdm.plot(t, p[0, :], label=r'$P_x$ (ML)')
        ax_cdm.plot(t, p[1, :], label=r'$P_y$ (AP)')
        ax_cdm.plot(t, p[2, :], label=r'$P_z$ (DV, apical)', lw=2)
        ax_cdm.legend(fontsize=7)

        # Info annotation
        fig.text(
            0.005, 0.5,
            "M1 pos: ({:.1f}, {:.1f}, {:.1f}) mm\n"
            "4-sphere: brain {:.1f} | CSF {:.1f} | skull {:.1f} | scalp {:.1f} mm\n"
            "σ (S/m): brain {:.2f} | CSF {:.2f} | skull {:.3f} | scalp {:.2f}".format(
                dp_mm[0], dp_mm[1], dp_mm[2],
                r_mm[0], r_mm[1], r_mm[2], r_mm[3],
                0.33, 1.79, 0.008, 0.43),
            va='center', rotation=90, fontsize=7, color='#444444',
        )

    # ==========================================================================
    # Mouse M1 — morphologically realistic atlas model (Allen CCFv3 + BEM)
    # ==========================================================================
    elif head_model == 'mice_M1_atlas':
        import os as _os

        atlas_path = _os.path.normpath(
            _os.path.join(_os.path.dirname(__file__),
                          '..', 'support', 'mice_head', 'mice_M1_atlas.npz'))
        if not _os.path.exists(atlas_path):
            raise FileNotFoundError(
                "Atlas model not found at {}.\n"
                "Run netpyne/support/mice_head/build_head_model.py first.".format(atlas_path))

        atlas = np.load(atlas_path, allow_pickle=True)

        # --- geometry (all in mm) ---
        dp_mm   = atlas['dipole_pos_um']           / 1e3   # (3,)
        re_mm   = atlas['electrode_positions_mm']          # (3, n_elec)
        labels  = list(atlas['electrode_labels'])
        radii_um = atlas['radii_um']                       # (4,) µm
        r_mm    = radii_um / 1e3

        bv_mm   = atlas['brain_verts_mm']                  # (n, 3) centred mm
        cv_mm   = atlas['cortex_verts_mm']
        mv_mm   = atlas['m1_verts_mm']
        skv_mm  = atlas['skull_verts_mm']
        scv_mm  = atlas['scalp_verts_mm']
        bregma_mm = atlas['bregma_mm']                     # (3,)

        M_mat   = atlas['leadfield']                       # (n_elec, 3)

        # --- rotate dipole to M1 surface normal then compute EEG ---
        surface_normal = dp_mm / np.linalg.norm(dp_mm)
        p = _rotate_dipole_to_normal(p, surface_normal, orig_ax_vec=orig_ax_vec)
        eeg = M_mat @ p * 1e9   # nA·µm → mV → pV

        # closest electrode to M1 (by angular proximity on scalp)
        dipole_dir  = dp_mm / np.linalg.norm(dp_mm)
        cos_angles  = np.array([
            np.dot(re_mm[:, i] / np.linalg.norm(re_mm[:, i]), dipole_dir)
            for i in range(re_mm.shape[1])
        ])
        closest_idx = int(np.argmax(cos_angles))
        print("Closest electrode to M1 dipole (atlas): {}".format(labels[closest_idx]))

        max_elec_idx = np.argmax(np.std(eeg, axis=1))
        time_idx     = np.argmax(np.abs(eeg[max_elec_idx]))
        vmax         = np.max(np.abs(eeg[:, time_idx])) or 1.
        cmap         = lambda v: plt.cm.bwr((v + vmax) / (2 * vmax))

        lim  = max(np.abs(scv_mm).max() * 1.15, 9.)

        def _pts_near(verts, axis, val, tol=0.35):
            return verts[np.abs(verts[:, axis] - val) < tol]

        fig = plt.figure(figsize=figSize)
        fig.suptitle("Mouse M1 EEG — Allen CCFv3 atlas head model", fontsize=14, y=0.98)
        fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.42, wspace=0.38,
                            left=0.06, right=0.97)

        # ---- top row: real-anatomy cross-sections + EEG traces ----
        # Coronal slice at M1 A-P: LR (dim2) × DV (dim1)
        ax_cor = fig.add_subplot(241, aspect='equal',
                                  xlabel='L-R (mm)', ylabel='D-V (mm)',
                                  title='Coronal @ M1 A-P\n(real Allen CCFv3)',
                                  xlim=(-lim, lim), ylim=(-lim, lim))
        # Sagittal slice at M1 L-R: AP (dim0) × DV (dim1)
        ax_sag = fig.add_subplot(242, aspect='equal',
                                  xlabel='A-P (mm)', ylabel='D-V (mm)',
                                  title='Sagittal @ M1 L-R',
                                  xlim=(-lim, lim), ylim=(-lim, lim))
        # Axial slice at M1 D-V: AP (dim0) × LR (dim2)
        ax_ax  = fig.add_subplot(243, aspect='equal',
                                  xlabel='A-P (mm)', ylabel='L-R (mm)',
                                  title='Axial @ M1 D-V\n(dorsal view)',
                                  xlim=(-lim, lim), ylim=(-lim, lim))
        ax_eeg = fig.add_subplot(244, xlabel='Time (ms)', ylabel='pV',
                                  title='EEG traces')

        COL_SCALP  = '#f5cba7'; COL_SKULL = '#f9e79f'
        COL_BRAIN  = '#aed6f1'; COL_CX    = '#2e86c1'; COL_M1 = '#e74c3c'

        for ax, xi, yi, sax, sval in [
            (ax_cor, 2, 1, 0, dp_mm[0]),   # coronal: LR×DV, slice on AP
            (ax_sag, 0, 1, 2, dp_mm[2]),   # sagittal: AP×DV, slice on LR
            (ax_ax,  0, 2, 1, dp_mm[1]),   # axial:    AP×LR, slice on DV
        ]:
            ax.set_facecolor('#e8f0f7')
            for verts, col, s, a in [
                (scv_mm, COL_SCALP, 0.6, 0.25),
                (skv_mm, COL_SKULL, 0.6, 0.35),
                (bv_mm,  COL_BRAIN, 0.5, 0.50),
                (cv_mm,  COL_CX,    1.0, 0.75),
                (mv_mm,  COL_M1,    3.5, 1.00),
            ]:
                pts = _pts_near(verts, sax, sval)
                if len(pts):
                    ax.scatter(pts[:, xi], pts[:, yi],
                                c=col, s=s, alpha=a, linewidths=0, rasterized=True)
            # dipole star
            ax.plot(dp_mm[xi], dp_mm[yi], '*', ms=13, c=COL_M1,
                     markeredgecolor='k', markeredgewidth=0.5, zorder=10)
            # bregma marker on coronal/sagittal
            if sax in (0, 2):
                ax.axvline(bregma_mm[xi], color='purple', lw=0.8,
                            ls='--', alpha=0.5, label='Bregma')
            if yi == 1:
                ax.invert_yaxis()   # dorsal on top

        ax_cor.legend(fontsize=7, loc='lower right')

        # EEG traces
        for idx in range(eeg.shape[0]):
            ax_eeg.plot(t, eeg[idx, :], c='gray', lw=0.6)
        ax_eeg.plot(t, eeg[closest_idx, :], c='green', lw=2,
                    label=labels[closest_idx] + ' (closest)')
        ax_eeg.legend(fontsize=7)

        # ---- bottom row: EEG amplitude maps on real scalp ----
        ax_cor2 = fig.add_subplot(245, aspect='equal',
                                   xlabel='L-R (mm)', ylabel='D-V (mm)',
                                   title='EEG amplitude — coronal',
                                   xlim=(-lim, lim), ylim=(-lim, lim))
        ax_sag2 = fig.add_subplot(246, aspect='equal',
                                   xlabel='A-P (mm)', ylabel='D-V (mm)',
                                   title='EEG amplitude — sagittal',
                                   xlim=(-lim, lim), ylim=(-lim, lim))
        ax_ax2  = fig.add_subplot(247, aspect='equal',
                                   xlabel='A-P (mm)', ylabel='L-R (mm)',
                                   title='EEG amplitude — axial',
                                   xlim=(-lim, lim), ylim=(-lim, lim))
        ax_cdm  = fig.add_subplot(248, xlabel='Time (ms)', ylabel='nA·µm',
                                   title='Current dipole moment')

        for ax, xi, yi, sax, sval in [
            (ax_cor2, 2, 1, 0, dp_mm[0]),
            (ax_sag2, 0, 1, 2, dp_mm[2]),
            (ax_ax2,  0, 2, 1, dp_mm[1]),
        ]:
            ax.set_facecolor('#e8f0f7')
            # faint brain outline
            for verts, col, s, a in [
                (scv_mm, COL_SCALP, 0.4, 0.20),
                (bv_mm,  COL_BRAIN, 0.4, 0.35),
            ]:
                pts = _pts_near(verts, sax, sval)
                if len(pts):
                    ax.scatter(pts[:, xi], pts[:, yi],
                                c=col, s=s, alpha=a, linewidths=0, rasterized=True)
            # electrode markers coloured by EEG amplitude
            for idx in range(re_mm.shape[1]):
                ep  = re_mm[:, idx]
                c   = cmap(eeg[idx, time_idx])
                lw  = 2.0 if idx == closest_idx else 0.4
                ax.plot(ep[xi], ep[yi], 'o', ms=9, c=c,
                         markeredgecolor='k', markeredgewidth=lw, zorder=5)
                if ax is ax_ax2:
                    ax.annotate(labels[idx], (ep[xi], ep[yi]),
                                 fontsize=5.5, ha='center', va='bottom',
                                 xytext=(0, 3), textcoords='offset points', zorder=6)
            # M1 dipole
            ax.plot(dp_mm[xi], dp_mm[yi], '*', ms=13, c=COL_M1,
                     markeredgecolor='k', markeredgewidth=0.5, zorder=10)
            if yi == 1:
                ax.invert_yaxis()

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.bwr,
                                    norm=plt.Normalize(vmin=-vmax, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax_ax2, shrink=0.55, label='pV')

        # Dipole moment
        ax_cdm.plot(t, p[0, :], label=r'$P_x$ (A-P)')
        ax_cdm.plot(t, p[1, :], label=r'$P_y$ (D-V)')
        ax_cdm.plot(t, p[2, :], label=r'$P_z$ (L-R, apical)', lw=2)
        ax_cdm.legend(fontsize=7)

        fig.text(
            0.005, 0.5,
            "M1 pos: ({:.1f}, {:.1f}, {:.1f}) mm\n"
            "Brain: Allen CCFv3 #997  |  Isocortex: #315  |  M1/MOp: #985\n"
            "Skull/scalp: morphological dilation of CCFv3 brain mesh\n"
            "4-sphere BEM: brain {:.1f} | skull {:.1f} | scalp {:.1f} mm\n"
            "σ (S/m): brain {:.2f} | CSF {:.2f} | skull {:.3f} | scalp {:.2f}".format(
                dp_mm[0], dp_mm[1], dp_mm[2],
                r_mm[0], r_mm[2], r_mm[3],
                float(atlas['sigmas'][0]), float(atlas['sigmas'][1]),
                float(atlas['sigmas'][2]), float(atlas['sigmas'][3])),
            va='center', rotation=90, fontsize=7, color='#444444',
        )

    else:
        raise ValueError(
            "head_model '{}' not recognised. "
            "Choose 'NYHead', 'mice_M1', or 'mice_M1_atlas'.".format(head_model)
        )

    # ------------------------------------------------------------------ save/show
    if saveFig:
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = sim.cfg.filename + '_EEG.png'
        try:
            plt.savefig(filename, dpi=dpi)
        except:
            plt.savefig('EEG_fig.png', dpi=dpi)

    if showFig is True:
        plt.show()
