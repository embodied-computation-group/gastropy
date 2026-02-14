"""Brain map visualization for gastric-brain coupling results.

Provides convenience wrappers around nilearn plotting functions
with sensible defaults for PLV and z-score coupling maps.

Requires nilearn (optional dependency): ``pip install gastropy[neuro]``
"""

import numpy as np


def _to_img(data, affine):
    """Convert data to a nibabel image if not already one."""
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("nibabel is required: pip install gastropy[neuro]") from e

    if isinstance(data, nib.spatialimages.SpatialImage):
        return data
    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)


def plot_coupling_map(
    data, affine=None, threshold=None, title="Gastric-Brain Coupling (PLV)", cmap="YlOrRd", vmax=None, ax=None, **kwargs
):
    """Plot a volumetric coupling map as an anatomical overlay.

    Wrapper around ``nilearn.plotting.plot_stat_map`` with defaults
    appropriate for PLV maps (non-negative, warm colormap).

    Parameters
    ----------
    data : np.ndarray or nibabel.Nifti1Image
        3D coupling volume (e.g., PLV or z-score map). If a numpy
        array, ``affine`` must be provided.
    affine : np.ndarray, optional
        4x4 affine matrix. Required if ``data`` is a numpy array.
    threshold : float, optional
        Values below this threshold are not displayed.
    title : str, optional
        Plot title.
    cmap : str, optional
        Matplotlib colormap. Default is ``"YlOrRd"`` for PLV,
        use ``"RdBu_r"`` for z-score maps.
    vmax : float, optional
        Maximum value for the colormap.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes. If ``None``, a new figure is created.
    **kwargs
        Additional arguments passed to ``nilearn.plotting.plot_stat_map``.

    Returns
    -------
    display : nilearn.plotting.displays.OrthoSlicer
        Nilearn display object for further customization.
    """
    try:
        from nilearn import plotting
    except ImportError as e:
        raise ImportError("nilearn is required: pip install gastropy[neuro]") from e

    img = _to_img(data, affine)
    display = plotting.plot_stat_map(
        img,
        threshold=threshold,
        title=title,
        cmap=cmap,
        vmax=vmax,
        symmetric_cbar=False,
        axes=ax,
        **kwargs,
    )
    return display


def plot_glass_brain(data, affine=None, threshold=None, title="Gastric-Brain Coupling", cmap="YlOrRd", **kwargs):
    """Plot a coupling map as a transparent glass brain.

    Wrapper around ``nilearn.plotting.plot_glass_brain`` for
    quick overview visualizations of coupling results.

    Parameters
    ----------
    data : np.ndarray or nibabel.Nifti1Image
        3D coupling volume. If a numpy array, ``affine`` must
        be provided.
    affine : np.ndarray, optional
        4x4 affine matrix. Required if ``data`` is a numpy array.
    threshold : float, optional
        Values below this threshold are not displayed.
    title : str, optional
        Plot title.
    cmap : str, optional
        Matplotlib colormap. Default is ``"YlOrRd"``.
    **kwargs
        Additional arguments passed to ``nilearn.plotting.plot_glass_brain``.

    Returns
    -------
    display : nilearn.plotting.displays.OrthoProjector
        Nilearn display object for further customization.
    """
    try:
        from nilearn import plotting
    except ImportError as e:
        raise ImportError("nilearn is required: pip install gastropy[neuro]") from e

    img = _to_img(data, affine)
    display = plotting.plot_glass_brain(
        img,
        threshold=threshold,
        title=title,
        cmap=cmap,
        colorbar=True,
        **kwargs,
    )
    return display
