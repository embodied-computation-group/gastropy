"""Create the brain_maps example notebook."""

import json

cells = []


def md(source):
    lines = source.strip().split("\n")
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
        }
    )


def code(source):
    lines = source.strip().split("\n")
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
        }
    )


md(
    "# Visualizing Brain Coupling Maps\n"
    "\n"
    "This example shows how to create publication-ready brain\n"
    "visualizations of gastric-brain coupling results using\n"
    "GastroPy's nilearn-based plotting functions.\n"
    "\n"
    "We generate a synthetic PLV volume in MNI space to demonstrate\n"
    "the available visualization options."
)

code(
    "import matplotlib.pyplot as plt\n"
    "import nibabel as nib\n"
    "import numpy as np\n"
    "\n"
    "import gastropy as gp\n"
    "from gastropy.neuro.fmri import to_nifti\n"
    "\n"
    'plt.rcParams["figure.dpi"] = 100\n'
    'plt.rcParams["figure.facecolor"] = "white"'
)

md(
    "## Create a Synthetic PLV Volume\n"
    "\n"
    "We create a fake PLV volume in MNI space using the nilearn\n"
    "template for realistic brain geometry."
)

code(
    "from nilearn import datasets, image\n"
    "\n"
    "# Load MNI template for reference geometry\n"
    "mni = datasets.load_mni152_brain_mask(resolution=2)\n"
    "mask_data = mni.get_fdata().astype(bool)\n"
    "affine = mni.affine\n"
    "vol_shape = mask_data.shape\n"
    "\n"
    "# Create synthetic PLV: higher values in frontal/insular regions\n"
    "rng = np.random.default_rng(42)\n"
    "plv_flat = 0.02 + 0.01 * rng.standard_normal(mask_data.sum())\n"
    "plv_flat = np.clip(plv_flat, 0, 1)\n"
    "\n"
    "# Add a hotspot near the insula (MNI ~[-40, 10, 0])\n"
    "coords = np.argwhere(mask_data)\n"
    "center_vox = np.array([35, 65, 45])  # approximate insula\n"
    "dists = np.linalg.norm(coords - center_vox, axis=1)\n"
    "plv_flat[dists < 8] += 0.04\n"
    "\n"
    "plv_3d = np.zeros(vol_shape)\n"
    "plv_3d[mask_data] = plv_flat\n"
    'print(f"Volume shape: {plv_3d.shape}")\n'
    'print(f"Brain voxels: {mask_data.sum():,}")\n'
    'print(f"PLV range: [{plv_flat.min():.4f}, {plv_flat.max():.4f}]")'
)

md("## Anatomical Overlay (``plot_coupling_map``)")

code(
    "plv_img = to_nifti(plv_3d, affine)\n"
    "\n"
    "display = gp.plot_coupling_map(\n"
    '    plv_img, threshold=0.03, title="PLV Map (Anatomical Overlay)"\n'
    ")\n"
    "plt.show()"
)

md("## Glass Brain (``plot_glass_brain``)")

code('display = gp.plot_glass_brain(\n    plv_img, threshold=0.03, title="PLV Map (Glass Brain)"\n)\nplt.show()')

md(
    "## Customization\n"
    "\n"
    "Both functions accept nilearn keyword arguments and return\n"
    "display objects for further customization."
)

code(
    "# Different colormaps and thresholds\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "\n"
    "gp.plot_coupling_map(\n"
    '    plv_img, threshold=0.04, cmap="hot",\n'
    '    title="Hot colormap", ax=axes[0],\n'
    ")\n"
    "gp.plot_coupling_map(\n"
    '    plv_img, threshold=0.04, cmap="plasma",\n'
    '    title="Plasma colormap", ax=axes[1],\n'
    ")\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("docs/examples/brain_maps.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created notebook with {len(cells)} cells")
