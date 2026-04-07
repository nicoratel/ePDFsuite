# ePDFsuite

**ePDFsuite** is a Python toolkit for extracting the electron Pair Distribution Function (ePDF) from Selected Area Electron Diffraction (SAED) images acquired in a transmission electron microscope (TEM).

It provides both a Python API for scripted workflows and an interactive graphical interface built with Streamlit.

---

## What is ePDF?

The electron Pair Distribution Function (ePDF) is a real-space representation of atomic pair correlations, derived from the Fourier transform of the reduced structure factor $F(q)$ extracted from electron diffraction data. It is particularly useful for characterizing the local atomic structure of amorphous, nanocrystalline, and disordered materials.

$$G(r) = \frac{2}{\pi} \int_{q_{\min}}^{q_{\max}} F(q) \sin(qr)\, dq$$

---

## Features

- **Camera geometric calibration** using a known crystalline standard (e.g. Au) via [pyFAI](https://pyfai.readthedocs.io/)
- **MTF determination** from a beamstop image using the slanted edge method
- **MTF deconvolution** of diffraction images (Wiener filter or Richardson-Lucy algorithm)
- **Automatic beam center recalibration** before each integration
- **Azimuthal integration** of 2D SAED patterns to 1D profiles
- **ePDF extraction**: background subtraction, scattering factor normalization using Lobato parameterization, and Fourier transform
- **Interactive GUI** (Streamlit) for interactive parameter tuning
- Support for **DM4, DM3, TIFF** input formats (via [HyperSpy](https://hyperspy.org/))
- Works with or without a `.poni` calibration file

---

## Installation

**Requirements:** Python ≥ 3.8, conda recommended.

```bash
# Clone the repository
git clone https://github.com/nicoratel/ePDFsuite.git
cd ePDFsuite

# Create and activate a conda environment
conda create -n epdfsuite python=3.10
conda activate epdfsuite

# Install in editable mode
pip install -e .
```

---

## Launching the GUI

Once installed, launch the interactive Streamlit application from anywhere:

```bash
conda activate epdfsuite
epdfsuite-app
```

The app opens at `http://localhost:8501` in your browser.

> See [LAUNCH_APP.md](LAUNCH_APP.md) for alternative launch methods.

---

## Usage

### Step 0 — Camera calibration

Generate a `.poni` calibration file from a diffraction image of a known calibrant (e.g. Au nanoparticles) and a CIF structure file.

```python
from epdfsuite.calibration import perform_geometric_calibration

perform_geometric_calibration(
    image_file="Au_calibrant.dm4",
    cif_file="Au.cif",
    output_poni="calib.poni"
)
```

> This step is performed once per session/camera configuration. See [`notebooks/camera_calibration.ipynb`](notebooks/camera_calibration.ipynb) for a complete walkthrough.

---

### Step 1 — MTF determination (optional)

If you want to correct for the camera's Modulation Transfer Function, first measure it from a beamstop image using the slanted edge method.

```python
from epdfsuite.utilities import draw_mask, compute_mtf_slanted_edge

# Draw a mask to isolate the edge region
draw_mask("beamstop_image.dm4")

# Compute the MTF
freq, mtf = compute_mtf_slanted_edge(
    "beamstop_image.dm4",
    mask="mask.edf",
    roi_half_width=100,
    outputfile="camera.mtf"
)
```

> See [`notebooks/MTF_determination.ipynb`](notebooks/MTF_determination.ipynb) for details.

---

### Step 2 — Load and inspect SAED data

Create a `SAEDProcessor` instance for both the sample and the reference (amorphous carbon film) diffraction images.

```python
from epdfsuite import SAEDProcessor

# With pyFAI geometric calibration
sample = SAEDProcessor(
    "sample_diff.dm4",
    poni_file="calib.poni",
    mask="mask.edf",        # optional pixel mask
    mtf_file="camera.mtf",  # optional MTF correction
    filter="rl",            # 'rl' (Richardson-Lucy) or 'wiener'
    verbose=True
)

# Visualize the raw diffraction image
sample.plot()

# Set an approximate beam center and verify recalibration
sample.initial_center = [350, 313]
sample.plot_recalibrated_image()

# Integrate to 1D profile
q, I = sample.integrate(plot=True)
```

> If no `.poni` file is available, `SAEDProcessor` falls back to the pixel scale stored in the DM4 metadata.

---

### Step 3 — ePDF extraction

#### Using the `extract_epdf` standalone function

```python
from epdfsuite import SAEDProcessor, extract_epdf

sample = SAEDProcessor("sample_diff.dm4", poni_file="calib.poni", mask="mask.edf")
sample.initial_center = [350, 313]

ref = SAEDProcessor("ref_Cfilm.dm4", poni_file="calib.poni", mask="mask.edf")
ref.initial_center = [330, 232]

results = extract_epdf(
    sample,
    ref,
    composition="Au",
    rmin=0.0,
    rmax=50.0,
    rstep=0.01,
    interactive=True        # opens interactive parameter tuning
)
```

#### Using the method on the instance

```python
sample.extract_epdf(
    ref_diffraction_image="ref_Cfilm.dm4",
    composition="Au",
    rmin=0.1,
    rmax=50.0,
    rstep=0.01,
    bgscale=0.5,
    interactive=False,
    plot=True,
    outputfile="Au_NPs_epdf.gr"
)
```

> Output `.gr` files follow the PDFgui/diffpy-CMI format and can be read by any PDF analysis software.

---

## Notebooks

| Notebook | Description |
|---|---|
| [`camera_calibration.ipynb`](notebooks/camera_calibration.ipynb) | Geometric calibration with pyFAI + MTF measurement |
| [`ePDF_Workflow.ipynb`](notebooks/ePDF_Workflow.ipynb) | Complete ePDF extraction workflow (basic) |
| [`ePDF_Workflow_MTF.ipynb`](notebooks/ePDF_Workflow_MTF.ipynb) | Complete workflow with MTF correction |
| [`MTF_determination.ipynb`](notebooks/MTF_determination.ipynb) | MTF computation from a beamstop image |
| [`MTF_deconvolution.ipynb`](notebooks/MTF_deconvolution.ipynb) | Comparison of deconvolution methods |

---

## Project structure

```
ePDFsuite/
├── src/epdfsuite/
│   ├── ePDFsuite.py          # SAEDProcessor class and extract_epdf function
│   ├── calibration.py        # Geometric calibration from CIF
│   ├── recalibration.py      # Automatic beam center refinement
│   ├── pdf_extraction.py     # Structure factor and PDF computation
│   ├── lobato_scattering.py  # Lobato electron scattering factors
│   ├── filereader.py         # DM4/DM3/TIFF reader
│   ├── utilities.py          # MTF tools, mask drawing
│   ├── camera_library.py     # Known detector configurations
│   └── app_epdfsuite.py      # Streamlit GUI
├── notebooks/                # Example Jupyter notebooks
├── pyproject.toml
├── LAUNCH_APP.md             # GUI launch instructions
└── README.md
```

---

## Dependencies

| Package | Role |
|---|---|
| [pyFAI](https://pyfai.readthedocs.io/) | Geometric calibration and azimuthal integration |
| [HyperSpy](https://hyperspy.org/) | Reading DM4/DM3 files and pixel scale metadata |
| [diffpy-CMI / numpy](https://numpy.org/) | PDF computation and signal processing |
| [scikit-image](https://scikit-image.org/) | Image processing (MTF, beam center) |
| [Streamlit](https://streamlit.io/) | Interactive GUI |
| [plotly](https://plotly.com/python/) | Interactive plots in the GUI |
| [pymatgen](https://pymatgen.org/) | CIF reading for calibration |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Nicolas Ratel-Ramond  
[github.com/nicoratel/ePDFsuite](https://github.com/nicoratel/ePDFsuite)
