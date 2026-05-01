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
- **MTF deconvolution** of diffraction images (Wiener filter)
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

> See `https://epdfsuite.readthedocs.io/en/latest/`

## Notebooks

| Notebook | Description |
|---|---|
| [`camera_calibration.ipynb`](notebooks/camera_calibration.ipynb) | Geometric calibration with pyFAI + MTF measurement |
| [`ePDF_Workflow.ipynb`](notebooks/ePDF_Workflow.ipynb) | Complete ePDF extraction workflow (basic) |
| [`MTF_determination.ipynb`](notebooks/MTF_determination.ipynb) | MTF computation from a beamstop image |


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
