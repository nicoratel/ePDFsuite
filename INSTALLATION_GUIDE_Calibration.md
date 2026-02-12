# Installation Guide: `perform_geometric_calibration` Dependencies

This document describes all the dependencies required to use the `perform_geometric_calibration()` function from the ePDFsuite package.

## Overview

The `perform_geometric_calibration()` function performs geometric calibration of electron diffraction detectors using pyFAI-calib2 GUI. It requires both explicit and implicit dependencies.

## Dependencies

### Core Dependencies

#### 1. **PyFAI** (Explicit)
The primary library for geometric calibration of diffraction data.

```bash
pip install pyFAI
```

**Purpose**: Provides the calibration GUI (pyFAI-calib2), geometry management, and integration capabilities.

**Additional requirement**: pyFAI-calib2 requires **Qt** for the graphical interface. Ensure Qt5 or Qt6 is installed on your system.

---

#### 2. **Pymatgen** (Explicit)
Materials genome project library for crystal structure handling.

```bash
pip install pymatgen
```

**Purpose**: 
- Reads CIF (Crystallographic Information File) files
- Calculates expected X-ray diffraction patterns from crystal structures
- Generates calibration data from polycrystalline standards (e.g., Au, Si)

**Sub-dependencies**: 
- `monty`
- `spglib`
- `networkx`
- `sympy`
- `ruamel.yaml`

---

#### 3. **HyperSpy** (Explicit)
Hyperspectral data analysis library, specifically for reading electron diffraction images.

```bash
pip install hyperspy[mpl,gui]
```

**Purpose**: 
- Reads DM4/DM3 files (Gatan diffraction image formats)
- Handles metadata from electron microscopy files
- Provides image display and manipulation tools

**Note**: Install with `[mpl,gui]` extras for full functionality.

**Sub-dependencies**:
- `dask`
- `scikit-image`
- `sympy`
- `h5py`
- `tqdm`

---

#### 4. **Fabio** (Implicit via PyFAI)
Supports reading various diffraction image formats.

**Purpose**: Image I/O for multiple diffraction detector formats.

**Auto-installed**: Usually installed automatically as a dependency of PyFAI.

---

### Secondary Dependencies

#### 5. **NumPy** (Implicit)
Fundamental numerical computing library.

```bash
pip install numpy
```

**Purpose**: Array operations, numerical calculations for image processing.

---

#### 6. **SciPy** (Implicit)
Scientific computing library.

```bash
pip install scipy
```

**Purpose**: 
- Signal processing (`scipy.signal`)
- Image processing (`scipy.ndimage`)
- Interpolation functions

---

#### 7. **Matplotlib** (Implicit)
Plotting and visualization library.

```bash
pip install matplotlib
```

**Purpose**: Image display, peak plotting during calibration.

---

#### 8. **Scikit-Image** (Implicit via HyperSpy)
Image processing algorithms.

**Purpose**: Image analysis and enhancement.

**Auto-installed**: Usually installed as a dependency of HyperSpy.

---

## Installation Procedures

### Option 1: Complete Installation with pip

Install all dependencies at once:

```bash
pip install pyFAI pymatgen hyperspy[mpl,gui] numpy scipy matplotlib scikit-image
```

### Option 2: Using requirements.txt

Create a `requirements.txt` file with:

```
pyFAI>=0.22.0
pymatgen>=2022.11.0
hyperspy[mpl,gui]>=1.7.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
scikit-image>=0.19.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Option 3: Using Conda (Recommended for complex environments)

```bash
conda create -n epdf_env python=3.10
conda activate epdf_env
conda install -c conda-forge pyfai pymatgen hyperspy numpy scipy matplotlib scikit-image
```

---

## System Requirements

### Operating System
- **Linux**: Full support ✓
- **macOS**: Full support ✓
- **Windows**: Full support ✓ (Note: Qt GUI may require additional configuration)

### Python Version
- **Python 3.9+** (Recommended: 3.10 or later)

### Qt Framework (Required for pyFAI-calib2 GUI)

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install qt5-qmake qt5-default
```

#### macOS
```bash
brew install qt5
```

#### Windows
Qt is typically included with PyFAI when installing from pip. If issues occur, download from [qt.io](https://www.qt.io).

---

## Verification

After installation, verify all dependencies are installed correctly:

```python
import pyfai
import pymatgen
import hyperspy.api as hs
import numpy as np
import scipy
import matplotlib
import skimage

print("✓ All dependencies installed successfully!")
print(f"PyFAI version: {pyfai.__version__}")
print(f"Pymatgen version: {pymatgen.__version__}")
print(f"HyperSpy version: {hs.__version__}")
```

---

## Testing the Installation

To test if `perform_geometric_calibration()` is ready to use:

```python
from calibration import perform_geometric_calibration

# Example usage (requires test files)
perform_geometric_calibration(
    image_file="path/to/calibrant_image.dm4",
    cif_file="path/to/Au.cif"
)
```

---

## Troubleshooting

### Issue: Qt GUI not launching

**Solution**: Install PyQt5 or PyQt6 explicitly:
```bash
pip install PyQt5
# or
pip install PyQt6
```

### Issue: HyperSpy cannot read DM4 files

**Solution**: Ensure HyperSpy is installed with IO plugins:
```bash
pip install hyperspy[all]
```

### Issue: ImportError for pymatgen

**Solution**: Reinstall with build dependencies:
```bash
pip install --upgrade --force-reinstall pymatgen
```

### Issue: On macOS with Apple Silicon (M1/M2/M3)

**Solution**: Use conda with ARM64 support:
```bash
conda create -n epdf_env python=3.11
conda activate epdf_env
conda install -c conda-forge pyfai pymatgen hyperspy
```

---

## Optional Dependencies

The following packages are optional but recommended:

### Jupyter Lab/Notebook
For interactive data analysis:
```bash
pip install jupyter jupyterlab
```

### IPython
Enhanced interactive shell:
```bash
pip install ipython
```

### Pillow
Image processing utilities:
```bash
pip install pillow
```

---

## Summary Table

| Package | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| PyFAI | ≥0.22.0 | Geometric calibration | `pip install pyFAI` |
| Pymatgen | ≥2022.11.0 | CIF parsing & XRD | `pip install pymatgen` |
| HyperSpy | ≥1.7.0 | DM4 image reading | `pip install hyperspy[mpl,gui]` |
| NumPy | ≥1.20.0 | Numerical arrays | Auto-installed |
| SciPy | ≥1.7.0 | Scientific functions | Auto-installed |
| Matplotlib | ≥3.3.0 | Plotting | Auto-installed |
| Scikit-Image | ≥0.19.0 | Image processing | Auto-installed |
| Fabio | (any) | Image I/O | Auto-installed |

---

## Additional Resources

- [PyFAI Documentation](https://pyfai.readthedocs.io/)
- [Pymatgen Documentation](https://pymatgen.org/)
- [HyperSpy Documentation](https://hyperspy.org/)
- [ePDFsuite Documentation](./README_GUI.md)

---

**Last Updated**: February 2026
