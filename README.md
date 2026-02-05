# ePDFsuite

A Python library for processing Selected Area Electron Diffraction (SAED) data and extracting Pair Distribution Functions (PDF) for materials characterization.

## Overview

ePDFsuite is designed to streamline the analysis of electron diffraction patterns obtained from Transmission Electron Microscopy (TEM). The library provides tools for calibration, data loading, and PDF extraction, enabling researchers to characterize atomic-scale structures in nanomaterials, amorphous samples, and crystalline materials.

## Features

- **SAED Data Processing**: Process and analyze selected area electron diffraction patterns
- **Camera Calibration**: Calibrate TEM camera constants for accurate measurements
- **File Format Support**: Read various electron microscopy data formats
- **PDF Extraction**: Extract pair distribution functions from electron diffraction data
- **Recalibration Tools**: Fine-tune calibration parameters for improved accuracy
- **Workflow Integration**: Jupyter notebook workflow for streamlined analysis

## Repository Structure

```
ePDFsuite/
├── ePDFsuite.py                    # Main library module
├── calibration.py                  # Camera calibration utilities
├── recalibration.py                # Recalibration tools
├── filereader.py                   # File I/O handlers
├── pdf_extraction.py               # PDF extraction algorithms
├── camera_library.py               # Camera-specific configurations
├── ePDF_Workflow.ipynb            # Example workflow notebook
└── documentation/                  # PDF documentation files
    ├── ePDF_Workflow_SAEDProcessor_documentation.pdf
    ├── ePDFsuite_documentation_calibration.pdf
    ├── ePDFsuite_documentation_dataloading.pdf
    └── ePDFsuite_documentation_pdfextraction.pdf
```

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy
- SciPy
- Matplotlib
- Additional dependencies (see requirements.txt if available)

### Basic Installation

```bash
git clone https://github.com/nicoratel/ePDFsuite.git
cd ePDFsuite
pip install -r requirements.txt  # If available
```



### 5. Example Workflow

For a complete analysis workflow, refer to the Jupyter notebook:

```bash
jupyter notebook ePDF_Workflow.ipynb
```

## Documentation

Detailed documentation is available in PDF format:

- **SAED Processing Workflow**: `ePDF_Workflow_SAEDProcessor_documentation.pdf`
- **Calibration Guide**: `ePDFsuite_documentation_calibration.pdf`
- **Data Loading**: `ePDFsuite_documentation_dataloading.pdf`
- **PDF Extraction**: `ePDFsuite_documentation_pdfextraction.pdf`

## Key Concepts

### SAED (Selected Area Electron Diffraction)

SAED is a crystallographic technique performed in transmission electron microscopes to analyze the structure of materials at the nanoscale. It produces diffraction patterns that reveal information about crystal orientation, lattice parameters, and structural defects.

### PDF (Pair Distribution Function)

The atomic pair distribution function provides information about the local atomic structure of materials by analyzing the distribution of interatomic distances. This is particularly valuable for characterizing:

- Nanocrystalline materials
- Amorphous structures
- Local structural disorder
- Short-range and medium-range atomic ordering

## Use Cases

- **Nanomaterials Characterization**: Determine atomic structures of nanoparticles and nanoclusters
- **Phase Identification**: Identify crystal structures and phases in materials
- **Disorder Analysis**: Study local atomic arrangements in disordered materials
- **Catalysis Research**: Characterize active sites in catalytic nanoparticles
- **Materials Science**: Investigate structure-property relationships

## Workflow Overview

1. **Data Acquisition**: Collect SAED patterns using TEM
2. **Calibration**: Determine camera constant using known standards
3. **Preprocessing**: Background subtraction and normalization
4. **Azimuthal Integration**: Convert 2D patterns to 1D intensity profiles
5. **PDF Calculation**: Fourier transform to obtain pair distribution function
6. **Analysis**: Extract structural information (coordination numbers, bond lengths)

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Guidelines

- Follow Python PEP 8 style guidelines
- Include docstrings for all functions and classes
- Add unit tests for new features
- Update documentation as needed

## License

Please check the repository for license information.

## Citation

If you use ePDFsuite in your research, please cite:

```
[N. Ratel-Ramond]. (2026). ePDFsuite: Python library for SAED data processing and PDF extraction.
GitHub repository: https://github.com/nicoratel/ePDFsuite
```



## Support

For questions, issues, or support:

- Open an issue on [GitHub Issues](https://github.com/nicoratel/ePDFsuite/issues)
- Consult the documentation PDFs included in the repository
- Review the example workflow notebook

## Acknowledgments

This library builds upon established methods for electron diffraction analysis and PDF extraction, incorporating best practices from the electron microscopy and materials characterization communities.

---

**Note**: This is a research tool. Always validate results with appropriate standards and controls for your specific application.
