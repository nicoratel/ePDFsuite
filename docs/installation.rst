Installation
============

Requirements
------------

ePDFsuite requires Python 3.8 or later and the following packages:

- numpy, scipy, matplotlib
- hyperspy (DM4/DM3 file reading)
- pyFAI (geometric calibration and azimuthal integration)
- scikit-image (MTF computation, image processing)
- pymatgen (scattering factor tables)
- ipywidgets (interactive Jupyter GUI)

From PyPI (recommended)
-----------------------

.. code-block:: bash

   pip install ePDFsuite

From source
-----------

.. code-block:: bash

   git clone https://github.com/nicoratel/ePDFsuite.git
   cd ePDFsuite
   pip install -e .

Camera calibration
------------------

MTF correction requires a pre-measured MTF file (``.mtf``).
See the `calibration guide on GitHub
<https://github.com/nicoratel/ePDFsuite/blob/main/INSTALLATION_GUIDE_Calibration.md>`_
for instructions on measuring the MTF with the slanted-edge method.
