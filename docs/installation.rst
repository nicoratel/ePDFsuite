Installation
============

No-install option
-----------------

The fastest way to get started is the **hosted Streamlit app** — no Python,
no installation required:

   👉 `https://epdfsuite.streamlit.app/ <https://epdfsuite.streamlit.app/>`_

Open the link in your browser, upload your files and extract ePDFs directly.
See :doc:`streamlit_app` for a description of the interface.


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

To also install the optional **PDF analysis** companion library:

.. code-block:: bash

   pip install ePDFsuite[pdfanalysis]

See `Optional dependencies`_ below for details.

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

Optional dependencies
---------------------

`pdfanalysis <https://github.com/nicoratel/pdfanalysis>`_ is an independent
companion library for PDF analysis. It is **not required** by ePDFsuite but
provides an additional Streamlit-based application accessible via the
command line:

.. code-block:: bash

   pdfanalysis-app

Install it alongside ePDFsuite with:

.. code-block:: bash

   pip install ePDFsuite[pdfanalysis]

or as a standalone package:

.. code-block:: bash

   pip install pdfanalysis
