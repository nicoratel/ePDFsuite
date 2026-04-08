Streamlit application
=====================

For users who prefer a graphical interface over a Python script, ePDFsuite
includes a **Streamlit web application** that exposes the full workflow through
a browser-based GUI — no programming required.

Online version (no installation)
---------------------------------

The app is publicly hosted on the Streamlit Community Cloud — **no Python,
no installation required**:

   👉 `https://epdfsuite.streamlit.app/ <https://epdfsuite.streamlit.app/>`_

Simply open the link in your browser, upload your files and start processing.

Local installation
------------------

For offline use or to work with a local installation, see the `launch guide on GitHub
<https://github.com/nicoratel/ePDFsuite/blob/main/LAUNCH_APP.md>`_
for detailed instructions (conda environment setup, ``epdfsuite-app``
command, or bash script).

In short:

.. code-block:: bash

   conda activate epdfsuite
   epdfsuite-app

The app opens at ``http://localhost:8501``.

Interface overview
------------------

The app is organised in two tabs:

**Tab 1 — Define Sample and Reference**

Upload your files and configure the processing parameters:

- **Sample image** — DM4, DM3, tif or tiff diffraction image
- **PONI file** *(optional)* — pyFAI geometric calibration file
- **Mask file** *(optional)* — EDF mask for invalid pixels or beamstop
- **MTF file** *(optional)* — for MTF deconvolution (Richardson-Lucy or Wiener)
- **Reference image** — background / amorphous film image (same format)

For each image, a log-scale preview is displayed. Enter the approximate
beam centre ``(x, y)`` in pixels — the app will automatically refine it
using the iterative ring-detection algorithm.

**Tab 2 — Extract ePDF**

Adjust the PDF computation parameters with sliders and extract G(r):

- ``bgscale`` — background scaling factor
- ``qmin``, ``qmax``, ``qmaxinst`` — Q-range limits (Å⁻¹)
- ``rpoly`` — polynomial background degree (PDFgetX3 convention)
- Lorch modification function toggle

The G(r) plot updates in real time. Click **Save** to export the result
as a ``.gr`` file compatible with PDFgui and PDFBatchAnalysis.

When to use the app vs. the Python API
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Use case
     - App
     - Python API
   * - Quick exploration of a new dataset
     - ✅ recommended
     - possible
   * - Reproducible batch processing
     - ✗
     - ✅ recommended
   * - Processing multiple files automatically
     - ✗
     - ✅ :func:`~epdfsuite.ePDFsuite.extract_ePDF_from_mutliple_files`
   * - Integration in a Jupyter notebook
     - ✗
     - ✅ recommended
