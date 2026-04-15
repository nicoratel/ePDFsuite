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

The app is organised in two tabs.

Tab 1 — Define Sample and Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each column (Sample / Reference) follows the same two-step flow:

**Step 1 — Upload Files**

- **Image** — DM4, DM3, tif or tiff diffraction image
- **PONI file** *(optional)* — pyFAI geometric calibration file
- **Mask file** *(optional)* — EDF mask; or click **🎨 Draw mask** to open
  the pyFAI-drawmask GUI directly on the loaded image
- **MTF file** *(optional)* — for detector deconvolution

  - When a MTF file is provided, an expander **🔧 MTF Deconvolution Parameters**
    appears.  The Wiener regularisation parameter ``ε`` is **read automatically
    from column 3 of the MTF file** and can be adjusted if needed.

For the reference column, PONI, mask and MTF files default to the sample
values when left empty (a caption indicates which file is reused).

**Step 2 — Create Processor & Detect Centre**

Click **🚀 Create SAEDProcessor** to:

1. Instantiate :class:`~epdfsuite.ePDFsuite.SAEDProcessor` (MTF deconvolution
   applied here if a MTF file was provided).
2. Run the iso-intensity contour algorithm automatically to locate the beam
   centre — no manual estimate required.
3. Pre-fill the **Center X / Y** number inputs with the detected coordinates.

A log-scale image of the diffraction pattern is displayed with a white
crosshair at the current centre position.

If detection fails (unusual geometry, very strong beamstop), the pixel with
the maximum intensity is used as a fallback and a warning is printed in the
console.

**Refining the centre manually**

Edit the **Center X** and **Center Y** fields as needed, then click
**🔄 Update** to apply the new coordinates to the processor.  The crosshair
updates accordingly.

.. note::
   When you later click **🚀 Calculate PDF** in Tab 2, the centre is
   synchronised from the number-input fields automatically — the values
   shown in the boxes are always the ones used for integration, whether
   they came from auto-detection or manual entry.

Tab 2 — Extract ePDF
~~~~~~~~~~~~~~~~~~~~~

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
