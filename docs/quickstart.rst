Quick start
===========

Typical workflow
----------------

A complete ePDF extraction follows four steps:

1. Load the image and inspect it visually to estimate the beam centre.
2. Provide this estimate as ``initial_center`` — the software then
   **automatically recalibrates** the exact centre by fitting the most
   intense diffraction ring (iterative moment method).
3. Verify the recalibration visually.
4. Integrate and extract G(r).

All processing goes through :class:`~epdfsuite.ePDFsuite.SAEDProcessor` and the
standalone :func:`~epdfsuite.ePDFsuite.extract_epdf` function.

**Step 1 — Load and inspect the image**

.. code-block:: python

   from epdfsuite import SAEDProcessor

   proc = SAEDProcessor(
       'sample.dm4',
       poni_file='calibration.poni',
       mtf_file='camera.mtf',   # optional: apply MTF deconvolution at load time
   )
   proc.plot()   # display the image in log scale to visually locate the beam centre

Read the approximate beam centre ``(x, y)`` in pixels from the plot —
this will be used as the starting point for automatic recalibration.

**Step 2 — Set the initial centre estimate**

.. code-block:: python

   proc.initial_center = (338, 271)   # (x, y) in pixels, rough estimate from plot

.. note::
   This is only an *initial guess*. When ``integrate()`` is called,
   ePDFsuite automatically refines the beam centre by iteratively
   detecting the brightest diffraction ring and computing an
   intensity-weighted centroid until convergence
   (see :func:`~epdfsuite.recalibration.recalibrate_with_beamstop`).
   The PONI file is updated internally with the corrected centre.

**Step 3 — Verify the recalibration**

Before extracting the PDF, it is strongly recommended to check that the
recalibrated centre is correct:

.. code-block:: python

   proc.plot_recalibrated_image()   # displays image with detected centre and ring overlay

If the detected centre looks wrong, adjust ``initial_center`` and repeat.
Once satisfied, you can skip recalibration on subsequent calls to speed
things up:

.. code-block:: python

   proc.skip_center_recalibration = True   # reuse the centre found above

**Step 4 — Integrate and extract the ePDF**

.. code-block:: python

   from epdfsuite import SAEDProcessor, extract_epdf

   # Sample
   sample = SAEDProcessor('sample.dm4', poni_file='calib.poni')
   sample.initial_center = (338, 271)

   # Reference (background / amorphous carbon film)
   ref = SAEDProcessor('reference.dm4', poni_file='calib.poni')
   ref.initial_center = (335, 268)

   # Verify recalibrations
   sample.plot_recalibrated_image()
   ref.plot_recalibrated_image()

   # Interactive mode: adjust sliders, then unpack results
   results = extract_epdf(sample, ref, composition='Au', interactive=True)
   r, G = results   # values reflect the last slider state

   # Non-interactive mode (fixed parameters)
   r, G = extract_epdf(sample, ref, composition='Au',
                       interactive=False, plot=True,
                       bgscale=1.0, qmin=1.5, qmax=20, rpoly=1.4)

With MTF deconvolution
----------------------

Pass ``mtf_file`` at initialisation to apply Richardson-Lucy deconvolution
**before** integration. This corrects for the detector point-spread function.

.. code-block:: python

   proc = SAEDProcessor(
       'sample.dm4',
       poni_file='calib.poni',
       mtf_file='OneView_2k.mtf',
       filter='rl',        # 'rl' (Richardson-Lucy) or 'wiener'
       n_iterations=50,
   )

.. note::
   For integrating detectors (e.g. Gatan OneView, US1000), the MTF decays
   strongly with spatial frequency and has **no plateau** — this is physically
   expected. Deconvolution is particularly beneficial for such cameras.

Without a PONI file (scaled images)
-------------------------------------

If no pyFAI calibration file is available, ePDFsuite falls back to a
pixel-scale integrator using the pixel size stored in the DM4 metadata:

.. code-block:: python

   proc = SAEDProcessor('sample.dm4')   # no poni_file
   proc.initial_center = (512, 512)
   q, I = proc.integrate()
