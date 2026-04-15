Quick start
===========

Typical workflow
----------------

A complete ePDF extraction follows three steps:

1. Create a :class:`~epdfsuite.ePDFsuite.SAEDProcessor` — the beam centre is
   **detected automatically** at initialisation using the iso-intensity
   contour method.
2. Verify (and optionally refine) the centre visually.
3. Integrate and extract G(r).

All processing goes through :class:`~epdfsuite.ePDFsuite.SAEDProcessor` and the
standalone :func:`~epdfsuite.ePDFsuite.extract_epdf` function.

**Step 1 — Create the processor (auto-detects beam centre)**

.. code-block:: python

   from epdfsuite import SAEDProcessor

   proc = SAEDProcessor(
       'sample.dm4',
       poni_file='calibration.poni',
       mtf_file='camera.mtf',   # optional: apply MTF deconvolution at load time
   )
   # proc.center is now set automatically as (cx, cy) in pixels

At initialisation :class:`~epdfsuite.ePDFsuite.SAEDProcessor` calls
:func:`~epdfsuite.recalibration.recalibrate_from_isocurve` on the loaded
image to locate the beam centre from iso-intensity contours.  No manual
estimate is required.  If the automatic detection fails (unusual image
geometry, very strong beamstop), the pixel with the maximum intensity is
used as a safe fallback and a warning is printed.

**Step 2 — Inspect and refine the centre if needed**

.. code-block:: python

   print(proc.center)   # (cx, cy) in pixels — auto-detected

   # Visualise the result (iso-contour overlay):
   proc.plot_recalibrated_image()

   # Override manually if needed:
   proc.center = (338, 271)

.. note::
   ``proc.center`` is a plain ``(cx, cy)`` tuple; you can read and overwrite
   it freely at any point before calling ``integrate()``.

**Step 3 — Integrate and extract the ePDF**

.. code-block:: python

   from epdfsuite import SAEDProcessor, extract_epdf

   # Sample
   sample = SAEDProcessor('sample.dm4', poni_file='calib.poni')

   # Reference (background / amorphous carbon film)
   ref = SAEDProcessor('reference.dm4', poni_file='calib.poni')

   # Verify detected centres
   sample.plot_recalibrated_image()
   ref.plot_recalibrated_image()

   # Interactive mode: adjust sliders, then unpack results
   results = extract_epdf(sample, ref, composition='Au', interactive=True)
   r, G = results   # values reflect the last slider state

   # Non-interactive mode (fixed parameters)
   r, G = extract_epdf(sample, ref, composition='Au',
                       interactive=False, plot=True,
                       bgscale=1.0, qmin=1.5, qmax=20, rpoly=1.4)

Passing a custom centre to ``integrate()``
------------------------------------------

If you need to override the centre for a single integration call without
permanently changing ``proc.center``, pass it as a keyword argument:

.. code-block:: python

   q, I = proc.integrate(center=(340, 275))
   # proc.center is now permanently updated to (340, 275)

With MTF deconvolution
----------------------

Pass ``mtf_file`` at initialisation to apply deconvolution **before**
integration.  The regularisation parameter ``wiener_epsilon`` is read
automatically from the third column of the MTF file.

.. code-block:: python

   proc = SAEDProcessor(
       'sample.dm4',
       poni_file='calib.poni',
       mtf_file='OneView_2k.mtf',
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
   # proc.center auto-detected; override if needed:
   proc.center = (512, 512)
   q, I = proc.integrate()
