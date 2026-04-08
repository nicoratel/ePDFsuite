Quick start
===========

Basic workflow
--------------

All processing goes through :class:`~epdfsuite.ePDFsuite.SAEDProcessor` and the
standalone :func:`~epdfsuite.ePDFsuite.extract_epdf` function.

**1 — Load and inspect your SAED image**

.. code-block:: python

   from epdfsuite import SAEDProcessor

   proc = SAEDProcessor(
       'sample.dm4',
       poni_file='calibration.poni',
       mtf_file='camera.mtf',   # optional: MTF deconvolution
   )
   proc.plot()   # inspect to locate beam centre

**2 — Set the beam centre and integrate**

.. code-block:: python

   proc.initial_center = (338, 271)   # (x, y) in pixels from plot
   q, I = proc.integrate(plot=True)

**3 — Extract the ePDF**

.. code-block:: python

   from epdfsuite import SAEDProcessor, extract_epdf

   sample = SAEDProcessor('sample.dm4', poni_file='calib.poni')
   sample.initial_center = (338, 271)

   ref = SAEDProcessor('reference.dm4', poni_file='calib.poni')
   ref.initial_center = (335, 268)

   # Interactive mode: adjust sliders, then unpack
   results = extract_epdf(sample, ref, composition='Au', interactive=True)
   r, G = results

   # Non-interactive mode
   r, G = extract_epdf(sample, ref, composition='Au',
                       interactive=False, plot=True)

With MTF deconvolution
----------------------

Pass the ``mtf_file`` argument at initialisation to apply Richardson-Lucy
deconvolution before integration:

.. code-block:: python

   proc = SAEDProcessor(
       'sample.dm4',
       poni_file='calib.poni',
       mtf_file='OneView_2k.mtf',
       filter='rl',
       n_iterations=50,
   )

Without a PONI file (scaled images)
------------------------------------

If no pyFAI calibration file is available, ePDFsuite falls back to a
pixel-scale integrator using the scale stored in the DM4 metadata:

.. code-block:: python

   proc = SAEDProcessor('sample.dm4')   # no poni_file
   proc.initial_center = (512, 512)
   q, I = proc.integrate()
