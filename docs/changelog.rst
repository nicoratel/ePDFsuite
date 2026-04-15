Changelog
=========

0.2.0 (current)
----------------

**Beam-centre detection — complete refactor**

- Unique centre-detection method: **iso-intensity contour fitting**
  (:func:`~epdfsuite.recalibration.recalibrate_from_isocurve`).
  All previous methods (iterative ring moment, ``recalibrate_with_beamstop``)
  have been removed.
- Centre detected **automatically at** ``SAEDProcessor.__init__`` — no
  manual ``initial_center`` needed.  Result stored in ``proc.center`` as a
  plain ``(cx, cy)`` tuple.
- ``proc.center`` can be read and overwritten freely before calling
  ``integrate()`` or ``extract_epdf()``.
- Removed attributes: ``initial_center``, ``skip_center_recalibration``,
  ``use_ring_fit``, ``use_isocurve``.
- Graceful fallback to intensity-maximum pixel if isocurve detection fails
  (e.g. unusual image geometry or very strong beamstop), with a printed warning.

**MTF deconvolution**

- Richardson-Lucy filter removed; **Wiener filter** is now the only deconvolution
  method.
- Wiener ``epsilon`` is **read from column 3 of the MTF file** when
  ``wiener_epsilon=None`` (default).

**Streamlit app — Tab 1 redesign**

- New two-step flow per column (Sample / Reference):

  1. Upload files (image, PONI, mask, MTF) — mask can also be drawn
     interactively via the pyFAI-drawmask GUI.
  2. Click **🚀 Create SAEDProcessor** — processor is created, isocurve
     runs, centre auto-filled in the **Center X / Y** fields.

- Wiener ``ε`` field pre-filled from column 3 of the MTF file on upload.
- **🔄 Update** button applies edited centre coordinates to the processor.
- At PDF calculation time (Tab 2), centre is synchronised from the
  number-input fields before ``integrate()`` — the displayed value is
  always the one used.
- Removed: global "Validate and Create Processors" button, dropdown for
  recalibration method selection, duplicated MTF parameter section below
  the columns.

**NaN handling in** ``compute_ePDF``

- Masked azimuthal-integration bins (NaN / zero-weight) are interpolated
  over before PDF computation, preventing crashes in PDFgetX3.

0.1.4
-----

- Asymmetric ROI in ``compute_mtf_slanted_edge`` for beamstop-side images
- MTF deconvolution: Wiener filter
- Interactive ePDF extraction GUI (ipywidgets)
- pyFAI-based beam-centre recalibration with beamstop
- NumPy docstrings across all modules

0.1.0
-----

- Initial release
