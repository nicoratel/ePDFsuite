from matplotlib import pyplot as plt
from .lobato_scattering import LobatoScatteringCalculator
import numpy as np
import re
from numpy.polynomial import Polynomial

# --------------------------------------------------
# Chemistry utilities
# --------------------------------------------------

def parse_formula(formula):
    """
    Parse a chemical formula string into element symbols and molar fractions.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. ``'SiO2'``, ``'Al2O3'``, ``'Fe0.5Ni0.5'``.

    Returns
    -------
    elements : list of str
        Element symbols in the order they appear in the formula.
    ratios : list of float
        Molar fractions of each element (sum to 1).
    """
    tokens = re.findall(r'([A-Z][a-z]*)([0-9.]+)?', formula)
    elements, counts = [], []
    for elem, count in tokens:
        elements.append(elem)
        counts.append(float(count) if count else 1.0)
    counts = np.array(counts)
    ratios = counts / counts.sum()
    return elements, ratios.tolist()


def compute_avg_scattering_factor(
    formula,
    x_max,
    x_step,
    qvalues=True,
    xray=False,
):
    """
    Compute the composition-averaged atomic scattering factor f_avg(q).

    The average is the weighted sum over all elements:
    ``f_avg(q) = sum_i x_i * f_i(q)``
    where ``x_i`` are the molar fractions parsed from ``formula``.

    Parameters
    ----------
    formula : str
        Chemical formula of the sample (e.g. ``'SiO2'``).
    x_max : float
        Upper limit of the scattering variable axis.
        Interpreted as q (Å⁻¹) if ``qvalues=True``, otherwise as s = q/(2π).
    x_step : float
        Sampling step of the scattering variable axis, same units as ``x_max``.
    qvalues : bool, optional
        If ``True`` (default), ``x_max`` and ``x_step`` are in q units (Å⁻¹).
        If ``False``, they are in s = q/(2π) units.
    xray : bool, optional
        If ``True``, use X-ray scattering factors instead of electron scattering
        factors. Default is ``False`` (electron factors).

    Returns
    -------
    q : ndarray
        Momentum transfer axis in Å⁻¹.
    favg : ndarray
        Composition-averaged scattering factor f_avg(q).
    """
    elements, ratios = parse_formula(formula)

    if qvalues:
        s_max = x_max / (2 * np.pi)
        s_step = x_step / (2 * np.pi)
    else:
        s_max, s_step = x_max, x_step

    parametrization = LobatoScatteringCalculator()
    name = "x_ray_scattering_factor" if xray else "scattering_factor"

    sf = parametrization.line_profiles(
        elements,
        cutoff=s_max,
        sampling=s_step,
        name=name,
    )

    npts = sf.array.shape[1]
    s = np.arange(npts) * s_step
    q = 2 * np.pi * s

    favg = np.zeros(npts)
    for i in range(len(elements)):
        favg += ratios[i] * sf.array[i]

    return q, favg


def compute_f2avg(
    formula,
    x_max,
    x_step,
    qvalues=True,
    xray=False,
):
    """
    Compute the composition-averaged squared scattering factor <f²>(q).

    The average is the weighted sum of squared individual factors:
    ``<f²>(q) = sum_i x_i * f_i²(q)``
    where ``x_i`` are the molar fractions parsed from ``formula``.
    This quantity is used for the normalisation of the reduced structure
    function F(Q) in the PDFgetX3 formalism.

    Parameters
    ----------
    formula : str
        Chemical formula of the sample (e.g. ``'SiO2'``).
    x_max : float
        Upper limit of the scattering variable axis.
        Interpreted as q (Å⁻¹) if ``qvalues=True``, otherwise as s = q/(2π).
    x_step : float
        Sampling step of the scattering variable axis, same units as ``x_max``.
    qvalues : bool, optional
        If ``True`` (default), ``x_max`` and ``x_step`` are in q units (Å⁻¹).
        If ``False``, they are in s = q/(2π) units.
    xray : bool, optional
        If ``True``, use X-ray scattering factors. Default is ``False``
        (electron scattering factors).

    Returns
    -------
    q : ndarray
        Momentum transfer axis in Å⁻¹.
    f2avg : ndarray
        Composition-averaged squared scattering factor <f²>(q).
    """
    elements, ratios = parse_formula(formula)

    if qvalues:
        s_max = x_max / (2 * np.pi)
        s_step = x_step / (2 * np.pi)
    else:
        s_max, s_step = x_max, x_step

    parametrization = LobatoScatteringCalculator()
    name = "x_ray_scattering_factor" if xray else "scattering_factor"

    sf = parametrization.line_profiles(
        elements,
        cutoff=s_max,
        sampling=s_step,
        name=name,
    )

    npts = sf.array.shape[1]
    s = np.arange(npts) * s_step
    q = 2 * np.pi * s

    f2avg = np.zeros(npts)
    for i in range(len(elements)):
        f2avg += ratios[i] * sf.array[i]**2

    return q, f2avg


# --------------------------------------------------
# Polynomial background (PDFgetX3 style)
# --------------------------------------------------

def fit_polynomial_background(q, Fm, rpoly=0.9, qmin=0.3, qmax=None):
    """
    Fit and return a polynomial background to the reduced structure function F(Q).

    Follows the PDFgetX3 convention: the polynomial degree is determined by
    ``deg = round(rpoly * qmax / π)``, and the fit is performed on F(Q)/Q
    to enforce the correct low-Q behaviour.

    Parameters
    ----------
    q : ndarray
        Momentum transfer axis in Å⁻¹.
    Fm : ndarray
        Reduced structure function F(Q) = Q * (I_norm / I_inf - 1).
    rpoly : float, optional
        Polynomial degree control parameter (PDFgetX3 convention). Default is 0.9.
    qmin : float, optional
        Lower bound of the fitting range in Å⁻¹. Default is 0.3.
    qmax : float, optional
        Upper bound of the fitting range in Å⁻¹. Defaults to ``q.max()``.

    Returns
    -------
    background : ndarray
        Polynomial background evaluated on the full ``q`` grid, same shape as ``Fm``.
    """
    if qmax is None:
        qmax = q.max()

    mask = (q >= qmin) & (q <= qmax)
    deg = int(round(rpoly * qmax / np.pi))
    deg = max(1, min(deg, mask.sum() - 1))

    y = Fm[mask] / q[mask]
    poly = Polynomial.fit(q[mask], y, deg=deg, domain=[qmin, qmax])

    return q * poly(q)


# --------------------------------------------------
# PDFgetX3-like PDF (ELECTRONS)
# --------------------------------------------------

def compute_ePDF(
    q,
    Iexp,
    composition,
    Iref=None,
    bgscale=1.0,
    qmin=0.3,
    qmax=None,
    qmaxinst=None,
    rmin=0.0,
    rmax=50.0,
    rstep=0.01,
    rpoly=1.4,
    Lorch=True,
    plot=False,
):
    """
    Compute the electron Pair Distribution Function G(r) from a SAED intensity profile.

    Follows the PDFgetX3 formalism adapted for electron scattering:

    1. Optional background subtraction: ``I = Iexp - bgscale * Iref``
     2. High-Q scaling ``alpha`` so that ``alpha*I(Q)`` matches ``<f²>(Q)``
     3. Construction of the reduced structure function from
         ``S(Q)-1 = (alpha*I(Q) - <f²>(Q)) / <f>(Q)²`` and ``F(Q)=Q*(S(Q)-1)``
    4. Polynomial background removal (PDFgetX3 convention, controlled by ``rpoly``)
    5. Optional Lorch modification function to suppress Fourier ripples
    6. Sine Fourier transform to obtain G(r)

    Parameters
    ----------
    q : ndarray
        Momentum transfer axis in Å⁻¹.
    Iexp : ndarray
        Experimental azimuthally averaged intensity profile.
    composition : str
        Chemical formula of the sample (e.g. ``'SiO2'``, ``'Al2O3'``).
    Iref : ndarray, optional
        Reference (background) intensity profile. If its length differs from
        ``Iexp``, it is interpolated onto the ``q`` grid. Default is ``None``.
    bgscale : float, optional
        Scaling factor applied to the reference before subtraction. Default is 1.0.
    qmin : float, optional
        Minimum Q used for the Fourier transform (Å⁻¹). Default is 0.3.
    qmax : float, optional
        Maximum Q used for the Fourier transform (Å⁻¹). Defaults to ``q.max()``.
    qmaxinst : float, optional
        Maximum Q used for the polynomial background fit. Defaults to ``qmax``.
        Useful when the data are noisy near ``qmax``.
    rmin : float, optional
        Minimum real-space distance r (Å). Default is 0.0.
    rmax : float, optional
        Maximum real-space distance r (Å). Default is 50.0.
    rstep : float, optional
        Step size in real space (Å). Default is 0.01.
    rpoly : float, optional
        Polynomial degree control for background removal (PDFgetX3 convention).
        Default is 1.4.
    Lorch : bool, optional
        If ``True`` (default), apply the Lorch modification function
        ``sinc(Q/Qmax)`` before the Fourier transform to reduce termination ripples.
    plot : bool, optional
        If ``True``, display diagnostic plots of the raw intensities, F(Q),
        and G(r). Default is ``False``.

    Returns
    -------
    r : ndarray
        Real-space distance axis in Å.
    G : ndarray
        Reduced pair distribution function G(r) in Å⁻².

    Notes
    -----
    The scale factor ``alpha`` is estimated from the top 10 % of the Q range
    (``q > 0.9 * qmax``) so that ``alpha * I(Q)`` matches ``<f²>(Q)`` in the
    high-Q region. The reduced function is then built explicitly as
    ``S(Q)-1 = (alpha*I(Q) - <f²>(Q)) / <f>(Q)²`` and ``F(Q)=Q*(S(Q)-1)``.
    """
    if qmax is None or qmax > q.max():
        qmax = q.max()
    if qmaxinst is None or qmaxinst > q.max():
        qmaxinst = qmax
    Iraw= Iexp.copy()  # Keep a copy of the raw intensity for plotting

    # --- Interpolate over NaN/Inf bins (from masked radial bins) ---
    finite_exp = np.isfinite(Iexp)
    if not np.all(finite_exp):
        Iexp = np.interp(q, q[finite_exp], Iexp[finite_exp])
        Iraw = Iexp.copy()

    # --- Background subtraction ---
    # First, ensure Iref is on the same q-grid as Iexp by interpolation if needed
    if Iref is not None:
        finite_ref = np.isfinite(Iref)
        if not np.all(finite_ref) and finite_ref.any():
            q_ref_full = np.linspace(q[0], q[-1], len(Iref))
            Iref = np.interp(q, q_ref_full[finite_ref], Iref[finite_ref])
        elif len(Iref) != len(Iexp):
            # Create a q-grid for the reference data based on its length
            q_ref = np.linspace(q[0], q[-1], len(Iref))
            # Interpolate reference intensity to match the sample's q-grid
            Iref = np.interp(q, q_ref, Iref)
    
    # Then subtract the background
    if Iref is not None:
        Iexp = Iexp - bgscale * Iref

    qstep = q[1] - q[0]

    # --- Electron scattering normalization ---
    q_f2, f2avg = compute_f2avg(
        composition,
        x_max=qmax,
        x_step=qstep,
        qvalues=True,
        xray=False,
    )
    f2avg = np.interp(q, q_f2, f2avg)

    q_f, favg = compute_avg_scattering_factor(
        composition,
        x_max=qmax,
        x_step=qstep,
        qvalues=True,
        xray=False,
    )
    favg = np.interp(q, q_f, favg)

    mask_inf = q > 0.9 * qmax
    if np.any(mask_inf):
        alpha = np.mean(f2avg[mask_inf]) / np.mean(Iexp[mask_inf])
    else:
        alpha = np.mean(f2avg) / np.mean(Iexp)

    # --- Modified intensity F(Q) ---
    S_minus_1 = (alpha * Iexp - f2avg) / (favg ** 2)
    Fm = q * S_minus_1

    # --- Polynomial background (PDFgetX3 philosophy) ---
    background = fit_polynomial_background(
        q, Fm, rpoly=rpoly, qmin=qmin, qmax=qmaxinst
    )

    Fc = Fm - background  # NO Q-DAMPING

    # --- Fourier transform ---
    r = np.arange(rmin, rmax + rstep, rstep)
    mask = (q >= qmin) & (q <= qmax)
    qv = q[mask]

    if Lorch:
        Fv = Fc[mask] * np.sinc(qv / qmax)
    else:
        Fv = Fc[mask]

    integrand = Fv[None, :] * np.sin(np.outer(r, qv))
    # Use np.trapezoid (NumPy >= 1.22) with fallback to np.trapz for older versions
    # G = (2 / np.pi) * np.trapz(integrand, qv, axis=1)  # np.trapz is deprecated in NumPy 1.22+
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    G = (2 / np.pi) * trapz_func(integrand, qv, axis=1)

    # Optional diagnostic plots
    if plot:
        fig, ax = plt.subplots(3, figsize=(4, 6))
        
        # Plot 1: Raw intensities
        ax[0].plot(q, Iraw, label="Iexp")
        if Iref is not None:
            ax[0].plot(q, bgscale * Iref, label="Ref*bgscale")
        ax[0].legend()
        ax[0].set_xlabel("Q ($\\AA^{-1}$)")
        ax[0].set_ylabel("Intensity")
        # set q limits to [qmin,qmax]
        mask_plot = (q >= qmin) & (q <= qmax)
        ax[0].set_xlim([qmin, qmax])
        # set intensity limits to [min(Iexp), max(Iexp)] in the q range
        Iraw_valid = Iraw[mask_plot][np.isfinite(Iraw[mask_plot])]
        if len(Iraw_valid) > 0:
            ax[0].set_ylim([np.min(Iraw_valid), np.max(Iraw_valid)])

        # Plot 2: Corrected structure factor
        ax[1].plot(q, Fc, label=f"rpoly={rpoly:.2f}")
        ax[1].legend()
        ax[1].set_xlabel("Q ($\\AA^{-1}$)")
        ax[1].set_ylabel("F(Q)")
        ax[1].set_xlim([qmin, qmax])
        # Filter out NaN and Inf values before setting y limits
        Fc_valid = Fc[mask_plot][np.isfinite(Fc[mask_plot])]
        if len(Fc_valid) > 0:
            ax[1].set_ylim([np.min(Fc_valid), np.max(Fc_valid)])
        else:
            ax[1].set_ylim([0, 1])  # Fallback to default limits if no valid values

        # Plot 3: Final PDF
        ax[2].plot(r, G, label=f"rpoly={rpoly:.2f}")
        ax[2].legend()
        ax[2].set_xlabel("r ($\\AA$)")
        ax[2].set_ylabel("G(r)")

        fig.tight_layout()
        plt.show()

    return r, G


# --------------------------------------------------
# PDFgetX3-like PDF (X-RAYS)
# --------------------------------------------------

def compute_xPDF(
    q,
    Iexp,
    composition,
    Iref=None,
    bgscale=1.0,
    qmin=0.3,
    qmax=None,
    qmaxinst=None,
    rmin=0.0,
    rmax=50.0,
    rstep=0.01,
    rpoly=1.4,
    Lorch=True,
    plot=False,
):
    """
    Compute the X-ray Pair Distribution Function G(r) from an azimuthally
    integrated diffraction intensity profile.

    Follows the PDFgetX3 formalism adapted for X-ray scattering:

    1. Optional background subtraction: ``I = Iexp - bgscale * Iref``
     2. High-Q scaling ``alpha`` so that ``alpha*I(Q)`` matches ``<f²>(Q)``
     3. Construction of the reduced structure function from
         ``S(Q)-1 = (alpha*I(Q) - <f²>(Q)) / <f>(Q)²`` and ``F(Q)=Q*(S(Q)-1)``
    4. Polynomial background removal (PDFgetX3 convention, controlled by ``rpoly``)
    5. Optional Lorch modification function to suppress Fourier ripples
    6. Sine Fourier transform to obtain G(r)

    Parameters
    ----------
    q : ndarray
        Momentum transfer axis in Å⁻¹.
    Iexp : ndarray
        Experimental azimuthally averaged intensity profile.
    composition : str
        Chemical formula of the sample (e.g. ``'SiO2'``, ``'Al2O3'``).
    Iref : ndarray, optional
        Reference (background) intensity profile. If its length differs from
        ``Iexp``, it is interpolated onto the ``q`` grid. Default is ``None``.
    bgscale : float, optional
        Scaling factor applied to the reference before subtraction. Default is 1.0.
    qmin : float, optional
        Minimum Q used for the Fourier transform (Å⁻¹). Default is 0.3.
    qmax : float, optional
        Maximum Q used for the Fourier transform (Å⁻¹). Defaults to ``q.max()``.
    qmaxinst : float, optional
        Maximum Q used for the polynomial background fit. Defaults to ``qmax``.
        Useful when the data are noisy near ``qmax``.
    rmin : float, optional
        Minimum real-space distance r (Å). Default is 0.0.
    rmax : float, optional
        Maximum real-space distance r (Å). Default is 50.0.
    rstep : float, optional
        Step size in real space (Å). Default is 0.01.
    rpoly : float, optional
        Polynomial degree control for background removal (PDFgetX3 convention).
        Default is 1.4.
    Lorch : bool, optional
        If ``True`` (default), apply the Lorch modification function
        ``sinc(Q/Qmax)`` before the Fourier transform to reduce termination ripples.
    plot : bool, optional
        If ``True``, display diagnostic plots of the raw intensities, F(Q),
        and G(r). Default is ``False``.

    Returns
    -------
    r : ndarray
        Real-space distance axis in Å.
    G : ndarray
        Reduced pair distribution function G(r) in Å⁻².

    Notes
    -----
    The scale factor ``alpha`` is estimated from the top 10 % of the Q range
    (``q > 0.9 * qmax``) so that ``alpha * I(Q)`` matches ``<f²>(Q)`` in the
    high-Q region. The reduced function is then built explicitly as
    ``S(Q)-1 = (alpha*I(Q) - <f²>(Q)) / <f>(Q)²`` and ``F(Q)=Q*(S(Q)-1)``.
    """
    if qmax is None or qmax > q.max():
        qmax = q.max()
    if qmaxinst is None or qmaxinst > q.max():
        qmaxinst = qmax
    Iraw = Iexp.copy()

    # --- Interpolate over NaN/Inf bins (from masked radial bins) ---
    finite_exp = np.isfinite(Iexp)
    if not np.all(finite_exp):
        Iexp = np.interp(q, q[finite_exp], Iexp[finite_exp])
        Iraw = Iexp.copy()

    # --- Background subtraction ---
    if Iref is not None:
        finite_ref = np.isfinite(Iref)
        if not np.all(finite_ref) and finite_ref.any():
            q_ref_full = np.linspace(q[0], q[-1], len(Iref))
            Iref = np.interp(q, q_ref_full[finite_ref], Iref[finite_ref])
        elif len(Iref) != len(Iexp):
            q_ref = np.linspace(q[0], q[-1], len(Iref))
            Iref = np.interp(q, q_ref, Iref)

    if Iref is not None:
        Iexp = Iexp - bgscale * Iref

    qstep = q[1] - q[0]

    # --- X-ray scattering normalization ---
    q_f2, f2avg = compute_f2avg(
        composition,
        x_max=qmax,
        x_step=qstep,
        qvalues=True,
        xray=True,
    )
    f2avg = np.interp(q, q_f2, f2avg)

    q_f, favg = compute_avg_scattering_factor(
        composition,
        x_max=qmax,
        x_step=qstep,
        qvalues=True,
        xray=True,
    )
    favg = np.interp(q, q_f, favg)

    mask_inf = q > 0.9 * qmax
    if np.any(mask_inf):
        alpha = np.mean(f2avg[mask_inf]) / np.mean(Iexp[mask_inf])
    else:
        alpha = np.mean(f2avg) / np.mean(Iexp)

    # --- Modified intensity F(Q) ---
    S_minus_1 = (alpha * Iexp - f2avg) / (favg ** 2)
    Fm = q * S_minus_1

    # --- Polynomial background (PDFgetX3 philosophy) ---
    background = fit_polynomial_background(
        q, Fm, rpoly=rpoly, qmin=qmin, qmax=qmaxinst
    )

    Fc = Fm - background

    # --- Fourier transform ---
    r = np.arange(rmin, rmax + rstep, rstep)
    mask = (q >= qmin) & (q <= qmax)
    qv = q[mask]

    if Lorch:
        Fv = Fc[mask] * np.sinc(qv / qmax)
    else:
        Fv = Fc[mask]

    integrand = Fv[None, :] * np.sin(np.outer(r, qv))
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    G = (2 / np.pi) * trapz_func(integrand, qv, axis=1)

    # Optional diagnostic plots
    if plot:
        fig, ax = plt.subplots(3, figsize=(4, 6))

        ax[0].plot(q, Iraw, label="Iexp")
        if Iref is not None:
            ax[0].plot(q, bgscale * Iref, label="Ref*bgscale")
        ax[0].legend()
        ax[0].set_xlabel("Q ($\\AA^{-1}$)")
        ax[0].set_ylabel("Intensity")
        mask_plot = (q >= qmin) & (q <= qmax)
        ax[0].set_xlim([qmin, qmax])
        Iraw_valid = Iraw[mask_plot][np.isfinite(Iraw[mask_plot])]
        if len(Iraw_valid) > 0:
            ax[0].set_ylim([np.min(Iraw_valid), np.max(Iraw_valid)])

        ax[1].plot(q, Fc, label=f"rpoly={rpoly:.2f}")
        ax[1].legend()
        ax[1].set_xlabel("Q ($\\AA^{-1}$)")
        ax[1].set_ylabel("F(Q)")
        ax[1].set_xlim([qmin, qmax])
        Fc_valid = Fc[mask_plot][np.isfinite(Fc[mask_plot])]
        if len(Fc_valid) > 0:
            ax[1].set_ylim([np.min(Fc_valid), np.max(Fc_valid)])
        else:
            ax[1].set_ylim([0, 1])

        ax[2].plot(r, G, label=f"rpoly={rpoly:.2f}")
        ax[2].legend()
        ax[2].set_xlabel("r ($\\AA$)")
        ax[2].set_ylabel("G(r)")

        fig.tight_layout()
        plt.show()

    return r, G

