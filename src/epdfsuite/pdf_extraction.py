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


def compute_scattering_factors(
    formula,
    x_max,
    x_step,
    qvalues=True,
    xray=False,
):
    """
    Compute both composition-averaged scattering factors in a single pass.

    Returns ``f_avg(q) = sum_i x_i * f_i(q)`` and
    ``<f²>(q) = sum_i x_i * f_i²(q)`` using a single call to
    :class:`LobatoScatteringCalculator`.

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
        If ``True`` (default), axes are in q units (Å⁻¹).
        If ``False``, in s = q/(2π) units.
    xray : bool, optional
        If ``True``, use X-ray scattering factors. Default is ``False``
        (electron scattering factors).

    Returns
    -------
    q : ndarray
        Momentum transfer axis in Å⁻¹.
    favg : ndarray
        Composition-averaged scattering factor <f>(q).
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

    favg  = np.zeros(npts)
    f2avg = np.zeros(npts)
    for i in range(len(elements)):
        favg  += ratios[i] * sf.array[i]
        f2avg += ratios[i] * sf.array[i]**2

    return q, favg, f2avg


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
    q, favg, _ = compute_scattering_factors(
        formula, x_max, x_step, qvalues=qvalues, xray=xray
    )
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
    q, _, f2avg = compute_scattering_factors(
        formula, x_max, x_step, qvalues=qvalues, xray=xray
    )
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


def estimate_affine_high_q_normalization(q, Iexp, f2avg, qmax):
    """
    Estimate affine high-Q normalization parameters alpha and beta.

    Fits ``alpha`` from the covariance/variance relation on the high-Q region,
    then sets ``beta`` to enforce ``mean(alpha*I + beta - <f²>) ~ 0`` there.
    Lightweight guards are applied for numerical stability.

    Parameters
    ----------
    q : ndarray
        Momentum transfer axis in Å⁻¹.
    Iexp : ndarray
        Background-subtracted experimental intensity.
    f2avg : ndarray
        Composition-averaged squared scattering factor <f²>(q).
    qmax : float
        Upper Q limit used for selecting the high-Q region.

    Returns
    -------
    alpha : float
        Multiplicative normalization factor.
    beta : float
        Additive normalization offset.
    """
    eps = np.finfo(float).eps
    mask_inf = q > 0.9 * qmax
    valid_inf = mask_inf & np.isfinite(Iexp) & np.isfinite(f2avg)

    # Fallback for short datasets where top-10% may be too sparse.
    if valid_inf.sum() < 3:
        valid_inf = np.isfinite(Iexp) & np.isfinite(f2avg)

    Ih = Iexp[valid_inf]
    F2h = f2avg[valid_inf]

    # Degenerate fallback (should be rare): keep a sane ratio estimate.
    if Ih.size == 0:
        alpha = 1.0
        beta = 0.0
        return alpha, beta

    Ihc = Ih - Ih.mean()
    F2hc = F2h - F2h.mean()
    var_Ih = np.mean(Ihc**2)

    if var_Ih > eps:
        alpha = np.mean(Ihc * F2hc) / var_Ih
    else:
        alpha = np.mean(F2h / (Ih + eps))

    # Keep alpha close to the physically expected high-Q ratio scale.
    ratio_ref = np.mean(F2h) / (np.mean(Ih) + eps)
    if np.isfinite(ratio_ref) and ratio_ref > 0:
        alpha = np.clip(alpha, 0.2 * ratio_ref, 5.0 * ratio_ref)

    beta = np.mean(F2h - alpha * Ih)

    # Prevent large DC offsets from destabilizing low-Q behavior.
    beta_clip = 0.01 * np.mean(np.abs(f2avg[np.isfinite(f2avg)]))
    if np.isfinite(beta_clip) and beta_clip > 0:
        beta = np.clip(beta, -beta_clip, beta_clip)

    return alpha, beta


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
    2. Normalisation by the composition-averaged squared scattering factor <f²>(Q)
    3. Construction of the reduced structure function:
       ``F(Q) = Q * (I_norm / I_inf - 1)``
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
        ``sinc(Q/Qmax)`` before the Fourier transform to reduce termination
        ripples, with a partial RMS renormalization to limit ON/OFF
        amplitude bias.
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
    The normalization uses a high-Q affine model ``alpha * I + beta``.
    ``alpha`` is estimated on the high-Q region from a covariance/variance
    relation, then ``beta`` is chosen so that the high-Q mean of
    ``alpha * I + beta - <f²>`` is close to zero. The reduced structure
    function follows:
    ``F(Q) = Q * (alpha * I(Q) + beta - <f²>(Q)) / <f>(Q)²``.
    """
    if qmax is None:
        qmax = q.max()
    if qmaxinst is None:
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

    # --- Electron scattering factors (single Lobato call) ---
    q_sf, favg, f2avg = compute_scattering_factors(
        composition,
        x_max=qmax,
        x_step=qstep,
        qvalues=True,
        xray=False,
    )
    favg  = np.interp(q, q_sf, favg)
    f2avg = np.interp(q, q_sf, f2avg)

    # Affine high-Q normalization: alpha*I + beta -> <f²>
    alpha, beta = estimate_affine_high_q_normalization(q, Iexp, f2avg, qmax)

    # --- Modified intensity F(Q) = Q * (S(Q) - 1) ---
    # S(Q) - 1 = (alpha*I + beta - <f²>) / <f>²
    Fm = q * (alpha * Iexp + beta - f2avg) / favg**2

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
        lorch = np.sinc(qv / qmax)
        rms_lorch = np.sqrt(np.mean(lorch**2))
        # Partial RMS renormalization reduces ON/OFF amplitude bias.
        lorch /= (rms_lorch**0.7 + np.finfo(float).eps)
        Fv = Fc[mask] * lorch
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

