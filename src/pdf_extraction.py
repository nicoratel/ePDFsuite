from matplotlib import pyplot as plt
from abtem.parametrizations import LobatoParametrization
import numpy as np
import re
import ipywidgets as widgets
from IPython.display import display
from numpy.polynomial import Polynomial

# --------------------------------------------------
# Chemistry utilities
# --------------------------------------------------

def parse_formula(formula):
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
    elements, ratios = parse_formula(formula)

    if qvalues:
        s_max = x_max / (2 * np.pi)
        s_step = x_step / (2 * np.pi)
    else:
        s_max, s_step = x_max, x_step

    parametrization = LobatoParametrization()
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
    elements, ratios = parse_formula(formula)

    if qvalues:
        s_max = x_max / (2 * np.pi)
        s_step = x_step / (2 * np.pi)
    else:
        s_max, s_step = x_max, x_step

    parametrization = LobatoParametrization()
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
    if qmax is None:
        qmax = q.max()
    if qmaxinst is None:
        qmaxinst = qmax
    Iraw= Iexp.copy()  # Keep a copy of the raw intensity for plotting
    # --- Background subtraction ---
    # First, ensure Iref is on the same q-grid as Iexp by interpolation if needed
    if Iref is not None:
        if len(Iref) != len(Iexp):
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

    mask_inf = q > 0.9 * qmax
    I_inf = np.mean(Iexp[mask_inf])

    Inorm = Iexp / f2avg

    # --- Modified intensity F(Q) ---
    Fm = q * (Inorm / I_inf - 1)

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
    G = (2 / np.pi) * np.trapz(integrand, qv, axis=1)

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
        ax[0].set_ylim([np.min(Iraw[mask_plot]), np.max(Iraw[mask_plot])])

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

