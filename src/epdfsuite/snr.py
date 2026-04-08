import numpy as np

def compute_SNR(r, g, r_cut=0.75):
    """
    Compute the signal-to-noise ratio of a PDF curve G(r).

    The signal is defined as the maximum of G(r). The noise is estimated
    as the standard deviation of G(r) in the high-r tail, where the PDF
    is expected to converge to zero.

    Parameters
    ----------
    r : ndarray
        Real-space distance axis in Å.
    g : ndarray
        Reduced pair distribution function G(r).
    r_cut : float, optional
        Fraction of ``r.max()`` above which the signal is considered noise.
        Default is 0.75 (i.e. the top 25 % of the r range is used).

    Returns
    -------
    snr : float
        Signal-to-noise ratio: ``max(G) / std(G[r > r_cut * r_max])``.
    """
    mask = r > r_cut * r.max()
    noise = np.std(g[mask])
    return g.max()/noise


def compute_SNR_from_file(file, r_cut=0.75):
    """
    Load a G(r) file and compute its signal-to-noise ratio.

    The file is expected to be a two-column text file (r, G) with 27 header
    lines, as produced by ePDFsuite.

    Parameters
    ----------
    file : str
        Path to the G(r) text file.
    r_cut : float, optional
        Fraction of ``r.max()`` used to define the noise tail.
        See :func:`compute_SNR`. Default is 0.75.

    Returns
    -------
    snr : float
        Signal-to-noise ratio of the loaded G(r) curve.
    """
    r, g = np.loadtxt(file, skiprows=27, unpack=True)
    return compute_SNR(r,g,r_cut  = r_cut)