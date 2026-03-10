import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt



def estimate_mtf_amorphous_carbon(processor,
                                  q_fit_max = 0.9,
                                  window=31,
                                  percentile=90,
                                  smooth_sigma=3,                                  
                                  plot=False,
                                  outputfile='./mtf_camera.mtf'):
    """
    Robust estimation of detector MTF from amorphous carbon diffraction.

    Parameters
    ----------
    processor : SAEDProcessor
        SAED processor instance with amorphous carbon data loaded, and optional mask /poni files
    q_fit : tuple
        q-range used for fitting
    window : int
        rolling window size
    percentile : float
        percentile used for envelope extraction
    smooth_sigma : float
        gaussian smoothing applied to MTF
    diagnostics : bool
        return intermediate curves

    Returns
    -------
    mtf_fit : array
        fitted MTF(q)
    a : float
        parameter of exp(-a q²)
    """

    q,I = processor.integrate(plot=False)

    mask = (q > 0) & (I > 0) & np.isfinite(I)
    q = q[mask]
    I = I[mask]

    # -----------------------------
    # remove atomic factor decay
    # -----------------------------

    I_flat = q**2 * I

    # -----------------------------
    # rolling percentile envelope
    # -----------------------------

    env = np.zeros_like(I_flat)

    half = window // 2

    for i in range(len(I_flat)):
        lo = max(0, i-half)
        hi = min(len(I_flat), i+half)
        env[i] = np.percentile(I_flat[lo:hi], percentile)

    env = gaussian_filter(env, 3)

    # -----------------------------
    # experimental MTF
    # -----------------------------

    mtf_exp = I_flat / env
    
    # Apply smoothing before normalization
    mtf_exp = gaussian_filter(mtf_exp, smooth_sigma)
    
    # Normalize by the first value in the fit range
    q_fit=[0.5,q_fit_max*q.max()]
    fit_mask = (q > q_fit[0]) & (q < q_fit[1])
    mtf_exp /= mtf_exp[fit_mask][0]

    # -----------------------------
    # fit exponential model
    # -----------------------------

    def mtf_model(q, a, q0):
        q_eff = np.maximum(q - q0, 0)
        return np.exp(-a * q_eff**2)

    popt, _ = curve_fit(mtf_model, q[fit_mask], mtf_exp[fit_mask], p0=[1, np.mean(q[fit_mask])],bounds=[[0,q[fit_mask].min()],[np.inf,q[fit_mask].max()]])

    a = popt[0]
    q0 = popt[1]
  
    mtf_fit = mtf_model(q, a, q0)
    
    if plot:
        
        plt.figure()
        plt.plot(q, mtf_exp, label="MTF experimental", linewidth=2)
        plt.plot(q, mtf_fit, label=f"MTF fit (a={a:.5f})", linewidth=2, linestyle='--')
        plt.axvline(q_fit[0], color='r', linestyle='--', alpha=0.5, label='fit range')
        plt.axvline(q_fit[1], color='r', linestyle='--', alpha=0.5)
        plt.legend()
        plt.xlabel("q (Å⁻¹)")
        plt.ylabel("MTF")
        #plt.ylim([0, 1.1])
        plt.title("MTF estimation")

        plt.show()

    np.savetxt(outputfile, np.column_stack((q, mtf_fit)), comments='')
    print(f'a={a}, q0={q0}')
    return mtf_fit


def correct_intensity_for_mtf(q, I, q_mtf,mtf):
    """
    Correct measured intensity for detector MTF.

    Parameters
    ----------
    q : array
        scattering vector (Å⁻¹)
    I : array
        measured intensity
    q_mtf : array
        q-values for MTF
    mtf : array
        detector MTF
    """
    # interp MTF to measurement q values
    mtf_interp = np.interp(q, q_mtf, mtf, left=1, right=0)

    return I / mtf_interp