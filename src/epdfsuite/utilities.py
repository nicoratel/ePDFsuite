from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf
import numpy as np
from matplotlib import pyplot as plt
from .filereader import load_data
from scipy.ndimage import rotate, binary_erosion
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
import fabio
import os
import sys
import shutil


def draw_mask(dm4_image):
    """
    Launch the pyFAI-drawmask GUI to interactively draw a pixel mask.

    The input DM4 image is temporarily exported as an EDF file, passed to
    the ``pyFAI-drawmask`` tool, then the EDF file is deleted. The mask
    produced by the GUI is saved alongside the image by pyFAI.

    Parameters
    ----------
    dm4_image : str
        Path to the DM4 image file.
    """
    # load data and metadata
    detector_info, raw_image = load_data(dm4_image)

    # Define output EDF file name
    edffile = dm4_image.replace('.dm4', '.edf')

    # Create EDF image and save
    edf_image = fabio.edfimage.EdfImage(data=raw_image, header=detector_info)
    edf_image.write(edffile)
    # edit command to use the same python executable as the current environment (important for pyFAI-drawmask to find the right fabio installation)
    path = shutil.which("pyFAI-drawmask")
    os.system(f'"{sys.executable}" {path} {edffile}')
    os.remove(edffile)

def detect_edge_angle_hough(edge_data, sigma=1, erosion_px=10,
                            num_peaks=5, plot=False):
    """
    Detect the dominant straight edge in an image using the Hough transform.

    The pipeline is: normalise → erode NaN mask → Canny edge detection →
    standard Hough transform (0.05° angular resolution) → extract dominant peak.

    Parameters
    ----------
    edge_data : ndarray
        2D image, possibly with NaN pixels marking invalid regions.
    sigma : float, optional
        Gaussian smoothing sigma passed to the Canny detector. Default is 1.
        Use 1–2 for quasi-binary (beamstop/background) images.
    erosion_px : int, optional
        Number of pixels to erode from the border of the valid mask before
        running Canny, to avoid false edges at mask boundaries. Default is 10.
    num_peaks : int, optional
        Maximum number of peaks to extract from the Hough accumulator.
        Only the strongest peak is used. Default is 5.
    plot : bool, optional
        If ``True``, display diagnostic plots of the masked image, Canny edges,
        and the Hough accumulator. Default is ``False``.

    Returns
    -------
    line_angle_rad : float
        Angle of the detected edge line with respect to the horizontal, in radians.
    line_angle_deg : float
        Same angle in degrees.
    edge_point : tuple of float
        ``(x, y)`` coordinates of the point on the line at mid-image height.
    edge_line : tuple
        ``(theta, rho, line_angle_deg)`` — Hough normal angle (rad), signed
        distance from origin (px), and line angle (deg).
    """
    arr = edge_data.astype(float)
    valid = ~np.isnan(arr)

    # Normalise to [0, 1]
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    arr_norm = (arr - vmin) / (vmax - vmin + 1e-12)

    # Erosion: remove border pixels of the NaN mask
    valid_eroded = binary_erosion(valid, iterations=erosion_px)

    # Apply mask: pixels outside eroded region → 0
    arr_masked = np.where(valid_eroded, arr_norm, 0.0)

    # Canny edge detection (normalised, masked image)
    # low_threshold / high_threshold: adjust to SNR
    edge_map = canny(arr_masked, sigma=sigma,
                     low_threshold=0.1, high_threshold=0.3,
                     mask=valid_eroded)

       

    # Standard Hough transform
    # tested_angles: angular resolution — 3600 pts = 0.05° precision
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 3600, endpoint=False)
    h, theta, d = hough_line(edge_map, theta=tested_angles)

    # Extract peaks
    _, peak_angles, peak_dists = hough_line_peaks(
        h, theta, d,
        num_peaks=num_peaks,
        threshold=0.3 * h.max()   # ignore weak peaks
    )

    if len(peak_angles) == 0:
        print("[WARN] No Hough peak detected.")
        return 0.0, 0.0, None, None

    theta = peak_angles[0]   # normal to the line
    rho   = peak_dists[0]    # signed distance from origin to line

    # Line angle (convention: angle w.r.t. horizontal)
    line_angle_rad = theta + np.pi / 2
    line_angle_rad = (line_angle_rad + np.pi / 2) % np.pi - np.pi / 2
    line_angle_deg = np.degrees(line_angle_rad)

    # ----------------------------------------------------------------
    # Geometric reconstruction of the line from (theta, rho)
    # Equation: x*cos(theta) + y*sin(theta) = rho
    # ----------------------------------------------------------------
    ny, nx = edge_data.shape
    x0_img = nx / 2.0   # image centre (origin of Hough frame if skimage
    y0_img = ny / 2.0   # native frame is used, i.e. corner (0,0))

    # Point on the line at y = image centre
    # → solve: x*cos(theta) + y_mid*sin(theta) = rho
    y_mid = ny / 2.0
    if np.abs(np.cos(theta)) > 1e-6:
        x_at_ymid = (rho - y_mid * np.sin(theta)) / np.cos(theta)
    else:
        x_at_ymid = rho / (np.cos(theta) + 1e-12)   # near-horizontal line

    # Point on the line at x = image centre
    x_mid = nx / 2.0
    if np.abs(np.sin(theta)) > 1e-6:
        y_at_xmid = (rho - x_mid * np.cos(theta)) / np.sin(theta)
    else:
        y_at_xmid = rho / (np.sin(theta) + 1e-12)

    # Returned parameters summary
    edge_point = (x_at_ymid, y_mid)          # a point on the line
    edge_line  = (theta, rho, line_angle_deg) # (normal, distance, line angle in °)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Masked image
        axes[0].imshow(arr_masked, cmap='gray')
        axes[0].set_title(f'Masked image (erosion {erosion_px}px)')

        # Canny edge map
        axes[1].imshow(edge_map, cmap='gray')
        axes[1].set_title(f'Canny edges ({edge_map.sum()} px)')

        # Hough accumulator
        axes[2].imshow(
            np.log(1 + h),
            extent=[np.degrees(theta[0]), np.degrees(theta[-1]),
                    d[-1], d[0]],
            aspect='auto', cmap='hot'
        )
        axes[2].set_xlabel('θ (degrees)')
        axes[2].set_ylabel('ρ (pixels)')
        axes[2].set_title('Hough accumulator (log)')
        # Mark peaks
        for a, dist in zip(peak_angles, peak_dists):
            axes[2].plot(np.degrees(a), dist, 'c+', ms=10, mew=2)

        # Overlay detected line on image
        axes[1].set_title(f'Canny + detected edge ({line_angle_deg:.2f}°)')
        h_img, w_img = edge_map.shape
        angle = peak_angles[0]
        rho = peak_dists[0]
        if np.abs(np.sin(angle)) > 1e-6:
            x_vals = np.array([0, w_img])
            y_vals = (rho - x_vals * np.cos(angle)) / np.sin(angle)
        else:
            x_vals = np.array([rho, rho])
            y_vals = np.array([0, h_img])
        axes[1].plot(x_vals, y_vals, 'r-', lw=2,
                     label=f'{line_angle_deg:.2f}°')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return line_angle_rad, line_angle_deg, edge_point, edge_line

def compute_mtf_slanted_edge(image_path,
                             mask=None,
                             pixel_size=None,
                             binning_factor=1,
                             roi_half_width=15,
                             nbins=500,
                             smooth_sigma=0.5,
                             use_erf_fit=True,
                             plot=True,
                             outputfile=None):
    """
    Compute the MTF using the slanted-edge method, with automatic edge
    angle and position detection via Hough transform.

    Parameters
    ----------
    image_path     : str   - Path to the image file.
    mask           : str   - Path to a fabio mask file (0=valid, 1=masked).
    pixel_size     : float - Pixel size in µm.
    binning_factor : int   - Binning factor applied to the detector (default 1).
    roi_half_width : int   - Half-width of the band around the edge (pixels).
    nbins          : int   - Number of sub-pixel bins for the ESF.
    smooth_sigma   : float - Sigma of the Gaussian smoothing applied to the ESF.
    use_erf_fit    : bool  - Fit the ESF with an error function before differentiation.
    plot           : bool  - Display diagnostic plots.
    outputfile     : str   - If provided, save the MTF to this text file.

    Returns
    -------
    freq_pixel : 1D array - Spatial frequencies (cycles/pixel)
    mtf        : 1D array - Corresponding MTF values
    """
    # ------------------------------------------------------------------
    # 1. Load image and mask
    # ------------------------------------------------------------------
    detector_info, image = load_data(image_path, normalize=False, verbose=False)
    if pixel_size is None:
        pixel_size = detector_info.get('pixel_size', None)
        if pixel_size is None:
            raise ValueError("Pixel size not found in metadata.")
    pixel_size = pixel_size * binning_factor

    if mask is not None:
        import fabio
        maskdata = fabio.open(mask).data
        image = image.astype(float)
        image[maskdata != 0] = np.nan

    # ------------------------------------------------------------------
    # 2. Detect edge angle and position (single Hough call)
    # ------------------------------------------------------------------
    edge_angle_rad, edge_angle_deg, edge_point, edge_line = detect_edge_angle_hough(
        image, plot=False
    )
    theta_hough, rho_hough, _ = edge_line
    x_edge_at_ymid = edge_point[0]   # x position of the edge at mid-height (info/debug)

    print(f"[INFO] Edge detected: angle={edge_angle_deg:.2f}°, "
          f"rho={rho_hough:.1f} px, x_edge≈{x_edge_at_ymid:.1f} px")

    # ------------------------------------------------------------------
    # 3. Signed distance of each pixel to the Hough line
    #    Line equation: x·cos(θ) + y·sin(θ) = ρ
    #    → signed distance: d(x,y) = x·cos(θ) + y·sin(θ) − ρ
    #    (sign encodes which side of the edge the pixel lies on)
    # ------------------------------------------------------------------
    
    ny, nx = image.shape
    y_idx, x_idx = np.indices((ny, nx))

    d_raw    = x_idx * np.cos(theta_hough) + y_idx * np.sin(theta_hough) - rho_hough
    d_offset = (x_edge_at_ymid * np.cos(theta_hough)
                + (ny / 2.0)   * np.sin(theta_hough)
                - rho_hough)
    d = d_raw - d_offset
    # ------------------------------------------------------------------
    # 3b. Adapt ROI bounds independently on each side of the edge.
    #     No symmetry is required: the ESF is normalised to [0,1] so
    #     each side only needs enough pixels to establish its plateau.
    #     The beamstop side can be much narrower than the bright side.
    # ------------------------------------------------------------------
    valid = ~np.isnan(image)

    # Maximum available distance on each side within the valid mask
    d_pos_max = d[valid & (d > 0)].max() if (valid & (d > 0)).any() else roi_half_width
    d_neg_max = np.abs(d[valid & (d < 0)].min()) if (valid & (d < 0)).any() else roi_half_width

    # Independent limits: use as much as available up to roi_half_width
    d_pos_lim = min(roi_half_width, d_pos_max)   # bright side
    d_neg_lim = min(roi_half_width, d_neg_max)   # dark (beamstop) side

    print(f"[INFO] Asymmetric ROI: dark side={d_neg_lim:.1f} px, "
          f"bright side={d_pos_lim:.1f} px "
          f"(available: left={d_neg_max:.1f}, right={d_pos_max:.1f})")

    # ------------------------------------------------------------------
    # 4. Select pixels inside the asymmetric ROI band around the edge
    # ------------------------------------------------------------------
    valid = ~np.isnan(image)
    roi = (d > -d_neg_lim) & (d < d_pos_lim)
    valid_roi = valid & roi

    d_vals = d[valid_roi]
    i_vals = image[valid_roi].astype(float)

    if len(d_vals) < 100:
        raise ValueError("Too few valid pixels in ROI. "
                         "Check the mask or increase roi_half_width.")

    # ------------------------------------------------------------------
    # 5. Sub-pixel binning → ESF
    # ------------------------------------------------------------------
    d_min, d_max = d_vals.min(), d_vals.max()
    bins        = np.linspace(d_min, d_max, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    esf_sum    = np.zeros(nbins)
    esf_counts = np.zeros(nbins)
    bin_idx    = np.clip(np.digitize(d_vals, bins) - 1, 0, nbins - 1)

    np.add.at(esf_sum,    bin_idx, i_vals)
    np.add.at(esf_counts, bin_idx, 1)

    valid_bins = esf_counts > 0
    x_esf      = bin_centers[valid_bins]
    esf        = esf_sum[valid_bins] / esf_counts[valid_bins]

    if len(esf) < 10:
        raise ValueError("ESF too short after binning. "
                         "Increase nbins or roi_half_width.")

    # ------------------------------------------------------------------
    # 6. Normalise ESF to [0, 1] and enforce rising orientation
    # ------------------------------------------------------------------
    esf_min, esf_max = np.nanmin(esf), np.nanmax(esf)
    esf_norm = (esf - esf_min) / (esf_max - esf_min + 1e-12)

    if esf_norm[0] > esf_norm[-1]:
        esf_norm = esf_norm[::-1]
        x_esf    = x_esf[::-1]

    # ------------------------------------------------------------------
    # 7. Optional erf fit → regularised ESF on a uniform grid
    # ------------------------------------------------------------------
    if use_erf_fit:
        def erf_model(x, x0, sigma, a, b):
            """Generalised error function with free amplitude and offset."""
            return a * 0.5 * (1 + erf((x - x0) / (np.sqrt(2) * sigma))) + b

        try:
            p0 = [np.median(x_esf), 1.0, 1.0, 0.0]
            popt, _ = curve_fit(erf_model, x_esf, esf_norm,
                                p0=p0, maxfev=5000)
            x_fit    = np.linspace(x_esf.min(), x_esf.max(), nbins)
            esf_fit  = erf_model(x_fit, *popt)
            esf_fit  = (esf_fit - esf_fit.min()) / (esf_fit.max() - esf_fit.min() + 1e-12)
            x_esf    = x_fit
            esf_norm = esf_fit
            print(f"[INFO] erf fit: x0={popt[0]:.2f} px, sigma={popt[1]:.3f} px")
        except Exception as e:
            print(f"[WARN] erf fit failed ({e}), continuing without fit.")

    # Light Gaussian smoothing
    # Light Gaussian smoothing (skip if sigma == 0)
    esf_smooth = gaussian_filter1d(esf_norm, sigma=smooth_sigma) if smooth_sigma > 0 else esf_norm.copy()

    # ------------------------------------------------------------------
    # 8. LSF = derivative of the ESF
    #    dx is the sub-pixel geometric step (used for np.gradient only)
    # ------------------------------------------------------------------
    dx  = np.abs(np.mean(np.diff(x_esf)))   # sub-pixel step in pixels, always > 0
    lsf = np.gradient(esf_smooth, dx)

    # Hanning window to suppress spectral leakage
    window = np.hanning(len(lsf))
    lsf   *= window

    # Normalise so that the area under the LSF equals 1
    lsf_sum = np.sum(np.abs(lsf))
    if lsf_sum > 0:
        lsf /= lsf_sum

    # Centre the LSF peak to avoid FFT phase artefacts
    peak_idx     = np.argmax(lsf)
    shift        = len(lsf) // 2 - peak_idx
    lsf_centered = np.roll(lsf, shift)

    # ------------------------------------------------------------------
    # 8b. Resample LSF onto a 1-pixel grid before FFT
    #     dx < 1 px (sub-pixel binning) would push the Nyquist frequency
    #     above 0.5 cyc/px, which is unphysical.
    #     We interpolate the LSF onto a regular 1-pixel grid so that
    #     freq_pixel is correctly bounded to [0, 0.5] cycles/pixel.
    # ------------------------------------------------------------------
    x_lsf_subpix = np.arange(len(lsf_centered)) * dx   # sub-pixel axis (pixels)
    x_lsf_1px    = np.arange(x_lsf_subpix[0],
                              x_lsf_subpix[-1], 1.0)    # 1-pixel-step grid
    lsf_1px      = np.interp(x_lsf_1px, x_lsf_subpix, lsf_centered)

    # Re-normalise after resampling
    lsf_sum = np.sum(np.abs(lsf_1px))
    if lsf_sum > 0:
        lsf_1px /= lsf_sum

    # ------------------------------------------------------------------
    # 9. FFT → MTF  (on the 1-pixel-grid LSF)
    # ------------------------------------------------------------------
    mtf_complex = np.fft.fft(lsf_1px)
    mtf         = np.abs(mtf_complex)
    mtf        /= mtf[0]                     # normalise to 1 at f = 0

    n_half     = len(mtf) // 2
    mtf        = mtf[:n_half]
    freq_pixel = np.fft.fftfreq(len(lsf_1px), d=1.0)[:n_half]  # 0 → 0.5 cyc/px

    # Physical and normalised frequencies
    freq_phys  = freq_pixel / pixel_size      # cycles/µm
    fnyq_phys  = 1.0 / (2.0 * pixel_size)    # Nyquist frequency in cycles/µm
    freq_norm  = freq_phys / fnyq_phys        # normalised to Nyquist

    # MTF50 and MTF20
    mtf50_idx = np.argmin(np.abs(mtf - 0.5))
    mtf20_idx = np.argmin(np.abs(mtf - 0.2))
    print(f"MTF50: {freq_norm[mtf50_idx]:.3f} f_Nyq  "
          f"({freq_phys[mtf50_idx]:.3f} µm⁻¹)")
    print(f"MTF20: {freq_norm[mtf20_idx]:.3f} f_Nyq  "
          f"({freq_phys[mtf20_idx]:.3f} µm⁻¹)")
    
    # Determine Wiener epsilon from the noise level in the ESF tail (where signal is flat)
    signal_patch, noise_patch = extract_noise_and_signal_patches(image, edge_line)
    wiener_epsilon = estimate_wiener_epsilon_spectral(noise_patch, signal_patch)
    print(f"Estimated Wiener epsilon (noise/signal ratio): {wiener_epsilon:.4f}")

    # ------------------------------------------------------------------
    # 10. Diagnostic plots
    # ------------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- Image with detected edge line and ROI band ---
        axes[0, 0].imshow(image, cmap='gray', origin='upper')
        if np.abs(np.sin(theta_hough)) > 1e-6:
            x_line = np.array([0, nx - 1])
            y_line = (rho_hough - x_line * np.cos(theta_hough)) / np.sin(theta_hough)
        else:
            x_line = np.array([rho_hough, rho_hough])
            y_line = np.array([0, ny - 1])
        axes[0, 0].plot(x_line, y_line, 'r-', lw=2,
                        label=f'Edge {edge_angle_deg:.2f}°')
        axes[0, 0].contour((d > -d_neg_lim) & (d < d_pos_lim), levels=[0.5],
                           colors='cyan', linewidths=1, linestyles='--')
        axes[0, 0].set_title('Image + detected edge (red) + ROI (cyan)')
        axes[0, 0].legend(fontsize=8)

        # --- ESF ---
        axes[0, 1].plot(x_esf, esf_norm, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Distance to edge (pixels)')
        axes[0, 1].set_ylabel('Normalised intensity')
        axes[0, 1].set_title('Edge Spread Function (ESF)')
        axes[0, 1].grid(True, alpha=0.3)

        # --- LSF (resampled at 1 px for consistency with MTF) ---
        axes[1, 0].plot(x_lsf_1px, lsf_1px, 'g-', linewidth=2)
        axes[1, 0].set_title('Line Spread Function (LSF, 1-px grid)')
        axes[1, 0].set_xlabel('Position (pixels)')
        axes[1, 0].grid(True, alpha=0.3)

        # --- MTF (normalised frequency axis) ---
        axes[1, 1].plot(freq_pixel, mtf, 'r-', linewidth=2, label='Measured MTF')
        axes[1, 1].set_xlabel('Spatial frequency (cycles/pixel)')
        axes[1, 1].set_ylabel('MTF')
        axes[1, 1].set_xlim(0, 0.5)
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Modulation Transfer Function (MTF)')
        secax = axes[1, 1].secondary_xaxis(
            'top',
            functions=(lambda f: f * fnyq_phys, lambda f: f / fnyq_phys)
        )
        secax.set_xlabel('Spatial frequency (µm⁻¹)')

        plt.tight_layout()
        plt.savefig('debug_mtf_slanted.png', dpi=100)
        plt.show()

    # ------------------------------------------------------------------
    # 11. Optional output file
    # ------------------------------------------------------------------
    if outputfile is not None:
        header = ("# MTF computed from slanted-edge image\n"
                  "# Col 1: spatial frequency (cycles/pixel)\n"
                  "# Col 2: MTF\n"
                  "# Col 3: Wiener epsilon (noise/signal ratio) used for deconvolution\n")
        np.savetxt(outputfile,
                   np.column_stack((freq_pixel, mtf, np.full_like(freq_pixel, wiener_epsilon))),
                   header=header, comments='')
        print(f"MTF saved: {outputfile}")

    return freq_pixel, mtf

def estimate_wiener_epsilon_spectral(noise_patch, signal_patch, subtract_noise=True):
    """
    Estimate the Wiener regularisation parameter epsilon from image data.

    Computes epsilon as the square root of the ratio of the mean power spectral
    densities (PSD) of the noise and signal patches:
    ``epsilon = sqrt( <|N(f)|²> / <|S(f)|²> )``

    This estimate is used to set the noise-to-signal power ratio in the Wiener
    filter: ``W(f) = MTF / (MTF² + epsilon²)``.

    Parameters
    ----------
    noise_patch : ndarray
        2D (or 1D) sub-image extracted from the beamstop region (dark, noisy side).
    signal_patch : ndarray
        2D (or 1D) sub-image extracted from the bright background region.
    subtract_noise : bool, optional
        If ``True`` (default), subtract the mean of ``noise_patch`` from
        ``signal_patch`` before computing the signal PSD, to account for
        any DC offset in the background.

    Returns
    -------
    epsilon : float
        Estimated noise-to-signal PSD ratio, suitable for use as ``wiener_epsilon``
        in :func:`deconvolve_mtf_2d`.
    """
    # Centre signals
    noise = noise_patch - np.nanmean(noise_patch)
    if subtract_noise:
        signal = signal_patch - np.nanmean(noise_patch)
    else:
        signal = signal_patch - np.nanmean(signal_patch)
    # Prepare for FFT (fill NaN with 0)
    noise_f = np.nan_to_num(noise, nan=0.0)
    signal_f = np.nan_to_num(signal, nan=0.0)
    # Compute PSD (FFT²), 1D or 2D depending on shape
    if noise_f.ndim == 1:
        noise_psd = np.abs(np.fft.fftshift(np.fft.fft(noise_f)))**2
    else:
        noise_psd = np.abs(np.fft.fftshift(np.fft.fft2(noise_f)))**2
    if signal_f.ndim == 1:
        signal_psd = np.abs(np.fft.fftshift(np.fft.fft(signal_f)))**2
    else:
        signal_psd = np.abs(np.fft.fftshift(np.fft.fft2(signal_f)))**2
    # Average PSD
    mean_noise_psd = np.mean(noise_psd)
    mean_signal_psd = np.mean(signal_psd)
    # Noise/signal PSD ratio
    epsilon = np.sqrt(mean_noise_psd / (mean_signal_psd + 1e-12))
    return epsilon


def extract_noise_and_signal_patches(image, edge_line, band_width=500, noise_box=None, erosion_px=5):
    """
    Extract noise and signal pixel patches on each side of the detected edge.

    The image is split along the Hough line into two regions:
    - **signal patch** (bright side, ``d > +erosion_px``): background pixels.
    - **noise patch** (dark side, ``d < -erosion_px``): beamstop pixels.

    An erosion band of ``erosion_px`` pixels around the edge is excluded from
    both patches to avoid contamination by the edge transition itself.
    Diagnostic plots are displayed showing the two zones.

    Parameters
    ----------
    image : ndarray
        2D image, with NaN for masked/invalid pixels.
    edge_line : tuple
        ``(theta, rho, angle_deg)`` as returned by :func:`detect_edge_angle_hough`.
    band_width : float, optional
        Total width of the extraction band centred on the edge (pixels). Default is 500.
    noise_box : ignored
        Reserved for future use.
    erosion_px : int, optional
        Width of the exclusion zone on each side of the edge (pixels). Default is 10.

    Returns
    -------
    signal_patch : ndarray
        1D array of pixel values from the bright (background) side.
    noise_patch : ndarray
        1D array of pixel values from the dark (beamstop) side.
    """
    theta, rho, _ = edge_line
    ny, nx = image.shape
    y_idx, x_idx = np.indices((ny, nx))
    # Signed distance to the Hough line (same convention as in the plot)
    d = x_idx * np.cos(theta) + y_idx * np.sin(theta) - rho
    # Build masks for each side of the edge, with erosion
    # Convention: d < 0 = beamstop side (shadow, noise), d > 0 = background side (signal)
    mask_band = np.abs(d) < (band_width / 2)
    valid = ~np.isnan(image)
    # Erosion: exclude a band of +/- erosion_px around the edge
    mask_signal = (d > +erosion_px) & mask_band & valid  # background only, distance > erosion_px
    mask_noise  = (d < -erosion_px) & mask_band & valid  # beamstop only, distance > erosion_px
    signal_patch = image[mask_signal]
    noise_patch  = image[mask_noise]
    n_signal = signal_patch.size
    n_noise = noise_patch.size
    print(f"Signal patch (background, eroded {erosion_px}px): {n_signal} pixels, "
          f"Noise patch (beamstop, eroded {erosion_px}px): {n_noise} pixels")

    # Overlay on original image
    plt.figure(figsize=(7, 7))
    img_disp = np.copy(image)
    img_disp = np.where(np.isnan(img_disp), np.nanmedian(img_disp), img_disp)
    plt.imshow(img_disp, cmap='gray', origin='upper')
    # Overlay noise band (beamstop) in red
    mask_noise_disp = np.zeros_like(image, dtype=float)
    mask_noise_disp[mask_noise] = 1.0
    plt.contour(mask_noise_disp, levels=[0.5], colors='red', linewidths=2, linestyles='-', label='Noise (beamstop)')
    # Overlay signal band (background) in blue
    mask_signal_disp = np.zeros_like(image, dtype=float)
    mask_signal_disp[mask_signal] = 1.0
    plt.contour(mask_signal_disp, levels=[0.5], colors='blue', linewidths=1, linestyles='--', label='Signal (background)')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Noise (beamstop)'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Signal (background)')
    ]
    plt.title(f'Extracted zones\nNoise (beamstop, red), Signal (background, blue)\n'
              f'Signal: {n_signal} px, Noise: {n_noise} px\nErosion: {erosion_px} px')
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.show()

    # 1D patch values (sanity check)
    plt.figure(figsize=(8, 3))
    plt.plot(noise_patch, '.', color='red', alpha=0.7, label='Noise (beamstop)')
    plt.plot(signal_patch, '.', color='blue', alpha=0.5, label='Signal')
    plt.title('Extracted values (1D)')
    plt.xlabel('Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return signal_patch, noise_patch
#-----------------------------------------------------------------------
# Wiener 2D deconvolution with radial MTF
#-----------------------------------------------------------------------

def deconvolve_mtf_2d(image, mtf_file, clip=True,
                       wiener_epsilon=None,
                       min_epsilon=0.005,
                       pre_smooth_sigma=0.5,
                       use_rolloff=True,
                       u_cutoff=0.4,
                       rolloff_window='tukey',
                       rolloff_alpha=0.5,
                       rolloff_order=4,
                       plot=False):
    """
    Wiener 2D MTF deconvolution with optional high-frequency roll-off.

    Applies a Wiener filter built from a radially symmetric MTF to restore
    spatial frequencies attenuated by the detector. An optional roll-off
    window suppresses noise amplification at high frequencies.

    Parameters
    ----------
    image : ndarray
        2D image to deconvolve.
    mtf_file : str
        Path to the MTF file (3-column text: freq (cyc/px), MTF, epsilon).
    clip : bool, optional
        If ``True`` (default), clip negative values in the output to zero.
    wiener_epsilon : float or None, optional
        Regularisation parameter. If ``None``, read from column 3 of
        ``mtf_file`` (floored at ``min_epsilon``).
    min_epsilon : float, optional
        Minimum allowed epsilon to prevent filter instability. Default is 0.005.
    pre_smooth_sigma : float, optional
        Sigma (pixels) of Gaussian pre-smoothing applied before deconvolution
        to reduce Poisson noise amplification. Default is 0.5. Set to 0 to disable.
    use_rolloff : bool, optional
        If ``True`` (default), multiply the Wiener filter by a roll-off window
        to suppress noise at frequencies above ``u_cutoff``.
    u_cutoff : float or None, optional
        Roll-off cutoff frequency in cycles/pixel (max 0.5 = Nyquist). If
        ``None``, automatically set to the frequency where ``MTF = epsilon``.
        Default is 0.4.
    rolloff_window : {'tukey', 'hann', 'butterworth'}, optional
        Shape of the roll-off window. Default is ``'tukey'``.
    rolloff_alpha : float, optional
        For the Tukey window: fraction of the passband that is flat
        (0 = Hann, 1 = rectangular). Default is 0.5.
    rolloff_order : int, optional
        For the Butterworth window: filter order (higher = steeper). Default is 4.
    plot : bool, optional
        If ``True``, display the Wiener filter profile. Default is ``False``.

    Returns
    -------
    image_deconv : ndarray
        Deconvolved image, same shape as ``image``. NaN pixels are preserved.
    """

    # ------------------------------------------------------------------
    # 1. Load MTF
    # ------------------------------------------------------------------
    mtf_data = np.loadtxt(mtf_file, comments='#')
    if mtf_data.ndim != 2 or mtf_data.shape[1] < 2:
        raise ValueError("MTF file must have 2 columns: freq (cyc/px) and MTF.")
    freq_1d = mtf_data[:, 0]
    mtf_1d  = mtf_data[:, 1]
    if wiener_epsilon is None:
        wiener_epsilon = max(mtf_data[0, 2], min_epsilon)
    if freq_1d[0] > 0:
        freq_1d = np.concatenate([[0.0], freq_1d])
        mtf_1d  = np.concatenate([[1.0], mtf_1d])

    # ------------------------------------------------------------------
    # 2. Handle NaNs
    # ------------------------------------------------------------------
    nan_mask = np.isnan(image)
    image_filled = image.copy().astype(float)
    if nan_mask.any():
        image_filled[nan_mask] = np.nanmean(image)

    # ------------------------------------------------------------------
    # 3. Optional pre-smoothing to reduce Poisson noise before deconv
    #    Acts as a noise regulariser without affecting the MTF correction
    # ------------------------------------------------------------------
    if pre_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        image_filled = gaussian_filter(image_filled, sigma=pre_smooth_sigma)
        print(f"[INFO] Pre-smoothing applied: sigma={pre_smooth_sigma} px")

    # ------------------------------------------------------------------
    # 4. Build 2D radial frequency grid
    # ------------------------------------------------------------------
    ny, nx = image_filled.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    freq_radial = np.sqrt(FX**2 + FY**2)

    # ------------------------------------------------------------------
    # 5. Interpolate MTF onto 2D grid
    # ------------------------------------------------------------------
    mtf_2d = np.interp(freq_radial, freq_1d, mtf_1d, left=1.0, right=0.0)

    # ------------------------------------------------------------------
    # 6. Wiener filter normalised to W(0)=1
    # ------------------------------------------------------------------
    wiener_filter = mtf_2d / (mtf_2d**2 + wiener_epsilon**2)
    mtf_at_zero   = np.interp(0.0, freq_1d, mtf_1d)
    w_at_zero     = mtf_at_zero / (mtf_at_zero**2 + wiener_epsilon**2)
    wiener_filter /= w_at_zero

    #print(f"[INFO] Wiener filter: max={wiener_filter.max():.2f}, "
    #      f"epsilon={wiener_epsilon}, W(0)={w_at_zero:.6f}")

    # ------------------------------------------------------------------
    # 7. Optional roll-off window
    # ------------------------------------------------------------------
    if use_rolloff:
        # Auto u_cutoff: frequency where MTF(u) = epsilon → amplification ~0.5/epsilon
        # beyond this point, the filter significantly amplifies noise
        if u_cutoff is None:
            mtf_epsilon_idx = np.argmin(np.abs(mtf_1d - wiener_epsilon))
            u_cutoff = freq_1d[mtf_epsilon_idx]
            u_cutoff = np.clip(u_cutoff, 0.1, 0.5)
            print(f"[INFO] Auto u_cutoff = {u_cutoff:.3f} cyc/px "
                  f"(MTF = epsilon = {wiener_epsilon:.4f})")

        u = freq_radial / u_cutoff

        if rolloff_window == 'hann':
            R = np.where(u <= 1.0,
                         0.5 * (1.0 + np.cos(np.pi * u)),
                         0.0)

        elif rolloff_window == 'butterworth':
            R = 1.0 / (1.0 + u ** (2 * rolloff_order))

        elif rolloff_window == 'tukey':
            R = np.ones_like(u)
            mask_rolloff = (u >= rolloff_alpha) & (u <= 1.0)
            mask_zero    = u > 1.0
            R[mask_rolloff] = 0.5 * (1.0 + np.cos(
                np.pi * (u[mask_rolloff] - rolloff_alpha) / (1.0 - rolloff_alpha)
            ))
            R[mask_zero] = 0.0

        else:
            raise ValueError(f"Unknown rolloff_window: '{rolloff_window}'. "
                             f"Choose 'hann', 'butterworth', or 'tukey'.")

        wiener_filter *= R
        #print(f"[INFO] Roll-off applied: window={rolloff_window}, "
        #      f"u_cutoff={u_cutoff:.3f} cyc/px, "
        #      f"alpha={rolloff_alpha if rolloff_window == 'tukey' else 'N/A'}")

    # ------------------------------------------------------------------
    # 8. Diagnostic plot
    # ------------------------------------------------------------------
    if plot:
        f_plot   = np.linspace(0, 0.5, 300)
        mtf_plot = np.interp(f_plot, freq_1d, mtf_1d)
        W_plot   = mtf_plot / (mtf_plot**2 + wiener_epsilon**2)
        W_plot  /= W_plot[0]

        plt.figure(figsize=(8, 4))
        plt.plot(f_plot, mtf_plot, 'b-',  lw=1.5, label='MTF')
        plt.plot(f_plot, W_plot,   'r--', lw=1.5, label=f'Wiener only (ε={wiener_epsilon})')

        if use_rolloff:
            u_plot = f_plot / u_cutoff
            if rolloff_window == 'hann':
                R_plot = np.where(u_plot <= 1.0,
                                  0.5 * (1.0 + np.cos(np.pi * u_plot)), 0.0)
            elif rolloff_window == 'butterworth':
                R_plot = 1.0 / (1.0 + u_plot ** (2 * rolloff_order))
            elif rolloff_window == 'tukey':
                R_plot = np.ones_like(u_plot)
                m_r = (u_plot >= rolloff_alpha) & (u_plot <= 1.0)
                m_z = u_plot > 1.0
                R_plot[m_r] = 0.5 * (1.0 + np.cos(
                    np.pi * (u_plot[m_r] - rolloff_alpha) / (1.0 - rolloff_alpha)
                ))
                R_plot[m_z] = 0.0
            plt.plot(f_plot, R_plot,          'g--', lw=1.5, label=f'Roll-off ({rolloff_window})')
            plt.plot(f_plot, W_plot * R_plot, 'k-',  lw=2.0, label='Wiener × roll-off')
            plt.axvline(u_cutoff, color='gray', linestyle=':', alpha=0.6,
                        label=f'u_cutoff={u_cutoff:.3f}')

        plt.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
        plt.axvline(0.5, color='gray', linestyle=':', alpha=0.4, label='Nyquist')
        plt.xlabel('Spatial frequency (cycles/pixel)')
        plt.ylabel('Amplitude')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.title('Wiener deconvolution filter')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 9. FFT → deconvolution → IFFT
    # ------------------------------------------------------------------
    image_fft        = np.fft.fft2(image_filled)
    image_fft_deconv = image_fft * wiener_filter
    image_deconv     = np.real(np.fft.ifft2(image_fft_deconv))

    # ------------------------------------------------------------------
    # 10. Restore NaNs and clip
    # ------------------------------------------------------------------
    if nan_mask.any():
        image_deconv[nan_mask] = np.nan
    if clip:
        image_deconv = np.clip(image_deconv, 0, None)

    #print(f"[INFO] Done. Input range:  [{np.nanmin(image):.1f}, {np.nanmax(image):.1f}]")
    #print(f"       Output range: [{np.nanmin(image_deconv):.1f}, {np.nanmax(image_deconv):.1f}]")

    return image_deconv



## Function to compute DQE from flat and dark images, and MTF file


def compute_dqe(flat_paths, dark_paths, mtf_file,
                gain_reference=None,
                n_freq=128,
                plot=False,
                save=None):
    """
    Compute the radially-averaged DQE from flat-field and dark-field images.

    The DQE is defined as:

    .. math::

        \\mathrm{DQE}(f) = \\frac{\\mathrm{MTF}^2(f)}{\\bar{n} \\cdot \\mathrm{NPS}(f)}

    where :math:`\\bar{n}` is the mean number of electrons per pixel (signal
    level) and :math:`\\mathrm{NPS}(f)` is the normalised noise power spectrum:

    .. math::

        \\mathrm{NPS}(f) = \\frac{1}{N_{\\mathrm{img}}\\, N_x N_y\\, \\bar{n}^2}
        \\sum_k \\left| \\mathcal{F}\\!\\left[ I_k - \\bar{I} \\right](f) \\right|^2

    The dark-field mean is subtracted from each flat-field image before
    computing the NPS, so that the detector read-noise is excluded from
    :math:`\\bar{n}` but its contribution to the NPS is correctly accounted for.

    Parameters
    ----------
    flat_paths : list of str
        Paths to the flat-field (uniform illumination) images.  At least 5
        images are recommended for a stable NPS estimate; 20–50 are ideal.
    dark_paths : list of str
        Paths to the dark-field (shutter closed) images.  Used to estimate
        and subtract the detector dark offset.
    mtf_file : str
        Path to the MTF file (columns: frequency in cyc/px, MTF value), as
        produced by :func:`compute_mtf_slanted_edge`.
    gain_reference : ndarray or None, optional
        2D gain reference map (same shape as the images).  When provided,
        each flat-field image is divided by ``gain_reference`` before
        computing the NPS to correct for pixel-to-pixel sensitivity
        variations.  ``None`` skips gain correction. Default is ``None``.
    n_freq : int, optional
        Number of radial frequency bins for the azimuthal average.
        Default is 128.
    plot : bool, optional
        If ``True``, display MTF², NPS, and DQE curves. Default is ``False``.
    save : str or None, optional
        If a file path is given, save the result as a two-column text file
        (frequency in cyc/px, DQE value) readable by
        :func:`deconvolve_mtf_2d_rl`. Default is ``None``.

    Returns
    -------
    freq_bins : ndarray, shape (n_freq,)
        Radial frequency axis in cycles/pixel (0 to 0.5).
    dqe : ndarray, shape (n_freq,)
        Radially-averaged DQE, values in [0, 1].

    Notes
    -----
    * All images must have the same shape.
    * Images are expected to be in raw detector counts (electrons or ADU).
    * At very low dose the DQE drops because read noise dominates; at very
      high dose it drops due to detector non-linearity.  Run this function
      at several dose levels to characterise the dose dependence.
    """
    # ------------------------------------------------------------------
    # 1. Load dark images → mean dark frame
    # ------------------------------------------------------------------
    dark_stack = np.array([load_data(p)[1].astype(float) for p in dark_paths])
    dark_mean  = dark_stack.mean(axis=0)

    # ------------------------------------------------------------------
    # 2. Load flat images, subtract dark, optional gain correction
    # ------------------------------------------------------------------
    flat_list = []
    for p in flat_paths:
        img = load_data(p)[1].astype(float) - dark_mean
        if gain_reference is not None:
            img = img / np.where(gain_reference > 0, gain_reference, 1.0)
        flat_list.append(img)

    flat_stack = np.array(flat_list)          # shape (N, ny, nx)
    n_img, ny, nx = flat_stack.shape

    # Mean signal level (electrons/pixel) over all frames and pixels
    n_bar = float(np.mean(flat_stack))
    if n_bar <= 0:
        raise ValueError("Mean flat signal is non-positive after dark subtraction. "
                         "Check dark and flat images.")

    # ------------------------------------------------------------------
    # 3. Compute NPS
    #    NPS(f) = 1/(N * nx * ny * n_bar²) * Σ_k |FFT(I_k - n_bar)|²
    # ------------------------------------------------------------------
    nps_sum = np.zeros((ny, nx), dtype=float)
    for img in flat_stack:
        diff     = img - n_bar
        fft_diff = np.fft.fft2(diff)
        nps_sum += np.abs(fft_diff) ** 2

    nps_2d = nps_sum / (n_img * nx * ny * n_bar ** 2)

    # ------------------------------------------------------------------
    # 4. Load MTF, build 2D MTF map, compute MTF²
    # ------------------------------------------------------------------
    mtf_data = np.loadtxt(mtf_file, comments='#')
    freq_1d  = mtf_data[:, 0]
    mtf_1d   = mtf_data[:, 1]
    if freq_1d[0] > 0:
        freq_1d = np.concatenate([[0.0], freq_1d])
        mtf_1d  = np.concatenate([[1.0], mtf_1d])

    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    freq_radial = np.sqrt(FX**2 + FY**2)
    mtf_2d  = np.interp(freq_radial, freq_1d, mtf_1d, left=1.0, right=0.0)
    mtf2_2d = mtf_2d ** 2

    # ------------------------------------------------------------------
    # 5. DQE 2D = MTF² / (n_bar * NPS)
    # ------------------------------------------------------------------
    dqe_2d = mtf2_2d / (n_bar * np.where(nps_2d > 0, nps_2d, np.inf))
    dqe_2d = np.clip(dqe_2d, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 6. Radial (azimuthal) average
    # ------------------------------------------------------------------
    freq_bins  = np.linspace(0, 0.5, n_freq + 1)
    freq_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])

    # Use fftshift so that freq_radial maps cleanly to positive half-axis
    freq_flat  = np.fft.fftshift(freq_radial).ravel()
    dqe_flat   = np.fft.fftshift(dqe_2d).ravel()
    nps_flat   = np.fft.fftshift(nps_2d).ravel()
    mtf2_flat  = np.fft.fftshift(mtf2_2d).ravel()

    dqe_radial  = np.zeros(n_freq)
    nps_radial  = np.zeros(n_freq)
    mtf2_radial = np.zeros(n_freq)

    for i, (f_lo, f_hi) in enumerate(zip(freq_bins[:-1], freq_bins[1:])):
        mask = (freq_flat >= f_lo) & (freq_flat < f_hi)
        if mask.any():
            dqe_radial[i]  = dqe_flat[mask].mean()
            nps_radial[i]  = nps_flat[mask].mean()
            mtf2_radial[i] = mtf2_flat[mask].mean()

    # ------------------------------------------------------------------
    # 7. Optional save
    # ------------------------------------------------------------------
    if save is not None:
        header = (f"# DQE computed from {n_img} flat / {len(dark_paths)} dark images\n"
                  f"# Mean signal level: {n_bar:.2f} counts/pixel\n"
                  f"# Columns: frequency (cyc/px)   DQE")
        np.savetxt(save,
                   np.column_stack([freq_centers, dqe_radial]),
                   header=header, fmt='%.6f')

    # ------------------------------------------------------------------
    # 8. Optional plot
    # ------------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(freq_centers, mtf2_radial, 'b-', lw=2)
        axes[0].set_title('MTF²')
        axes[0].set_xlabel('Frequency (cyc/px)')
        axes[0].set_ylabel('MTF²')
        axes[0].axvline(0.5, color='gray', ls=':', alpha=0.5, label='Nyquist')
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(freq_centers, nps_radial, 'r-', lw=2)
        axes[1].set_title(f'NPS  (n̄ = {n_bar:.1f} counts/px)')
        axes[1].set_xlabel('Frequency (cyc/px)')
        axes[1].set_ylabel('NPS (normalised)')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(freq_centers, dqe_radial, 'g-', lw=2)
        axes[2].set_title('DQE')
        axes[2].set_xlabel('Frequency (cyc/px)')
        axes[2].set_ylabel('DQE')
        axes[2].set_ylim(0, 1.05)
        axes[2].axvline(0.5, color='gray', ls=':', alpha=0.5, label='Nyquist')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.suptitle(f'DQE measurement  —  {n_img} flat images', y=1.01)
        plt.tight_layout()
        plt.show()

    return freq_centers, dqe_radial



def deconvolve_mtf_dqe_2d(image, mtf_file, dqe_file):
    """
    Deconvolve an image using a DQE-weighted Wiener filter.

    This is a simplified version of :func:`deconvolve_mtf_2d` that applies
    the DQE weighting directly to the Wiener filter without pre-smoothing or
    roll-off.  The filter is:

    .. math::

        W(f) = \\frac{\\mathrm{DQE}(f)}{\\mathrm{MTF}(f)}

   Parameters
    ----------
    image : ndarray
        2D image to deconvolve.
    mtf_file : str
        Path to the MTF file (columns: frequency in cyc/px, MTF value, epsilon).
    dqe_file : str
        Path to the DQE file (columns: frequency in cyc/px, DQE value).

    Returns
    -------
    image_deconv : ndarray
        Deconvolved image, same shape as ``image``. NaN pixels are preserved.
    """
    
    # ------------------------------------------------------------------
    # 1. Load MTF and DQE
    # ------------------------------------------------------------------
    mtf_data = np.loadtxt(mtf_file, comments='#')
    dqe_data = np.loadtxt(dqe_file, comments='#')
    if mtf_data.ndim != 2 or mtf_data.shape[1] < 2:
        raise ValueError("MTF file must have at least 2 columns: freq (cyc/px) and MTF.")
    if dqe_data.ndim != 2 or dqe_data.shape[1] < 2:
        raise ValueError("DQE file must have 2 columns: freq (cyc/px) and DQE.")
    freq_mtf = mtf_data[:, 0]
    mtf_1d   = mtf_data[:, 1]  # Toujours la 2e colonne, même si 3 colonnes
    freq_dqe = dqe_data[:, 0]
    dqe_1d   = dqe_data[:, 1]
    if freq_mtf[0] > 0:
        freq_mtf = np.concatenate([[0.0], freq_mtf])
        mtf_1d   = np.concatenate([[1.0], mtf_1d])
    if freq_dqe[0] > 0:
        freq_dqe = np.concatenate([[0.0], freq_dqe])
        dqe_1d   = np.concatenate([[1.0], dqe_1d])

    # ------------------------------------------------------------------
    # 2. Handle NaNs
    # ------------------------------------------------------------------
    nan_mask = np.isnan(image)
    image_filled = image.copy().astype(float)
    if nan_mask.any():
        image_filled[nan_mask] = np.nanmean(image)

    # ------------------------------------------------------------------
    # 3. Build 2D frequency grid and interpolate MTF/DQE
    # ------------------------------------------------------------------
    ny, nx = image_filled.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    freq_radial = np.sqrt(FX**2 + FY**2)
    mtf_2d = np.interp(freq_radial, freq_mtf, mtf_1d, left=1.0, right=0.0)
    dqe_2d = np.interp(freq_radial, freq_dqe, dqe_1d, left=1.0, right=0.0)

    # ------------------------------------------------------------------
    # 4. Build Wiener filter: W(f) = DQE(f) / MTF(f)
    #    Clamp MTF to avoid division by zero
    # ------------------------------------------------------------------
    mtf_2d_safe = np.where(mtf_2d > 1e-6, mtf_2d, 1e-6)
    wiener_filter = dqe_2d / mtf_2d_safe
    wiener_filter = np.clip(wiener_filter, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 5. FFT, apply filter, IFFT
    # ------------------------------------------------------------------
    image_fft = np.fft.fft2(image_filled)
    image_fft_deconv = image_fft * wiener_filter
    image_deconv = np.real(np.fft.ifft2(image_fft_deconv))

    # ------------------------------------------------------------------
    # 6. Restore NaNs and clip
    # ------------------------------------------------------------------
    if nan_mask.any():
        image_deconv[nan_mask] = np.nan
    image_deconv = np.clip(image_deconv, 0, None)

    return image_deconv


## RL filter for memory
def deconvolve_mtf_2d_rl(image, mtf_file, clip=True,
                          n_iterations=50,
                          tol=1e-2,
                          dqe_file=None,
                          pre_smooth_sigma=0,
                          verbose=False,
                          plot=False):
    """
    Richardson-Lucy 2D deconvolution with a radial MTF.

    Suited to Poisson noise (electron/photon counting). Regularisation is
    implicit: too few iterations under-deconvolves; too many amplify noise.

    The stopping criterion is the relative change of the current estimate *u*:

    .. math::

        \\text{rel} = \\frac{\\|u^{(k+1)} - u^{(k)}\\|_\\infty}{\\|u^{(k)}\\|_\\infty} < \\text{tol}

    **DQE-weighted correction (optional)**

    When ``dqe_file`` is provided, the back-projection step is weighted by the
    2D DQE map instead of the plain MTF conjugate:

    .. math::

        u^{(k+1)} = u^{(k)} \\cdot \\mathcal{F}^{-1}\\!\\left[
            \\mathrm{DQE}(f)\\, H(f)\\,
            \\mathcal{F}\\!\\left[\\frac{I}{h \\circledast u^{(k)}}\\right]
        \\right]

    where :math:`\\mathrm{DQE}(f) = \\mathrm{MTF}^2(f) / (\\bar{n}\\,\\mathrm{NPS}(f))`.
    Frequencies where :math:`\\mathrm{DQE}(f) \\approx 0` (noise-dominated) are
    naturally suppressed at every iteration, making the algorithm less sensitive
    to the choice of ``n_iterations`` and removing the need for
    ``pre_smooth_sigma`` in most cases.

    Without ``dqe_file`` the standard R-L update is used and regularisation
    relies entirely on early stopping via ``tol`` and ``n_iterations``.

    Parameters
    ----------
    image : ndarray
        2D image to deconvolve.
    mtf_file : str
        Path to the MTF file (columns: frequency in cyc/px, MTF value).
    clip : bool, optional
        If ``True``, clamp negative values to 0 after each iteration.
        Default is ``True``.
    n_iterations : int, optional
        Maximum number of iterations (safety cap). Default is 50.
    tol : float or None, optional
        Early-stopping threshold on the relative change ``||Δu||/||u||``.
        ``None`` disables early stopping. Default is ``1e-2``.
    dqe_file : str or None, optional
        Path to the DQE file (same format as ``mtf_file``: columns are
        frequency in cyc/px and DQE value in [0, 1]).  When provided, the
        correction at each iteration is weighted by the 2D DQE map, which
        suppresses noise-dominated frequencies without requiring aggressive
        early stopping or pre-smoothing.  ``None`` disables DQE weighting
        and reproduces the standard R-L behaviour. Default is ``None``.
    pre_smooth_sigma : float, optional
        Standard deviation (pixels) for Gaussian pre-smoothing applied
        before deconvolution. ``0`` disables smoothing. Default is ``0``.
    verbose : bool, optional
        If ``True``, print the relative change at each iteration.
        Default is ``False``.
    plot : bool, optional
        If ``True``, display the PSF profile. Default is ``False``.

    Returns
    -------
    image_deconv : ndarray
        Deconvolved 2D image.
    """
    # ------------------------------------------------------------------
    # 1. Load MTF
    # ------------------------------------------------------------------
    mtf_data = np.loadtxt(mtf_file, comments='#')
    if mtf_data.ndim != 2 or mtf_data.shape[1] < 2:
        raise ValueError("MTF file must have 2 columns: freq (cyc/px) and MTF.")
    freq_1d = mtf_data[:, 0]
    mtf_1d  = mtf_data[:, 1]
    if freq_1d[0] > 0:
        freq_1d = np.concatenate([[0.0], freq_1d])
        mtf_1d  = np.concatenate([[1.0], mtf_1d])

    # RL requires a normalised PSF: H(0) = 1 (total energy is conserved).
    # If MTF(DC) < 1, each iteration multiplies the estimate by MTF(0)^2 < 1,
    # driving it to zero after enough iterations.
    # Normalise here so RL is independent of the absolute MTF calibration.
    _mtf_dc = np.interp(0.0, freq_1d, mtf_1d)
    if _mtf_dc <= 0:
        raise ValueError(
            f"[RL] MTF value at DC = {_mtf_dc:.4g} ≤ 0. "
            "Check that column 1 of the MTF file contains MTF values (not epsilon)."
        )
    if abs(_mtf_dc - 1.0) > 1e-3:
        print(
            f"[RL] MTF(DC) = {_mtf_dc:.4f} ≠ 1.0 — normalising by MTF(DC). "
            "Without this, each RL iteration multiplies the estimate by "
            f"MTF(DC)² = {_mtf_dc**2:.4f}, collapsing the image to zero."
        )
        mtf_1d = mtf_1d / _mtf_dc


    # ------------------------------------------------------------------
    if dqe_file is not None:
        dqe_data = np.loadtxt(dqe_file, comments='#')
        if dqe_data.ndim != 2 or dqe_data.shape[1] < 2:
            raise ValueError("DQE file must have 2 columns: freq (cyc/px) and DQE.")
        dqe_freq_1d = dqe_data[:, 0]
        dqe_1d      = dqe_data[:, 1]
        if dqe_freq_1d[0] > 0:
            dqe_freq_1d = np.concatenate([[0.0], dqe_freq_1d])
            dqe_1d      = np.concatenate([[1.0], dqe_1d])

    # ------------------------------------------------------------------
    # 2. Handle NaNs
    # ------------------------------------------------------------------
    nan_mask = np.isnan(image)
    image_filled = image.copy().astype(float)
    if nan_mask.any():
        image_filled[nan_mask] = np.nanmean(image)

    # ------------------------------------------------------------------
    # 3. Optional pre-smoothing
    # ------------------------------------------------------------------
    if pre_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        image_filled = gaussian_filter(image_filled, sigma=pre_smooth_sigma)
        #print(f"[INFO] Pre-smoothing applied: sigma={pre_smooth_sigma} px")

    # ------------------------------------------------------------------
    # 4. Build 2D radial PSF (= IFFT of 2D MTF)
    #    Convolutions are performed in the frequency domain
    # ------------------------------------------------------------------
    ny, nx = image_filled.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    freq_radial = np.sqrt(FX**2 + FY**2)

    # For RL, the PSF must be non-negative everywhere to prevent divergence.
    # Using right=0.0 (hard cut beyond the last tabulated frequency) creates a
    # sharp edge that causes Gibbs oscillations: the PSF goes negative in the
    # spatial domain, and the clip(correction, 0) in the RL loop kills pixels
    # at every iteration → image collapses to zero.
    # Fix: hold the last tabulated MTF value for frequencies beyond the table.
    # The MTF at the table edge (~0.1) is small → negligible amplification,
    # but the PSF remains smooth and positive.
    mtf_2d = np.interp(freq_radial, freq_1d, mtf_1d, left=1.0, right=mtf_1d[-1])

    # Build 2D DQE map if provided
    if dqe_file is not None:
        dqe_2d = np.interp(freq_radial, dqe_freq_1d, dqe_1d, left=1.0, right=0.0)
    else:
        dqe_2d = None

    # ------------------------------------------------------------------
    # 5. Richardson-Lucy algorithm
    #    u^(k+1) = u^(k) * (h* ⊛ (I / (h ⊛ u^(k))))
    #    implemented in the frequency domain (convolution = multiplication)
    # ------------------------------------------------------------------
    # Initialise with the observed image (clipped positive)
    u = np.clip(image_filled.copy(), 1e-6, None)
    I = np.clip(image_filled.copy(), 1e-6, None)

    # MTF is symmetric (real, positive) → h* = h in frequency domain
    H  = mtf_2d          # convolution with h
    Ht = mtf_2d          # convolution with h* (symmetric PSF → identical)

    for i in range(n_iterations):
        u_prev = u.copy()

        # Convolution de l'estimation courante avec la PSF
        u_fft  = np.fft.fft2(u)
        Hu     = np.real(np.fft.ifft2(H * u_fft))
        Hu     = np.clip(Hu, 1e-12, None)   # avoid division by zero

        # Observed / convolved estimate ratio
        ratio     = I / Hu
        ratio_fft = np.fft.fft2(ratio)

        # Correlation with the flipped PSF, weighted by DQE if available
        if dqe_2d is not None:
            correction = np.real(np.fft.ifft2(dqe_2d * Ht * ratio_fft))
        else:
            correction = np.real(np.fft.ifft2(Ht * ratio_fft))

        # Update
        u = u * np.clip(correction, 0, None)

        # ------------------------------------------------------------------
        # Stopping criterion: relative change of the estimate u
        #   rel = ||u^(k+1) - u^(k)||_∞ / ||u^(k)||_∞
        # Convergence: rel → 0.  Divergence: rel grows (noise amplification).
        # ------------------------------------------------------------------
        if tol is not None:
            rel = np.max(np.abs(u - u_prev)) / (np.max(u_prev) + 1e-12)
            if verbose:
                print(f"[RL] iter {i+1:4d} | Δrel={rel:.4e}")
            if rel < tol:
                print(f"[RL] Converged at iteration {i+1} "
                      f"(Δrel={rel:.2e} < {tol})")
                break
            if i + 1 == n_iterations:
                print(f"[RL] Maximum iterations reached ({n_iterations}) "
                      f"without convergence (Δrel={rel:.2e}).\n"
                      f"    → Increase n_iterations or tol, "
                      f"or enable pre_smooth_sigma to slow divergence.")

    image_deconv = u

    # ------------------------------------------------------------------
    # 6. Diagnostic PSF plot
    # ------------------------------------------------------------------
    if plot:
        f_plot   = np.linspace(0, 0.5, 300)
        mtf_plot = np.interp(f_plot, freq_1d, mtf_1d)
        # 1D PSF ≈ IFFT of 1D MTF (central profile)
        n_psf = 256
        mtf_sym = np.interp(np.fft.fftfreq(n_psf, d=1.0), freq_1d, mtf_1d)
        psf_1d  = np.real(np.fft.ifftshift(np.fft.ifft(mtf_sym)))
        x_psf   = np.arange(n_psf) - n_psf // 2

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(f_plot, mtf_plot, 'b-', lw=2)
        axes[0].set_xlabel('Spatial frequency (cycles/pixel)')
        axes[0].set_ylabel('MTF')
        axes[0].set_title('MTF used (Richardson-Lucy)')
        axes[0].axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Nyquist')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(x_psf, psf_1d / psf_1d.max(), 'r-', lw=2)
        axes[1].set_xlabel('Position (pixels)')
        axes[1].set_ylabel('Normalised PSF')
        axes[1].set_title('Radial PSF (1D profile)')
        axes[1].set_xlim(-20, 20)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 7. Restore NaNs and clip
    # ------------------------------------------------------------------
    if nan_mask.any():
        image_deconv[nan_mask] = np.nan
    if clip:
        image_deconv = np.clip(image_deconv, 0, None)

    #print(f"[INFO] Richardson-Lucy ({n_iterations} iter). "
    #      f"Input range:  [{np.nanmin(image):.1f}, {np.nanmax(image):.1f}]")
    #print(f"       Output range: [{np.nanmin(image_deconv):.1f}, {np.nanmax(image_deconv):.1f}]")

    return image_deconv


