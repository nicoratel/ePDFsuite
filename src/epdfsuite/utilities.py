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
    Détection d'angle par transformée de Hough standard.

    Paramètres
    ----------
    edge_data   : masked 2D image
    sigma       : smoothing parameter for Canny edge detection (1–2 for quasi-binary images)
    erosion_px  : numbers of pixels to crop from the edges of the NaN mask (to avoid artefacts)
    num_peaks   : number of peaks to extract from the Hough accumulator (default=5)
    plot        : whether to display debug plots (default=False)
    """
    arr = edge_data.astype(float)
    valid = ~np.isnan(arr)

    # Normaliser entre 0 et 1
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    arr_norm = (arr - vmin) / (vmax - vmin + 1e-12)

    # Érosion : supprimer les bords du masque NaN
    valid_eroded = binary_erosion(valid, iterations=erosion_px)
    
    # Appliquer le masque : zones hors érosion → 0
    arr_masked = np.where(valid_eroded, arr_norm, 0.0)

    # Détection de contours Canny (image normalisée, masquée)
    # low_threshold / high_threshold à ajuster selon ton SNR
    edge_map = canny(arr_masked, sigma=sigma,
                     low_threshold=0.1, high_threshold=0.3,
                     mask=valid_eroded)

       

    # Transformée de Hough standard
    # tested_angles : résolution angulaire — 3600 pts = 0.05° de précision
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 3600, endpoint=False)
    h, theta, d = hough_line(edge_map, theta=tested_angles)

    # Extraction des pics
    _, peak_angles, peak_dists = hough_line_peaks(
        h, theta, d,
        num_peaks=num_peaks,
        threshold=0.3 * h.max()   # ignorer les pics faibles
    )

    if len(peak_angles) == 0:
        print("[WARN] Aucun pic Hough détecté.")
        return 0.0, 0.0, None, None

    theta = peak_angles[0]   # normale à la droite
    rho   = peak_dists[0]    # distance signée origine→droite

    # Angle de la droite (convention : angle par rapport à l'horizontale)
    line_angle_rad = theta + np.pi / 2
    line_angle_rad = (line_angle_rad + np.pi / 2) % np.pi - np.pi / 2
    line_angle_deg = np.degrees(line_angle_rad)

    # ----------------------------------------------------------------
    # Reconstruction géométrique de la droite à partir de (theta, rho)
    # Équation : x*cos(theta) + y*sin(theta) = rho
    # ----------------------------------------------------------------
    ny, nx = edge_data.shape
    x0_img = nx / 2.0   # centre image (origine du repère Hough si on garde
    y0_img = ny / 2.0   # le repère natif de skimage, qui est le coin (0,0))

    # Point de la droite à y = centre de l'image
    # → résoudre : x*cos(theta) + y_mid*sin(theta) = rho
    y_mid = ny / 2.0
    if np.abs(np.cos(theta)) > 1e-6:
        x_at_ymid = (rho - y_mid * np.sin(theta)) / np.cos(theta)
    else:
        x_at_ymid = rho / (np.cos(theta) + 1e-12)   # droite quasi-horizontale

    # Point de la droite à x = centre de l'image
    x_mid = nx / 2.0
    if np.abs(np.sin(theta)) > 1e-6:
        y_at_xmid = (rho - x_mid * np.cos(theta)) / np.sin(theta)
    else:
        y_at_xmid = rho / (np.sin(theta) + 1e-12)

    # Résumé des paramètres retournés
    edge_point = (x_at_ymid, y_mid)          # un point sur la droite
    edge_line  = (theta, rho, line_angle_deg) # (normale, distance, angle droite en °)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Image masquée
        axes[0].imshow(arr_masked, cmap='gray')
        axes[0].set_title(f'Image masquée (érosion {erosion_px}px)')

        # Carte Canny
        axes[1].imshow(edge_map, cmap='gray')
        axes[1].set_title(f'Contours Canny ({edge_map.sum()} px)')

        # Espace de Hough (accumulateur)
        axes[2].imshow(
            np.log(1 + h),
            extent=[np.degrees(theta[0]), np.degrees(theta[-1]),
                    d[-1], d[0]],
            aspect='auto', cmap='hot'
        )
        axes[2].set_xlabel('θ (degrés)')
        axes[2].set_ylabel('ρ (pixels)')
        axes[2].set_title('Accumulateur Hough (log)')
        # Marquer les pics
        for a, dist in zip(peak_angles, peak_dists):
            axes[2].plot(np.degrees(a), dist, 'c+', ms=10, mew=2)

        # Superposer la droite détectée sur l'image
        axes[1].set_title(f'Canny + edge détecté ({line_angle_deg:.2f}°)')
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

def rotate_with_nan(image, angle):
    nan_mask = np.isnan(image)
    
    # Remplacer les NaN par 0 (ou la moyenne) pour l'interpolation
    image_filled = np.where(nan_mask, 0, image)
    
    # Faire tourner l'image ET le masque
    image_rot = rotate(image_filled, angle, reshape=False, cval=0, order=1)
    mask_rot  = rotate(nan_mask.astype(float), angle, reshape=False, cval=1, order=1)
    
    # Remettre les NaN là où le masque > seuil (interpolation partielle)
    result = np.where(mask_rot > 0.5, np.nan, image_rot)
    return result

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
        image = image.copy()
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
    """
    ny, nx = image.shape
    y_idx, x_idx = np.indices((ny, nx))
    d = x_idx * np.cos(theta_hough) + y_idx * np.sin(theta_hough) - rho_hough
    """
    ny, nx = image.shape
    y_idx, x_idx = np.indices((ny, nx))

    d_raw    = x_idx * np.cos(theta_hough) + y_idx * np.sin(theta_hough) - rho_hough
    d_offset = (x_edge_at_ymid * np.cos(theta_hough)
                + (ny / 2.0)   * np.sin(theta_hough)
                - rho_hough)
    d = d_raw - d_offset
    # ------------------------------------------------------------------
    # 3b. Adapt roi_half_width to the available valid pixels on each side
    #     to enforce a symmetric ESF around the edge
    # ------------------------------------------------------------------
    valid = ~np.isnan(image)

    # Maximum available distance on each side within the valid mask
    d_pos_max = d[valid & (d > 0)].max() if (valid & (d > 0)).any() else roi_half_width
    d_neg_max = np.abs(d[valid & (d < 0)].min()) if (valid & (d < 0)).any() else roi_half_width

    # Symmetric half-width = smallest of: user setting, available left, available right
    roi_half_width_eff = min(roi_half_width, d_pos_max, d_neg_max)
    print(f"[INFO] Effective ROI half-width: {roi_half_width_eff:.1f} px "
        f"(left={d_neg_max:.1f}, right={d_pos_max:.1f})")

    roi = np.abs(d) < roi_half_width_eff   # symmetric ROI
    # ------------------------------------------------------------------
    # 4. Select pixels inside the ROI band around the edge
    # ------------------------------------------------------------------
    valid = ~np.isnan(image)
    roi = np.abs(d) < roi_half_width
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
        axes[0, 0].contour(np.abs(d) < roi_half_width, levels=[0.5],
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
        axes[1, 1].plot(freq_norm, mtf, 'r-', linewidth=2, label='Measured MTF')
        axes[1, 1].plot(freq_norm, np.sinc(freq_norm),
                        'k--', linewidth=1.5, label='Square pixel MTF (sinc)')
        axes[1, 1].axhline(0.5, color='gray',   linestyle='--', alpha=0.6)
        axes[1, 1].axhline(0.2, color='orange', linestyle='--', alpha=0.6)
        axes[1, 1].axvline(freq_norm[mtf50_idx], color='blue', linestyle=':',
                           label=f'MTF50 = {freq_norm[mtf50_idx]:.2f} $f_{{Nyq}}$')
        axes[1, 1].axvline(freq_norm[mtf20_idx], color='orange', linestyle=':',
                           label=f'MTF20 = {freq_norm[mtf20_idx]:.2f} $f_{{Nyq}}$')
        axes[1, 1].set_xlabel('Normalised spatial frequency ($f / f_{Nyquist}$)')
        axes[1, 1].set_ylabel('MTF')
        axes[1, 1].set_xlim(0, 1)
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
                  "# Col 2: MTF\n")
        np.savetxt(outputfile,
                   np.column_stack((freq_pixel, mtf)),
                   header=header, comments='')
        print(f"MTF saved: {outputfile}")

    return freq_pixel, mtf

def deconvolve_mtf_2d(image, mtf_file, clip=True, 
                       wiener_epsilon=0.05,
                       pre_smooth_sigma=0.0,
                       plot=False):
    """
    ...(docstring inchangée)...
    
    Parameters (ajouts)
    ----------
    pre_smooth_sigma : float, default=0.0
        Gaussian pre-smoothing sigma (pixels) applied BEFORE deconvolution.
        Reduces Poisson noise amplification at the cost of slight blurring.
        Recommended: 0.5–1.0 for noisy diffraction images.
    plot : bool, default=False
        Display the Wiener filter profile.
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
    wiener_filter  = mtf_2d / (mtf_2d**2 + wiener_epsilon**2)
    mtf_at_zero    = np.interp(0.0, freq_1d, mtf_1d)
    w_at_zero      = mtf_at_zero / (mtf_at_zero**2 + wiener_epsilon**2)
    wiener_filter /= w_at_zero

    print(f"[INFO] Wiener filter: max={wiener_filter.max():.2f}, "
          f"epsilon={wiener_epsilon}, W(0)={w_at_zero:.6f}")

    if plot:
        f_plot   = np.linspace(0, 0.5, 200)
        mtf_plot = np.interp(f_plot, freq_1d, mtf_1d)
        W_plot   = mtf_plot / (mtf_plot**2 + wiener_epsilon**2)
        W_plot  /= W_plot[0]
        plt.figure(figsize=(7, 4))
        plt.plot(f_plot, mtf_plot, 'b-',  label='MTF')
        plt.plot(f_plot, W_plot,   'r-',  label=f'Wiener filter (ε={wiener_epsilon})')
        plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
        plt.xlabel('Spatial frequency (cycles/pixel)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Wiener deconvolution filter')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 7. FFT → deconvolution → IFFT
    # ------------------------------------------------------------------
    image_fft        = np.fft.fft2(image_filled)
    image_fft_deconv = image_fft * wiener_filter
    image_deconv     = np.real(np.fft.ifft2(image_fft_deconv))

    # ------------------------------------------------------------------
    # 8. Restore NaNs and clip
    # ------------------------------------------------------------------
    if nan_mask.any():
        image_deconv[nan_mask] = np.nan
    if clip:
        image_deconv = np.clip(image_deconv, 0, None)

    print(f"[INFO] Done. Input range:  [{np.nanmin(image):.1f}, {np.nanmax(image):.1f}]")
    print(f"       Output range: [{np.nanmin(image_deconv):.1f}, {np.nanmax(image_deconv):.1f}]")

    return image_deconv








