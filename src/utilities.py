"""
TEM Diffraction Pattern Intensity Correction Functions

This module provides comprehensive functions for correcting intensity artifacts
in transmission electron microscopy (TEM) diffraction patterns, including
flat-field correction, MTF deconvolution, and geometric distortion correction.

Author: Assistant
Date: 2026-01-15
"""

import numpy as np
import hyperspy.api as hs
from scipy.ndimage import median_filter, geometric_transform
from scipy import signal
from typing import Optional, Tuple, Union


def apply_mtf_correction(
    signal: hs.signals.Signal2D,
    mtf_curve: np.ndarray,
    frequencies: np.ndarray,
    min_mtf_threshold: float = 0.1
) -> hs.signals.Signal2D:
    """
    Apply MTF (Modulation Transfer Function) correction to a diffraction pattern.
    
    This function corrects for the detector's MTF by performing deconvolution
    in Fourier space. The MTF describes how the detector attenuates different
    spatial frequencies.
    
    Parameters
    ----------
    signal : hyperspy.signals.Signal2D
        Input diffraction pattern to be corrected. Must have calibrated axes.
    mtf_curve : np.ndarray
        1D array containing MTF values as a function of spatial frequency.
        Values should range from 0 to 1, with 1 representing perfect transfer.
    frequencies : np.ndarray
        1D array of spatial frequencies corresponding to mtf_curve values.
        Units should match the reciprocal of signal's spatial calibration.
    min_mtf_threshold : float, optional
        Minimum MTF value to prevent division by zero and over-amplification
        of high-frequency noise. Values below this are clipped. Default is 0.1.
    
    Returns
    -------
    hyperspy.signals.Signal2D
        MTF-corrected diffraction pattern with same shape and calibration
        as input signal.
    
    Notes
    -----
    The correction is performed by:
    1. Computing the 2D FFT of the input image
    2. Creating a 2D MTF map by interpolating the 1D MTF curve radially
    3. Dividing the FFT by the MTF (deconvolution in Fourier space)
    4. Computing the inverse FFT to return to real space
    
    The min_mtf_threshold prevents numerical instability and excessive noise
    amplification at high spatial frequencies where MTF approaches zero.
    
    Examples
    --------
    >>> import hyperspy.api as hs
    >>> import numpy as np
    >>> 
    >>> # Load diffraction pattern
    >>> dp = hs.load("diffraction.dm3")
    >>> 
    >>> # Define measured MTF
    >>> freqs = np.linspace(0, 1.0, 100)  # Normalized frequency
    >>> mtf = np.exp(-2 * freqs)  # Example Gaussian MTF
    >>> 
    >>> # Apply correction
    >>> dp_corrected = apply_mtf_correction(dp, mtf, freqs)
    
    References
    ----------
    .. [1] Meyer, R. R., & Kirkland, A. I. (2000). The effects of electron 
           and photon scattering on signal and noise transfer properties of 
           scintillators in CCD cameras used for electron detection.
    """
    # Compute FFT of the image
    fft_image = np.fft.fft2(signal.data)
    fft_shifted = np.fft.fftshift(fft_image)
    
    # Create frequency grids
    ny, nx = signal.data.shape
    freq_y = np.fft.fftfreq(ny, d=signal.axes_manager[0].scale)
    freq_x = np.fft.fftfreq(nx, d=signal.axes_manager[1].scale)
    freq_y_shifted = np.fft.fftshift(freq_y)
    freq_x_shifted = np.fft.fftshift(freq_x)
    
    # Create 2D radial frequency grid
    FX, FY = np.meshgrid(freq_x_shifted, freq_y_shifted)
    freq_radial = np.sqrt(FX**2 + FY**2)
    
    # Interpolate MTF curve onto 2D grid
    mtf_2d = np.interp(freq_radial, frequencies, mtf_curve)
    
    # Apply threshold to prevent division by zero and over-amplification
    mtf_2d[mtf_2d < min_mtf_threshold] = min_mtf_threshold
    
    # Perform deconvolution (division in Fourier space)
    fft_corrected = fft_shifted / mtf_2d
    
    # Transform back to real space
    fft_unshifted = np.fft.ifftshift(fft_corrected)
    corrected_data = np.fft.ifft2(fft_unshifted).real
    
    # Create output signal with same metadata
    corrected = signal.deepcopy()
    corrected.data = corrected_data
    
    return corrected


def wiener_deconvolution(
    signal: hs.signals.Signal2D,
    mtf_curve: np.ndarray,
    frequencies: np.ndarray,
    noise_variance: Optional[float] = None,
    corner_size: int = 10
) -> hs.signals.Signal2D:
    """
    Apply Wiener deconvolution for optimal MTF correction with noise suppression.
    
    Wiener deconvolution provides optimal linear filtering in the presence of
    noise by balancing MTF correction against noise amplification. This is
    particularly useful for detector correction when signal-to-noise ratio
    varies across spatial frequencies.
    
    Parameters
    ----------
    signal : hyperspy.signals.Signal2D
        Input diffraction pattern to be corrected.
    mtf_curve : np.ndarray
        1D array of MTF values as function of spatial frequency.
    frequencies : np.ndarray
        1D array of spatial frequencies corresponding to mtf_curve.
    noise_variance : float, optional
        Variance of the noise in the image. If None, it will be estimated
        from the corner regions of the image. Default is None.
    corner_size : int, optional
        Size (in pixels) of corner regions used for noise estimation when
        noise_variance is None. Default is 10.
    
    Returns
    -------
    hyperspy.signals.Signal2D
        Wiener-filtered diffraction pattern.
    
    Notes
    -----
    The Wiener filter is given by:
    
    .. math::
        W(f) = \\frac{H^*(f)}{|H(f)|^2 + 1/SNR(f)}
    
    where H(f) is the MTF, H* is its complex conjugate, and SNR is the
    signal-to-noise ratio at frequency f.
    
    This approach provides better results than simple MTF division when
    dealing with noisy data, as it automatically reduces correction strength
    at frequencies where noise dominates.
    
    Examples
    --------
    >>> # Apply Wiener deconvolution with automatic noise estimation
    >>> dp_wiener = wiener_deconvolution(dp, mtf, freqs)
    >>> 
    >>> # Or specify known noise variance
    >>> dp_wiener = wiener_deconvolution(dp, mtf, freqs, noise_variance=0.05)
    
    References
    ----------
    .. [1] Wiener, N. (1949). Extrapolation, Interpolation, and Smoothing 
           of Stationary Time Series. MIT Press.
    """
    # Compute FFT
    fft_image = np.fft.fft2(signal.data)
    fft_shifted = np.fft.fftshift(fft_image)
    
    # Create 2D MTF map
    ny, nx = signal.data.shape
    freq_y = np.fft.fftfreq(ny, d=signal.axes_manager[0].scale)
    freq_x = np.fft.fftfreq(nx, d=signal.axes_manager[1].scale)
    FX, FY = np.meshgrid(np.fft.fftshift(freq_x), 
                          np.fft.fftshift(freq_y))
    freq_radial = np.sqrt(FX**2 + FY**2)
    mtf_2d = np.interp(freq_radial, frequencies, mtf_curve)
    
    # Estimate noise variance if not provided
    if noise_variance is None:
        # Use corner regions (typically background)
        corners = np.concatenate([
            signal.data[:corner_size, :corner_size].flatten(),
            signal.data[-corner_size:, :corner_size].flatten(),
            signal.data[:corner_size, -corner_size:].flatten(),
            signal.data[-corner_size:, -corner_size:].flatten()
        ])
        noise_variance = np.var(corners)
    
    # Compute signal power and SNR
    signal_power = np.abs(fft_shifted)**2
    snr = signal_power / noise_variance
    
    # Wiener filter: H* / (|H|^2 + SNR^-1)
    wiener_filter = np.conj(mtf_2d) / (np.abs(mtf_2d)**2 + 1.0 / (snr + 1e-10))
    
    # Apply filter
    fft_corrected = fft_shifted * wiener_filter
    
    # Transform back to real space
    corrected_data = np.fft.ifft2(np.fft.ifftshift(fft_corrected)).real
    
    # Create output signal
    corrected = signal.deepcopy()
    corrected.data = corrected_data
    
    return corrected


def complete_diffraction_correction(
    dp_raw: hs.signals.Signal2D,
    dark_ref: hs.signals.Signal2D = None,
    flat_ref: hs.signals.Signal2D = None,
    mtf_curve: Optional[np.ndarray] = None,
    mtf_frequencies: Optional[np.ndarray] = None,
    correct_bad_pixels: bool = True,
    bad_pixel_threshold: Tuple[float, float] = (0.5, 1.5)
) -> hs.signals.Signal2D:
    """
    Complete intensity correction pipeline for diffraction patterns.
    
    This function applies a comprehensive series of corrections to raw
    diffraction patterns, including dark current subtraction, flat-field
    correction, bad pixel correction, and optional MTF deconvolution.
    
    Parameters
    ----------
    dp_raw : hyperspy.signals.Signal2D
        Raw, uncorrected diffraction pattern.
    dark_ref : hyperspy.signals.Signal2D
        Dark reference image (acquired with no beam), same shape as dp_raw.
    flat_ref : hyperspy.signals.Signal2D
        Flat-field reference image (uniform illumination), same shape as dp_raw.
    mtf_curve : np.ndarray, optional
        1D array of MTF values for detector correction. If None, MTF
        correction is skipped. Default is None.
    mtf_frequencies : np.ndarray, optional
        1D array of frequencies corresponding to mtf_curve. Required if
        mtf_curve is provided. Default is None.
    correct_bad_pixels : bool, optional
        Whether to perform bad pixel correction. Default is True.
    bad_pixel_threshold : tuple of float, optional
        (lower, upper) threshold for flat-field normalized values to identify
        bad pixels. Pixels outside this range are interpolated. Default is (0.5, 1.5).
    
    Returns
    -------
    hyperspy.signals.Signal2D
        Fully corrected diffraction pattern.
    
    Notes
    -----
    The correction pipeline follows these steps:
    
    1. **Dark current subtraction**: Removes detector thermal noise
       
       .. math:: I_1 = I_{raw} - I_{dark}
    
    2. **Flat-field correction**: Corrects for non-uniform detector response
       
       .. math:: I_2 = I_1 / (I_{flat} / \\langle I_{flat} \\rangle)
    
    3. **Bad pixel correction**: Interpolates defective pixels using median filter
    
    4. **MTF deconvolution** (optional): Corrects for detector blur
    
    5. **Normalization**: Shifts minimum to zero
    
    Examples
    --------
    >>> # Acquire reference images
    >>> dark = hs.load("dark_ref.dm3")
    >>> flat = hs.load("flat_ref.dm3")
    >>> 
    >>> # Load diffraction pattern
    >>> dp_raw = hs.load("diffraction_raw.dm3")
    >>> 
    >>> # Apply full correction pipeline
    >>> dp_corrected = complete_diffraction_correction(
    ...     dp_raw, dark, flat,
    ...     mtf_curve=mtf_measured,
    ...     mtf_frequencies=freq_array
    ... )
    >>> 
    >>> # Without MTF correction
    >>> dp_corrected = complete_diffraction_correction(dp_raw, dark, flat)
    
    See Also
    --------
    apply_mtf_correction : MTF deconvolution function
    wiener_deconvolution : Alternative MTF correction with noise handling
    """
    # Step 1: Dark current subtraction
    if dark_ref is not None:
        dp = dp_raw - dark_ref
    else:
        dp = dp_raw
    # Step 2: Flat-field correction
    if flat_ref is not None:
        flat_norm = flat_ref / flat_ref.mean()
        dp = dp / flat_norm
    else:
        dp = dp_raw
    
    # Step 3: Bad pixel correction
    if correct_bad_pixels:
        low_thresh, high_thresh = bad_pixel_threshold
        bad_pixels = (flat_norm.data < low_thresh) | (flat_norm.data > high_thresh)
        
        dp_clean = dp.deepcopy()
        
        # Interpolate bad pixels using median of neighborhood
        for i in range(dp.data.shape[0]):
            for j in range(dp.data.shape[1]):
                if bad_pixels[i, j]:
                    # Extract 3x3 neighborhood
                    i_min, i_max = max(0, i-1), min(dp.data.shape[0], i+2)
                    j_min, j_max = max(0, j-1), min(dp.data.shape[1], j+2)
                    neighborhood = dp.data[i_min:i_max, j_min:j_max]
                    
                    # Replace with median (excluding the bad pixel itself)
                    dp_clean.data[i, j] = np.median(neighborhood)
    else:
        dp_clean = dp
    
    # Step 4: MTF correction (if provided)
    if mtf_curve is not None and mtf_frequencies is not None:
        dp_clean = apply_mtf_correction(
            dp_clean,
            mtf_curve,
            mtf_frequencies
        )
    
    # Step 5: Final normalization
    dp_clean = dp_clean - dp_clean.min()
    
    return dp_clean


def ellipse_correction(
    image: np.ndarray,
    ratio: float = 1.05,
    angle: float = 0.0,
    order: int = 3
) -> np.ndarray:
    """
    Correct elliptical distortion in diffraction patterns.
    
    Many detectors and projection systems introduce slight elliptical distortion
    where circular features appear as ellipses. This function corrects such
    distortion by applying an inverse elliptical transformation.
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D image array to be corrected.
    ratio : float, optional
        Ratio of ellipse major to minor axis (aspect ratio correction factor).
        Values > 1 compress the major axis, < 1 expand it. Default is 1.05.
    angle : float, optional
        Orientation angle of the ellipse major axis in radians.
        0 corresponds to horizontal, π/2 to vertical. Default is 0.0.
    order : int, optional
        Interpolation order (0-5). Higher values give smoother results but
        are slower. 3 (cubic) is a good balance. Default is 3.
    
    Returns
    -------
    np.ndarray
        Corrected image with same shape as input.
    
    Notes
    -----
    The correction transforms coordinates according to:
    
    1. Translate to image center
    2. Rotate by -angle
    3. Scale x-coordinates by 1/ratio
    4. Rotate by +angle
    5. Translate back
    
    This effectively applies the inverse of the elliptical distortion.
    
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Load diffraction pattern
    >>> dp = hs.load("diffraction.dm3")
    >>> 
    >>> # Correct 5% elliptical distortion along horizontal axis
    >>> corrected = ellipse_correction(dp.data, ratio=1.05, angle=0)
    >>> 
    >>> # Correct ellipse oriented at 30 degrees
    >>> corrected = ellipse_correction(
    ...     dp.data, 
    ...     ratio=1.08, 
    ...     angle=np.pi/6
    ... )
    
    See Also
    --------
    scipy.ndimage.geometric_transform : Underlying transformation function
    """
    def transform_func(output_coords):
        """
        Coordinate transformation function for geometric_transform.
        
        Maps output coordinates to input coordinates using inverse
        elliptical transformation.
        """
        x, y = output_coords
        
        # Image center
        cx, cy = image.shape[0] / 2, image.shape[1] / 2
        
        # Translate to center
        x_rel, y_rel = x - cx, y - cy
        
        # Inverse rotation
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        x_rot = x_rel * cos_a - y_rel * sin_a
        y_rot = x_rel * sin_a + y_rel * cos_a
        
        # Inverse ellipse correction (scale x-axis)
        x_corr = x_rot / ratio
        y_corr = y_rot
        
        # Forward rotation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_final = x_corr * cos_a - y_corr * sin_a + cx
        y_final = x_corr * sin_a + y_corr * cos_a + cy
        
        return x_final, y_final
    
    # Apply geometric transformation
    corrected = geometric_transform(
        image,
        transform_func,
        order=order,
        mode='constant',
        cval=0.0
    )
    
    return corrected


def compare_corrections(
    dp_before: hs.signals.Signal2D,
    dp_after: hs.signals.Signal2D,
    save_figure: Optional[str] = None
) -> dict:
    """
    Compare diffraction patterns before and after correction with visualization.
    
    This function generates a comprehensive comparison showing images,
    histograms, and radial profiles before and after correction, along
    with quantitative metrics.
    
    Parameters
    ----------
    dp_before : hyperspy.signals.Signal2D
        Diffraction pattern before correction.
    dp_after : hyperspy.signals.Signal2D
        Diffraction pattern after correction.
    save_figure : str, optional
        If provided, save the comparison figure to this filepath.
        Default is None (display only).
    
    Returns
    -------
    dict
        Dictionary containing comparison metrics:
        - 'snr_before': Signal-to-noise ratio before correction
        - 'snr_after': Signal-to-noise ratio after correction
        - 'mean_before': Mean intensity before correction
        - 'mean_after': Mean intensity after correction
        - 'std_before': Standard deviation before correction
        - 'std_after': Standard deviation after correction
    
    Notes
    -----
    Signal-to-noise ratio (SNR) is calculated as mean/std, which provides
    a simple quality metric. Higher SNR generally indicates better quality,
    though interpretation depends on the specific imaging conditions.
    
    Examples
    --------
    >>> # Compare before and after correction
    >>> metrics = compare_corrections(dp_raw, dp_corrected)
    >>> print(f"SNR improvement: {metrics['snr_after']/metrics['snr_before']:.2f}x")
    >>> 
    >>> # Save comparison figure
    >>> compare_corrections(
    ...     dp_raw, 
    ...     dp_corrected, 
    ...     save_figure='correction_comparison.png'
    ... )
    """
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Images
    im0 = axes[0, 0].imshow(dp_before.data, cmap='viridis')
    axes[0, 0].set_title('Before Correction')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[1, 0].imshow(dp_after.data, cmap='viridis')
    axes[1, 0].set_title('After Correction')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Histograms
    axes[0, 1].hist(dp_before.data.flatten(), bins=100, alpha=0.7)
    axes[0, 1].set_title('Intensity Histogram Before')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].hist(dp_after.data.flatten(), bins=100, alpha=0.7, color='orange')
    axes[1, 1].set_title('Intensity Histogram After')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Radial profiles
    radial_before = dp_before.get_radial_profile()
    radial_after = dp_after.get_radial_profile()
    
    axes[0, 2].plot(radial_before)
    axes[0, 2].set_title('Radial Profile Before')
    axes[0, 2].set_xlabel('Radius (pixels)')
    axes[0, 2].set_ylabel('Intensity')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].plot(radial_after, color='orange')
    axes[1, 2].set_title('Radial Profile After')
    axes[1, 2].set_xlabel('Radius (pixels)')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(save_figure, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Calculate metrics
    metrics = {
        'snr_before': dp_before.data.mean() / dp_before.data.std(),
        'snr_after': dp_after.data.mean() / dp_after.data.std(),
        'mean_before': float(dp_before.data.mean()),
        'mean_after': float(dp_after.data.mean()),
        'std_before': float(dp_before.data.std()),
        'std_after': float(dp_after.data.std())
    }
    
    # Print summary
    print("=" * 60)
    print("CORRECTION COMPARISON METRICS")
    print("=" * 60)
    print(f"SNR Before:  {metrics['snr_before']:.2f}")
    print(f"SNR After:   {metrics['snr_after']:.2f}")
    print(f"SNR Change:  {metrics['snr_after']/metrics['snr_before']:.2f}x")
    print("-" * 60)
    print(f"Mean Before: {metrics['mean_before']:.2f}")
    print(f"Mean After:  {metrics['mean_after']:.2f}")
    print("-" * 60)
    print(f"Std Before:  {metrics['std_before']:.2f}")
    print(f"Std After:   {metrics['std_after']:.2f}")
    print("=" * 60)
    
    return metrics