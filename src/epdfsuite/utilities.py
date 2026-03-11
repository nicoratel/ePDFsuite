import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from .filereader import load_data

def compute_mtf_from_images(image_list, 
                            pixel_size=None,
                            smooth_sigma=0,
                            plot=False,
                            outputfile=None):
    """
    Compute MTF from 2D amorphous carbon images.
    
    Method:
    1. Load multiple amorphous C images
    2. Compute 2D FFT of each image → NPS (Noise Power Spectrum)
    3. Average NPS to reduce noise
    4. Radialize to obtain MTF(f)
    5. Normalize between 0 and 1
    
    Parameters
    ----------
    image_list : list
        List of image paths (str) or numpy 2D arrays
    pixel_size : float, optional
        Pixel size in µm (to convert to µm⁻¹)
        If None, frequencies are in cycles/pixel
    smooth_sigma : float
        Sigma for Gaussian smoothing applied to MTF (0 = no smoothing)
    plot : bool
        Display diagnostic plots
    outputfile : str, optional
        Path to save MTF(f)
    
    Returns
    -------
    freq : array
        Spatial frequencies (cycles/pixel or µm⁻¹ if pixel_size provided)
    mtf : array
        MTF normalized between 0 and 1
    """
    
    print(f"Loading {len(image_list)} images...")
    
    # =====================================
    # Load and average images
    # =====================================
    
    images = []
    first_infos = None
    
    for img_path in image_list:
        if isinstance(img_path, str):
            try:
                infos, img = load_data(img_path)
                img = img.astype(float)
                # Keep first image info to extract pixel_size
                if first_infos is None:
                    first_infos = infos
            except Exception as e:
                print(f"⚠️ Unable to load {img_path}: {e}")
                continue
        else:
            # Already a numpy array
            img = img_path.astype(float)
        
        images.append(img)
    
    if len(images) == 0:
        raise ValueError("No images could be loaded!")
    
    # Extract pixel_size from info if not provided
    if pixel_size is None and first_infos is not None:
        # Extract pixel_size from metadata (all images from same camera)
        if 'pixel_size' in first_infos:
            pixel_size = first_infos['pixel_size'] * 1e6  # Conversion m → µm
            print(f"✓ Pixel size extracted: {pixel_size:.2f} µm")
        else:
            print("⚠️ Pixel size not found in metadata (key 'pixel_size')")
            print(f"   Frequencies will be in cycles/pixel")
    
    # Check that all images have the same size
    shape = images[0].shape
    for i, img in enumerate(images):
        if img.shape != shape:
            print(f"⚠️ Image {i} has different size: {img.shape} vs {shape}")
    
    print(f"✓ {len(images)} images loaded, size: {shape}")
    
    # =====================================
    # Compute 2D power spectrum
    # =====================================
    
    power_spectra = []
    
    for img in images:
        # Subtract mean (remove DC)
        img_centered = img - np.mean(img)
        
        # 2D FFT
        fft2d = np.fft.fft2(img_centered)
        fft2d_shifted = np.fft.fftshift(fft2d)
        
        # Power spectrum
        power_spectrum = np.abs(fft2d_shifted)**2
        power_spectra.append(power_spectrum)
    
    # Average power spectra
    avg_power_spectrum = np.mean(power_spectra, axis=0)
    
    print(f"✓ Average power spectrum computed")
    
    # =====================================
    # Radialize power spectrum
    # =====================================
    
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    
    # Create frequency grids
    fy = np.fft.fftfreq(ny, d=1.0)
    fx = np.fft.fftfreq(nx, d=1.0)
    fy = np.fft.fftshift(fy)
    fx = np.fft.fftshift(fx)
    
    FX, FY = np.meshgrid(fx, fy)
    freq_map = np.sqrt(FX**2 + FY**2)
    
    # Radialize
    max_freq = 0.5  # Nyquist in cycles/pixel
    n_bins = min(cy, cx)
    freq_bins = np.linspace(0, max_freq, n_bins)
    
    mtf_radial = np.zeros(n_bins - 1)
    freq_radial = np.zeros(n_bins - 1)
    
    for i in range(n_bins - 1):
        mask = (freq_map >= freq_bins[i]) & (freq_map < freq_bins[i + 1])
        if np.sum(mask) > 0:
            mtf_radial[i] = np.mean(avg_power_spectrum[mask])
            freq_radial[i] = (freq_bins[i] + freq_bins[i + 1]) / 2
    
    # =====================================
    # Normalize MTF
    # =====================================
    
    # MTF = square root of normalized power spectrum
    mtf_radial = np.sqrt(mtf_radial)
    
    # Normalize between 0 and 1 (MTF(0) = 1)
    mtf_radial /= mtf_radial[0]
    
    # Smooth if requested
    if smooth_sigma > 0:
        mtf_radial = gaussian_filter(mtf_radial, smooth_sigma)
        mtf_radial = np.clip(mtf_radial, 0, 1)  # Ensure MTF stays in [0,1]
    
    # Convert frequencies if pixel_size provided
    if pixel_size is not None:
        freq_radial = freq_radial / pixel_size  # Conversion to µm⁻¹
        freq_unit = "µm⁻¹"
    else:
        freq_unit = "cycles/pixel"
    
    print(f"✓ MTF radialized: {len(freq_radial)} points")
    
    # =====================================
    # Save results
    # =====================================
    
    if outputfile is not None:
        header = f"# MTF computed from {len(images)} amorphous carbon images\n"
        header += f"# Frequency ({freq_unit}), MTF\n"
        np.savetxt(outputfile, np.column_stack((freq_radial, mtf_radial)), 
                   header=header, comments='')
        print(f"✓ MTF saved: {outputfile}")
    
    # =====================================
    # Plots
    # =====================================
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Average image
        avg_image = np.mean(images, axis=0)
        ax = axes[0, 0]
        im = ax.imshow(avg_image, cmap='gray')
        ax.set_title("Average image (amorphous C)")
        plt.colorbar(im, ax=ax)
        
        # 2D power spectrum (log)
        ax = axes[0, 1]
        im = ax.imshow(np.log10(avg_power_spectrum + 1), cmap='hot')
        ax.set_title("2D power spectrum (log)")
        plt.colorbar(im, ax=ax)
        
        # Radial MTF
        ax = axes[1, 0]
        ax.plot(freq_radial, mtf_radial, 'b-', linewidth=2, label='MTF')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='MTF = 0.5')
        ax.set_xlabel(f"Spatial frequency ({freq_unit})")
        ax.set_ylabel("MTF")
        ax.set_title("Radial MTF")
        ax.set_ylim([0, 1.1])
        ax.grid(alpha=0.3)
        ax.legend()
        
        # MTF in log-log scale
        ax = axes[1, 1]
        ax.loglog(freq_radial[1:], mtf_radial[1:], 'b-', linewidth=2)
        ax.set_xlabel(f"Spatial frequency ({freq_unit})")
        ax.set_ylabel("MTF")
        ax.set_title("MTF (log-log scale)")
        ax.grid(alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.show()
    
    return freq_radial, mtf_radial


def correct_intensity_mtf(q,I, ponifile, mtf_file):
    """
    Correct the intensity from camera MTF.
    
    Parameters
    ----------
    q : array
        q values (diffraction space)
    I : array
        Intensity to correct with MTF
    mtf_file : str
        Path for MTF file containing two columns: frequency (f_pixel) and MTF(f_pixel)
    
    Returns
    -------
   I_corrected : array
        Intensity corrected by MTF
    """
    
    # Load MTF data
    mtf_data = np.loadtxt(mtf_file, comments='#')
    f_pixel_mtf = mtf_data[:, 0]
    mtf_values = mtf_data[:, 1]
    
    # Load pixel size cameralength and beam centre from poni file
    # Lecture du fichier PONI
    poni_data = {}
    with open(ponifile, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                key = key.strip()
                val = val.strip()
                try:
                    val = eval(val)
                except:
                    pass
                poni_data[key] = val

    # Paramètres
    pixel1 = poni_data['Detector_config']['pixel1']  # Y
    pixel2 = poni_data['Detector_config']['pixel2']  # X
    Nx, Ny = poni_data['Detector_config']['max_shape']
    L = poni_data['Distance']
    lambda_e = poni_data['Wavelength']

    # --- Map q -> spatial frequency in pixels ---
    # r = distance from beam center in pixels
    # Δq per pixel = (4*pi / lambda) * pixel_size / L
    delta_q_pixel = (4*np.pi / lambda_e) * pixel2 / L  # Å^-1 / pixel
    # Convert q array to equivalent f_pixel (cycles/pixel)
    f_pixel_q = q / delta_q_pixel / Nx                # cycles/pixel (normalized)

    # --- Interpolate MTF values to q-space ---
    mtf_interp = np.interp(f_pixel_q, f_pixel_mtf, mtf_values, left=1.0, right=0.0)

    # --- Correct intensity ---
    # Avoid division by very small MTF values to limit noise amplification
    mtf_safe = np.clip(mtf_interp, 0.05, 1.0)
    I_corrected = I / mtf_safe

    return I_corrected



    
    

