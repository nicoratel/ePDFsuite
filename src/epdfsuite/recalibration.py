import numpy as np
from skimage import filters, measure, morphology
from skimage.transform import hough_ellipse
from skimage.feature import canny
from pyFAI import load
from .filereader import load_data
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
2 possibilities to recalibrate the beam center (poni1, poni2):

1) Image with beamstop : Fit ellipse to diffraction ring to find center in pixels, then convert to poni
2) Image without beamstop : Find beam position on camera (e.g. max(intentsity)), then convert to poni

In both cases, we can create an AzimuthalIntegrator with the new poni1, poni2 values.

"""
def recalibrate_no_beamstop(dm4file, ponifile, output_ponifile=None,plot=False):
    """
    Recalibre le centre du faisceau à partir d'une image TEM sans beamstop.
    
    Parameters:
    -----------
    dm4file : str
        Chemin vers le fichier DM4 contenant l'image TEM
    ponifile : str
        Chemin vers le fichier poni initial
    output_ponifile : str
        Chemin vers le fichier poni mis à jour (optionnel)
    
    Returns:
    --------
    ai : AzimuthalIntegrator
        Intégrateur azimutal avec les nouvelles coordonnées du faisceau
    """
    metadata, image = load_data(dm4file,verbose=False)
    
    # Trouver le centre du faisceau (max intensity)
    cy, cx = np.unravel_index(np.argmax(image), image.shape)
    
    # Charger les paramètres initiaux
    ai = load(ponifile)
    
    # Détecter le binning (regroupement de pixels)
    detector_shape = ai.detector.shape
    binning_y = detector_shape[0] / image.shape[0]
    binning_x = detector_shape[1] / image.shape[1]
    
    # Appliquer les corrections d'orientation dans l'espace de l'image
    if ai.detector.orientation.value == 1: # Topleft Orientation
        cy_corrected = image.shape[0] - cy
        cx_corrected = image.shape[1] - cx
    elif ai.detector.orientation.value == 2: # Topright Orientation
        cy_corrected = image.shape[0] - cy
        cx_corrected = cx
    elif ai.detector.orientation.value == 3: # Bottomright Orientation
        cy_corrected = cy
        cx_corrected = cx
    elif ai.detector.orientation.value == 4: # Bottomleft Orientation
        cy_corrected = cy
        cx_corrected = image.shape[1] - cx
    
    # Redéfinir le détecteur pour correspondre à l'image binned
    # Au lieu de 4096x4096 avec pixels 15µm, utiliser 2048x2048 avec pixels 30µm
    ai.detector.shape = image.shape
    ai.detector.pixel1 = ai.detector.pixel1 * binning_y
    ai.detector.pixel2 = ai.detector.pixel2 * binning_x
    
    # Le centre est maintenant directement dans l'espace de l'image
    # Utiliser setFit2D pour mettre à jour le centre
    fit2d_params = ai.getFit2D()
    fit2d_params['centerX'] = cx_corrected
    fit2d_params['centerY'] = cy_corrected
    ai.setFit2D(**fit2d_params)
    
    # Convertir les coordonnées pixel en coordonnées PONI (en mètres)
    # poni2 correspond à la position X, poni1 à la position Y
    #ai.poni2 = cx_corrected * ai.detector.pixel2
    #ai.poni1 = cy_corrected * ai.detector.pixel1
    
    if output_ponifile:
        ai.write(output_ponifile)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Afficher l'image
        vmin, vmax = np.percentile(image, [1, 99])
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        # Afficher les coordonnées corrigées
        ax.plot(cx_corrected, cy_corrected, 'g+', markersize=12, markeredgewidth=1.5,
                label ='recalculated center')
        
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title(f'Beam Center Recalibration (Orientation: {ai.detector.orientation})', fontsize=14)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plt.show()
    
    return ai

def recalibrate_with_beamstop(dm4file, ponifile, center_mask_radius=None, threshold_rel=0.5, 
                              min_size=50, max_iterations=5, convergence_threshold=1.0,
                              initial_center=None, output_ponifile=None, plot=False):
    """
    Recalibrate beam center from a TEM image with beam stop.
    Uses an iterative ring detection method with initial center from max intensity.
    
    Parameters
    ----------
    dm4file : str
        Path to DM4 file containing TEM image
    ponifile : str
        Path to initial poni file
    center_mask_radius : float or None
        Radius of central mask in pixels to exclude central scattering.
        If None, automatically calculated (7.5% of min image size)
    threshold_rel : float
        Relative threshold to extract ring pixels (fraction of max intensity)
    min_size : int
        Minimum size of an object to be considered as a ring
    max_iterations : int
        Maximum number of iterations for center refinement (default: 5)
    convergence_threshold : float
        Stop iterations when center displacement is below this value in pixels (default: 1.0)
    initial_center : tuple or None
        Initial center coordinates as (x, y) in pixels. If None, uses max intensity position (default: None)
    output_ponifile : str
        Path to save updated poni file (optional)
    plot : bool
        If True, displays image with detected ellipse and corrected center (default: False)
    
    Returns
    -------
    ai_updated : AzimuthalIntegrator
        pyFAI azimuthal integrator updated with recalibrated center
    """

    # --- Load image ---
    metadata, image = load_data(dm4file, verbose=False)
    
    # --- Load initial poni parameters ---
    ai = load(ponifile)
    
    # Detect binning (pixel grouping)
    detector_shape = ai.detector.shape
    binning_y = detector_shape[0] / image.shape[0]
    binning_x = detector_shape[1] / image.shape[1]
    if binning_y !=1 or binning_x != 1:
        print("Image and detector do not have the same size. Check your calibration file.")
    
    # Redefine detector to match binned image
    ai.detector.shape = image.shape
    ai.detector.pixel1 = ai.detector.pixel1 * binning_y
    ai.detector.pixel2 = ai.detector.pixel2 * binning_x
    
    # --- Initial center estimation ---
    if initial_center is not None:
        x_c, y_c = initial_center
        if plot:
            print(f"Using user-provided initial center: ({x_c:.1f}, {y_c:.1f})")
    else:
        # Use max intensity as initial center
        x_c, y_c = np.unravel_index(np.argmax(image), image.shape)[::-1]  # x, y order
        if plot:
            print(f"Using max intensity as initial center: ({x_c:.1f}, {y_c:.1f})")
    
    # --- Define central mask radius if not provided ---
    if center_mask_radius is None:
        center_mask_radius = min(image.shape) * 0.075
    
    # --- Iterative refinement ---
    center_history = [(x_c, y_c)]
    iteration = 0
    displacement = float('inf')
    
    while displacement >= convergence_threshold :#and iteration < max_iterations:
        # --- Create mask to exclude central region ---
        Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
        distances = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
        mask_central = distances > center_mask_radius
        
        # --- Calculate radial profile to identify most intense ring ---
        max_radius = int(np.sqrt((image.shape[0]/2)**2 + (image.shape[1]/2)**2))
        radial_profile = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius)
        
        for r in range(int(center_mask_radius), max_radius):
            ring_mask = (distances >= r) & (distances < r + 1)
            ring_pixels = image[ring_mask]
            if len(ring_pixels) > 0:
                radial_profile[r] = np.mean(ring_pixels)
                radial_counts[r] = len(ring_pixels)
        
        # Find most intense peak (main ring)
        valid_radii = np.where(radial_counts > 0)[0]
        if len(valid_radii) == 0:
            raise ValueError("No ring detected. Check center_mask_radius or image.")
        
        peak_radius_idx = valid_radii[np.argmax(radial_profile[valid_radii])]
        peak_intensity = radial_profile[peak_radius_idx]
        
        # --- Thresholding to isolate ring around peak ---
        ring_width = peak_radius_idx * 0.4  # 40% of peak radius
        ring_inner = max(center_mask_radius, peak_radius_idx - ring_width/2)
        ring_outer = peak_radius_idx + ring_width/2
        ring_mask = (distances >= ring_inner) & (distances <= ring_outer)
        
        # Intensity thresholding in this annular region
        thresh = peak_intensity * threshold_rel
        binary = (image > thresh) & ring_mask
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        
        # --- Extract coordinates and calculate center by moments ---
        coords = np.column_stack(np.where(binary))
        if len(coords) == 0:
            raise ValueError("No pixels detected in ring. Adjust threshold_rel.")
        
        y = coords[:, 0]
        x = coords[:, 1]
        
        # Center by intensity-weighted average
        weights = image[y, x]
        x_c_new = np.average(x, weights=weights)
        y_c_new = np.average(y, weights=weights)
        
        # Check convergence
        displacement = np.sqrt((x_c_new - x_c)**2 + (y_c_new - y_c)**2)
        center_history.append((x_c_new, y_c_new))
        x_c, y_c = x_c_new, y_c_new
        iteration += 1
    
    if plot:
        if displacement < convergence_threshold:
            print(f"Convergence reached after {iteration} iterations (displacement: {displacement:.2f} px)")
        else:
            print(f"Max iterations ({max_iterations}) reached. Final displacement: {displacement:.2f} px)")
    
    # --- Ellipse calculation for visualization ---
    # Recalculate with final center for plotting
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    distances = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    ring_mask = (distances >= ring_inner) & (distances <= ring_outer)
    binary = (image > thresh) & ring_mask
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    
    coords = np.column_stack(np.where(binary))
    y = coords[:, 0]
    x = coords[:, 1]
    
    cov = np.cov(x - x_c, y - y_c)
    evals, evecs = np.linalg.eig(cov)
    a, b = np.sqrt(evals)
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    
    # --- Update poni with new center ---
    # Center is now directly in image space
    # Use setFit2D to properly update the center
    fit2d_params = ai.getFit2D()
    fit2d_params['centerX'] = x_c
    fit2d_params['centerY'] = y_c
    ai.setFit2D(**fit2d_params)
    
    if output_ponifile:
        ai.write(output_ponifile)
    
    # --- Plotting ---
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        
        # --- Plot 1: Original image with mask ---
        vmin, vmax = np.percentile(image, [1, 99])
        ax1.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        # Circle for central mask
        x_init, y_init = center_history[0]
        circle = plt.Circle((x_init, y_init), center_mask_radius,
                           fill=False, edgecolor='cyan', linewidth=2,
                           linestyle='--', label=f'Central mask (r={center_mask_radius:.0f}px)')
        ax1.add_patch(circle)
        
        # Annular search region
        circle_inner = plt.Circle((x_c, y_c), ring_inner,
                                 fill=False, edgecolor='orange', linewidth=1.5,
                                 linestyle=':', alpha=0.7)
        circle_outer = plt.Circle((x_c, y_c), ring_outer,
                                 fill=False, edgecolor='orange', linewidth=1.5,
                                 linestyle=':', alpha=0.7, label='Search region')
        ax1.add_patch(circle_inner)
        ax1.add_patch(circle_outer)
        
        # Show center evolution
        ax1.plot(x_init, y_init, 'b+', markersize=15, markeredgewidth=2,
                label='Initial center (max intensity)')
        
        # Plot intermediate centers if multiple iterations
        if len(center_history) > 2:
            for i, (x_h, y_h) in enumerate(center_history[1:-1], 1):
                ax1.plot(x_h, y_h, 'y+', markersize=10, markeredgewidth=1.5, alpha=0.6)
        
        ax1.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label=f'Final center ({len(center_history)-1} iter.)')
        
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        ax1.set_title(f'Image and search region (Orientation: {ai.detector.orientation})', fontsize=14)
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Radial profile ---
        ax2.plot(valid_radii, radial_profile[valid_radii], 'b-', linewidth=2, label='Radial profile')
        ax2.axvline(peak_radius_idx, color='red', linestyle='--', linewidth=2,
                   label=f'Main peak (r={peak_radius_idx}px)')
        ax2.axhline(thresh, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Threshold ({threshold_rel:.2f}×max)')
        ax2.axvline(ring_inner, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.axvline(ring_outer, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label='Annular region')
        
        ax2.set_xlabel('Radius (pixels)', fontsize=12)
        ax2.set_ylabel('Mean intensity', fontsize=12, color='b')
        ax2.set_title('Radial intensity profile', fontsize=14)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # --- Plot 3: Image with detected ellipse ---
        ax3.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        # Display detected pixels
        ax3.contour(binary, levels=[0.5], colors='yellow', linewidths=1,
                   linestyles='-', alpha=0.6, label='Detected pixels')
        
        # Calculated ellipse
        angle_deg = np.degrees(angle)
        ellipse = Ellipse(xy=(x_c, y_c), width=4*a, height=4*b, angle=angle_deg,
                         fill=False, edgecolor='red', linewidth=2.5,
                         label=f'Ellipse (a={2*a:.1f}, b={2*b:.1f}px)')
        ax3.add_patch(ellipse)
        
        ax3.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label='Detected center')
        
        ax3.set_xlabel('X (pixels)', fontsize=12)
        ax3.set_ylabel('Y (pixels)', fontsize=12)
        ax3.set_title('Detected ring and center', fontsize=14)
        ax3.legend(fontsize=10, loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plt.show()

    return ai


def recalibrate_with_beamstop_noponi(image, center_mask_radius=None, threshold_rel=0.5,
                                           min_size=50, max_iterations=5, convergence_threshold=1.0,
                                           initial_center=None, plot=False):
    """
    Recalibrate beam center from a TEM image with beam stop.
    Uses an iterative ring detection method for robust center determination.
    
    Method:
    1. Estimate initial center (max intensity or user-provided)
    2. Iteratively:
       - Mask central region to exclude direct beam
       - Detect most intense ring via radial profile
       - Extract pixels from this ring
       - Calculate center by moments
       - Check convergence
    
    Parameters
    ----------
    image : ndarray
        TEM image as 2D numpy array
    center_mask_radius : float or None
        Radius of central mask in pixels to exclude central scattering.
        If None, automatically calculated (7.5% of min image size)
    threshold_rel : float
        Relative threshold to extract ring pixels (fraction of max intensity)
    min_size : int
        Minimum size of an object to be considered as a ring
    max_iterations : int
        Maximum number of iterations for center refinement (default: 5)
    convergence_threshold : float
        Stop iterations when center displacement is below this value in pixels (default: 1.0)
    initial_center : tuple or None
        Initial center coordinates as (x, y) in pixels. If None, uses max intensity position (default: None)
    plot : bool
        If True, displays image with detected ellipse (default: False)
    
    Returns
    -------
    x_c : float
        X coordinate of center in pixels
    y_c : float
        Y coordinate of center in pixels
    """
    
    # --- Initial center estimation ---
    if initial_center is not None:
        x_c, y_c = initial_center
        if plot:
            print(f"Using user-provided initial center: ({x_c:.1f}, {y_c:.1f})")
    else:
        # Use max intensity as initial center
        y_init, x_init = np.unravel_index(np.argmax(image), image.shape)
        x_c, y_c = x_init, y_init
        if plot:
            print(f"Using max intensity as initial center: ({x_c:.1f}, {y_c:.1f})")
    
    # Define central mask radius if not provided
    if center_mask_radius is None:
        center_mask_radius = min(image.shape) * 0.075
    
    # --- Iterative refinement ---
    center_history = [(x_c, y_c)]
    iteration = 0
    displacement = float('inf')
    
    while displacement >= convergence_threshold and iteration < max_iterations:
        # --- Create mask to exclude central region ---
        Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
        distances = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
        mask_central = distances > center_mask_radius
        
        # --- Calculate radial profile to identify most intense ring ---
        max_radius = int(np.sqrt((image.shape[0]/2)**2 + (image.shape[1]/2)**2))
        radial_profile = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius)
        
        for r in range(int(center_mask_radius), max_radius):
            ring_mask = (distances >= r) & (distances < r + 1)
            ring_pixels = image[ring_mask]
            if len(ring_pixels) > 0:
                radial_profile[r] = np.mean(ring_pixels)
                radial_counts[r] = len(ring_pixels)
        
        # Find most intense peak (main ring)
        valid_radii = np.where(radial_counts > 0)[0]
        if len(valid_radii) == 0:
            raise ValueError("No ring detected. Check center_mask_radius or image.")
        
        peak_radius_idx = valid_radii[np.argmax(radial_profile[valid_radii])]
        peak_intensity = radial_profile[peak_radius_idx]
        
        # --- Thresholding to isolate ring around peak ---
        ring_width = peak_radius_idx * 0.4  # 40% of peak radius
        ring_inner = max(center_mask_radius, peak_radius_idx - ring_width/2)
        ring_outer = peak_radius_idx + ring_width/2
        ring_mask = (distances >= ring_inner) & (distances <= ring_outer)
        
        # Intensity thresholding in this annular region
        thresh = peak_intensity * threshold_rel
        binary = (image > thresh) & ring_mask
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        
        # --- Extract coordinates and calculate center by moments ---
        coords = np.column_stack(np.where(binary))
        if len(coords) == 0:
            raise ValueError("No pixels detected in ring. Adjust threshold_rel.")
        
        y = coords[:, 0]
        x = coords[:, 1]
        
        # Center by intensity-weighted average
        weights = image[y, x]
        x_c_new = np.average(x, weights=weights)
        y_c_new = np.average(y, weights=weights)
        
        # Check convergence
        displacement = np.sqrt((x_c_new - x_c)**2 + (y_c_new - y_c)**2)
        center_history.append((x_c_new, y_c_new))
        x_c, y_c = x_c_new, y_c_new
        iteration += 1
    
    if plot:
        if displacement < convergence_threshold:
            print(f"Convergence reached after {iteration} iterations (displacement: {displacement:.2f} px)")
        else:
            print(f"Max iterations ({max_iterations}) reached. Final displacement: {displacement:.2f} px)")
    
    # --- Ellipse calculation for visualization ---
    cov = np.cov(x - x_c, y - y_c)
    evals, evecs = np.linalg.eig(cov)
    a, b = np.sqrt(evals)
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        
        # --- Plot 1: Original image with mask ---
        vmin, vmax = np.percentile(image, [1, 99])
        ax1.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        # Circle for central mask
        circle = plt.Circle((x_init, y_init), center_mask_radius,
                           fill=False, edgecolor='cyan', linewidth=2,
                           linestyle='--', label=f'Central mask (r={center_mask_radius:.0f}px)')
        ax1.add_patch(circle)
        
        # Annular search region
        circle_inner = plt.Circle((x_init, y_init), ring_inner,
                                 fill=False, edgecolor='orange', linewidth=1.5,
                                 linestyle=':', alpha=0.7)
        circle_outer = plt.Circle((x_init, y_init), ring_outer,
                                 fill=False, edgecolor='orange', linewidth=1.5,
                                 linestyle=':', alpha=0.7, label='Search region')
        ax1.add_patch(circle_inner)
        ax1.add_patch(circle_outer)
        
        # Show center evolution
        ax1.plot(x_init, y_init, 'b+', markersize=15, markeredgewidth=2,
                label='Initial center (max intensity)')
        
        # Plot intermediate centers if multiple iterations
        if len(center_history) > 2:
            for i, (x_h, y_h) in enumerate(center_history[1:-1], 1):
                ax1.plot(x_h, y_h, 'y+', markersize=10, markeredgewidth=1.5, alpha=0.6)
        
        ax1.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label=f'Final center ({len(center_history)-1} iter.)')
        
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        ax1.set_title('Image and search region', fontsize=14)
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Radial profile ---
        ax2.plot(valid_radii, radial_profile[valid_radii], 'b-', linewidth=2, label='Radial profile')
        ax2.axvline(peak_radius_idx, color='red', linestyle='--', linewidth=2,
                   label=f'Main peak (r={peak_radius_idx}px)')
        ax2.axhline(thresh, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Threshold ({threshold_rel:.2f}×max)')
        ax2.axvline(ring_inner, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.axvline(ring_outer, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label='Annular region')
        
        ax2.set_xlabel('Radius (pixels)', fontsize=12)
        ax2.set_ylabel('Mean intensity', fontsize=12, color='b')
        ax2.set_title('Radial intensity profile', fontsize=14)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # --- Plot 3: Image with detected ellipse ---
        ax3.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        # Display detected pixels
        ax3.contour(binary, levels=[0.5], colors='yellow', linewidths=1,
                   linestyles='-', alpha=0.6, label='Detected pixels')
        
        # Calculated ellipse
        angle_deg = np.degrees(angle)
        ellipse = Ellipse(xy=(x_c, y_c), width=4*a, height=4*b, angle=angle_deg,
                         fill=False, edgecolor='red', linewidth=2.5,
                         label=f'Ellipse (a={2*a:.1f}, b={2*b:.1f}px)')
        ax3.add_patch(ellipse)
        
        ax3.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label='Detected center')
        
        ax3.set_xlabel('X (pixels)', fontsize=12)
        ax3.set_ylabel('Y (pixels)', fontsize=12)
        ax3.set_title('Detected ring and center', fontsize=14)
        ax3.legend(fontsize=10, loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plt.show()
    
    return x_c, y_c



