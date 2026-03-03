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
                              min_size=50, output_ponifile=None, plot=False):
    """
    Recalibrate beam center from a TEM image with beam stop.
    Uses the improved ring detection method with initial center from max intensity.
    
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
    
    # --- Initial center estimation (max intensity) ---
    y_init, x_init = np.unravel_index(np.argmax(image), image.shape)
    
    # --- Define central mask radius if not provided ---
    if center_mask_radius is None:
        center_mask_radius = min(image.shape) * 0.075
    
    # --- Create mask to exclude central region ---
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    distances = np.sqrt((X - x_init)**2 + (Y - y_init)**2)
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
    x_c = np.average(x, weights=weights)
    y_c = np.average(y, weights=weights)
    
    # --- Ellipse calculation for visualization ---
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
        
        ax1.plot(x_init, y_init, 'b+', markersize=15, markeredgewidth=2,
                label='Initial center (max intensity)')
        ax1.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label='Recalibrated center')
        
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

def recalibrate_with_beamstop_noponi_old(image, center_mask_radius=None, threshold_rel=0.5, 
                                      min_size=50, max_search_iterations=5, plot=False):
    """
    Recalibre le centre du faisceau à partir d'une image TEM avec beam stop,
    en détectant l'anneau de diffraction le plus intense distinct de la diffusion centrale.
    
    La méthode :
    1. Estime un centre initial (max d'intensité)
    2. Masque la région centrale pour exclure le faisceau direct et la diffusion centrale
    3. Détecte l'anneau le plus intense en dehors de cette région
    4. Ajuste une ellipse à l'anneau pour trouver le centre précis
    5. Itère si nécessaire pour affiner le centre
    
    Parameters
    ----------
    image : ndarray
        Image TEM sous forme de tableau numpy 2D
    center_mask_radius : float or None
        Rayon du masque central en pixels pour exclure la diffusion centrale.
        Si None, calculé automatiquement (15% de la taille min de l'image)
    threshold_rel : float
        Seuil relatif pour extraire les pixels de l'anneau (fraction du max d'intensité)
    min_size : int
        Taille minimale d'un objet pour être considéré comme anneau
    max_search_iterations : int
        Nombre max d'itérations pour raffiner le centre
    plot : bool
        Si True, affiche l'image avec l'ellipse détectée et le centre corrigé (default: False)
    
    Returns
    -------
    x_c : float
        Coordonnée X du centre en pixels
    y_c : float
        Coordonnée Y du centre en pixels
    """
    
    # --- Estimation initiale du centre (max d'intensité) ---
    y_init, x_init = np.unravel_index(np.argmax(image), image.shape)
    
    # Définir le rayon du masque central si non fourni
    if center_mask_radius is None:
        center_mask_radius = min(image.shape) * 0.075
    
    x_c, y_c = x_init, y_init
    
    # Itération pour affiner le centre
    for iteration in range(max_search_iterations):
        # --- Créer un masque pour exclure la région centrale ---
        Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
        distances = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
        mask_central = distances > center_mask_radius
        
        # Appliquer le masque
        image_masked = image.copy()
        image_masked[~mask_central] = 0
        
        # --- Calculer le profil radial pour identifier l'anneau le plus intense ---
        max_radius = int(np.sqrt((image.shape[0]/2)**2 + (image.shape[1]/2)**2))
        radial_profile = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius)
        
        for r in range(int(center_mask_radius), max_radius):
            ring_mask = (distances >= r) & (distances < r + 1)
            ring_pixels = image[ring_mask]
            if len(ring_pixels) > 0:
                radial_profile[r] = np.mean(ring_pixels)
                radial_counts[r] = len(ring_pixels)
        
        # Trouver le pic le plus intense (anneau principal)
        valid_radii = np.where(radial_counts > 0)[0]
        if len(valid_radii) == 0:
            raise ValueError("Aucun anneau détecté. Vérifier center_mask_radius ou l'image.")
        
        peak_radius_idx = valid_radii[np.argmax(radial_profile[valid_radii])]
        peak_intensity = radial_profile[peak_radius_idx]
        
        # --- Seuillage pour isoler l'anneau autour du pic ---
        # Créer un masque annulaire autour du pic
        ring_width = peak_radius_idx * 0.2  # 20% du rayon du pic
        ring_inner = peak_radius_idx - ring_width/2
        ring_outer = peak_radius_idx + ring_width/2
        ring_mask = (distances >= ring_inner) & (distances <= ring_outer)
        
        # Seuillage sur l'intensité dans cette région annulaire
        thresh = peak_intensity * threshold_rel
        binary = (image > thresh) & ring_mask
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        
        # --- Label et extraire la région annulaire ---
        labels = measure.label(binary)
        regions = measure.regionprops(labels)
        
        if len(regions) == 0:
            # Si aucune région, essayer avec un seuil plus bas
            thresh = peak_intensity * (threshold_rel * 0.7)
            binary = (image > thresh) & ring_mask
            binary = morphology.remove_small_objects(binary, min_size=min_size)
            labels = measure.label(binary)
            regions = measure.regionprops(labels)
            
            if len(regions) == 0:
                raise ValueError(f"Aucun anneau détecté à l'itération {iteration}. Ajuster threshold_rel ou center_mask_radius.")
        
        # Prendre la région avec le plus de pixels (ou la plus proche du rayon attendu)
        region = max(regions, key=lambda r: r.area)
        coords = region.coords  # (y, x)
        
        y = coords[:, 0]
        x = coords[:, 1]
        
        # --- Fit ellipse via moments pour trouver le nouveau centre ---
        x_c_new = x.mean()
        y_c_new = y.mean()
        
        # Vérifier la convergence
        shift = np.sqrt((x_c_new - x_c)**2 + (y_c_new - y_c)**2)
        x_c, y_c = x_c_new, y_c_new
        
        if shift < 0.5:  # Convergence atteinte
            break
    
    # --- Calcul final de l'ellipse pour visualisation ---
    cov = np.cov(x - x_c, y - y_c)
    evals, evecs = np.linalg.eig(cov)
    a, b = np.sqrt(evals)
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        
        # --- Plot 1: Image originale avec centre et masque ---
        vmin, vmax = np.percentile(image, [1, 99])
        ax1.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        # Cercle pour le masque central
        circle = plt.Circle((x_init, y_init), center_mask_radius, 
                           fill=False, edgecolor='cyan', linewidth=2, 
                           linestyle='--', label=f'Masque central (r={center_mask_radius:.0f}px)')
        ax1.add_patch(circle)
        
        ax1.plot(x_init, y_init, 'b+', markersize=15, markeredgewidth=2,
                label='Centre initial (max intensité)')
        ax1.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label='Centre recalibré')
        
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        ax1.set_title('Image et masque central', fontsize=14)
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Profil radial ---
        ax2.plot(valid_radii, radial_profile[valid_radii], 'b-', linewidth=2)
        ax2.axvline(peak_radius_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Pic principal (r={peak_radius_idx}px)')
        ax2.axhline(thresh, color='orange', linestyle='--', linewidth=1.5, 
                   label=f'Seuil ({threshold_rel:.2f}×max)')
        ax2.axvline(ring_inner, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.axvline(ring_outer, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Région annulaire')
        
        ax2.set_xlabel('Rayon (pixels)', fontsize=12)
        ax2.set_ylabel('Intensité moyenne', fontsize=12)
        ax2.set_title('Profil radial d\'intensité', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # --- Plot 3: Image avec ellipse ajustée ---
        ax3.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        angle_deg = np.degrees(angle)
        ellipse = Ellipse(
            xy=(x_c, y_c),
            width=4*a,  # 2 sigma
            height=4*b,
            angle=angle_deg,
            fill=False,
            edgecolor='red',
            linewidth=2,
            label=f'Ellipse ajustée (a={2*a:.1f}, b={2*b:.1f}px)'
        )
        ax3.add_patch(ellipse)
        
        # Afficher la région détectée
        ax3.contour(binary, levels=[0.5], colors='yellow', linewidths=1.5, 
                   linestyles='--', alpha=0.7)
        
        ax3.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label='Centre recalibré')
        
        ax3.set_xlabel('X (pixels)', fontsize=12)
        ax3.set_ylabel('Y (pixels)', fontsize=12)
        ax3.set_title('Anneau détecté et ellipse', fontsize=14)
        ax3.legend(fontsize=10, loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plt.show()
    
    return x_c, y_c


def recalibrate_with_beamstop_noponi(image, center_mask_radius=None, threshold_rel=0.5,
                                           min_size=50, plot=False):
    """
    Recalibrate beam center from a TEM image with beam stop.
    Simplified and fast version using detection of the most intense ring.
    
    Method:
    1. Estimate initial center (max intensity)
    2. Mask central region to exclude direct beam
    3. Detect most intense ring via radial profile
    4. Extract pixels from this ring
    5. Calculate center by moments
    
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
    plot : bool
        If True, displays image with detected ellipse (default: False)
    
    Returns
    -------
    x_c : float
        X coordinate of center in pixels
    y_c : float
        Y coordinate of center in pixels
    """
    
    # --- Initial center estimation (max intensity) ---
    y_init, x_init = np.unravel_index(np.argmax(image), image.shape)
    
    # Define central mask radius if not provided
    if center_mask_radius is None:
        center_mask_radius = min(image.shape) * 0.075
    
    # --- Create mask to exclude central region ---
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    distances = np.sqrt((X - x_init)**2 + (Y - y_init)**2)
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
    x_c = np.average(x, weights=weights)
    y_c = np.average(y, weights=weights)
    
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
        
        ax1.plot(x_init, y_init, 'b+', markersize=15, markeredgewidth=2,
                label='Initial center (max intensity)')
        ax1.plot(x_c, y_c, 'g+', markersize=15, markeredgewidth=2,
                label='Recalibrated center')
        
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        ax1.set_title('Image and search region', fontsize=14)
        ax1.legend(fontsize=10, loc='upper right')
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
        ax3.legend(fontsize=10, loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plt.show()
    
    return x_c, y_c



