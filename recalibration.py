import numpy as np
import numpy as np
from skimage import filters, measure, morphology
from pyFAI import load
from filereader import load_data
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
2 possibilities to recalibrate the beam center (poni1, poni2):

1) Image with beamstop : Fit ellipse to diffraction ring to find center in pixels, then convert to poni
2) Image without beamstop : Find beam position on camera (e.g. max(intentsity)), then convert to poni

In both cases, we can create an AzimuthalIntegrator with the new poni1, poni2 values.

"""
def center_to_poni(cx, cy, px_size, py_size, dist, rot1=0.0, rot2=0.0, rot3=0.0):
    """
    Convertit les coordonnées du centre en pixels (numpy) en poni1, poni2 pour pyFAI.

    Parameters
    ----------
    cx : float
        Coordonnée horizontale (colonne) du centre du faisceau en pixels
    cy : float
        Coordonnée verticale (ligne) du centre du faisceau en pixels
    px_size : float
        Taille du pixel horizontal (m/pixel)
    py_size : float
        Taille du pixel vertical (m/pixel)
    dist : float
        Distance échantillon → détecteur en mètres
    rot1, rot2, rot3 : float
        Rotations du détecteur en radians (tilt x, tilt y, in-plane)

    Returns
    -------
    poni1, poni2 : float
        Coordonnées du faisceau pour pyFAI (poni1 = vertical, poni2 = horizontal)
    """
    # Conversion pixels → m
    # ATTENTION : pyFAI attend poni1 = vertical, poni2 = horizontal
    v = np.array([cy * py_size, cx * px_size, dist])

    # Matrices de rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rot1), -np.sin(rot1)],
                   [0, np.sin(rot1),  np.cos(rot1)]])
    
    Ry = np.array([[ np.cos(rot2), 0, np.sin(rot2)],
                   [0, 1, 0],
                   [-np.sin(rot2), 0, np.cos(rot2)]])
    
    Rz = np.array([[np.cos(rot3), -np.sin(rot3), 0],
                   [np.sin(rot3),  np.cos(rot3), 0],
                   [0, 0, 1]])
    
    # matrice totale
    R = Rz @ Ry @ Rx
    
    # inversion pour obtenir le poni dans le repère du détecteur
    poni_xyz = np.linalg.inv(R) @ v
    poni1, poni2 = poni_xyz[0], poni_xyz[1]
    
    return poni1, poni2



def fit_ellipse_to_ring(image, threshold_rel=0.5, min_size=50):
    """
    Estime le centre d'un anneau de diffraction dans une image TEM
    via ellipse fitting.
    
    Parameters:
    -----------
    image : 2D np.array
        Image TEM contenant un anneau visible
    threshold_rel : float
        Seuil relatif pour extraire les pixels de l'anneau (0-1)
    min_size : int
        Taille minimale d'un objet pour être considéré comme anneau
    
    Returns:
    --------
    cx, cy : float
        Coordonnées estimées du centre du faisceau en pixels
    """
    # 1. Seuillage automatique relatif
    thresh = filters.threshold_otsu(image) * threshold_rel
    binary = image > thresh
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    
    # 2. Label des objets
    labels = measure.label(binary)
    regions = measure.regionprops(labels)
    if len(regions) == 0:
        raise ValueError("Aucun anneau détecté, ajuster threshold_rel")
    
    # Choisir la plus grande région (supposée être l'anneau)
    region = max(regions, key=lambda r: r.area)
    coords = region.coords  # coordonnées (y, x)
    
    # 3. Ajustement ellipse via moments
    y = coords[:, 0]
    x = coords[:, 1]
    
    x_c = x.mean()
    y_c = y.mean()
    
    # Optionnel : calcul axes et orientation pour info
    cov = np.cov(x - x_c, y - y_c)
    evals, evecs = np.linalg.eig(cov)
    a, b = np.sqrt(evals)  # demi-axes
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    
    return x_c, y_c, a, b, angle

def update_ai(intial_ponifile, poni1_new, poni2_new, output_ponifile = None):
    """
    Met à jour un fichier poni avec de nouvelles coordonnées du faisceau.
    
    Parameters:
    -----------
    intial_ponifile : str
        Chemin vers le fichier poni initial
    poni1_new, poni2_new : float
        Nouvelles coordonnées du faisceau en mètres
    output_ponifile : str
        Chemin vers le fichier poni mis à jour
    """
    ai = load(intial_ponifile)
    ai.poni1 = poni1_new
    ai.poni2 = poni2_new
    if output_ponifile is not None:
        ai.write(output_ponifile)
    return ai


def recalibrate_no_beamstop(dm4file, ponifile, output_ponifile=None):
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
    
    # Convertir en poni
    poni1_new, poni2_new = center_to_poni(cx_corrected, cy_corrected, ai.pixel1, ai.pixel2, ai.dist, ai.rot1, ai.rot2, ai.rot3)
    
    # Mettre à jour et sauvegarder
    ai_updated = update_ai(ponifile, poni1_new, poni2_new, output_ponifile)
    
    return ai_updated

def recalibrate_with_beamstop(dm4file, ponifile, threshold_rel=0.5, min_size=50, output_ponifile=None, plot=False):
    """
    Recalibre le centre du faisceau à partir d'une image TEM avec beam stop,
    en utilisant un fit d'ellipse sur l'anneau visible.
    
    Parameters
    ----------
    dm4file : str
        Chemin vers le fichier DM4 contenant l'image TEM
    ponifile : str
        Chemin vers le fichier poni initial
    threshold_rel : float
        Seuil relatif pour extraire les pixels de l'anneau
    min_size : int
        Taille minimale d'un objet pour être considéré comme anneau
    output_ponifile : str
        Chemin pour sauvegarder le poni mis à jour (optionnel)
    plot : bool
        Si True, affiche l'image avec l'ellipse détectée et le centre corrigé (default: False)
    
    Returns
    -------
    ai_updated : AzimuthalIntegrator
        Intégrateur azimutal pyFAI mis à jour avec le centre recalibré
    """

    # --- Charger image ---
    metadata, image = load_data(dm4file,verbose=False)  # ta fonction personnalisée

    # --- Seuillage pour détecter l'anneau ---
    thresh = filters.threshold_otsu(image) * threshold_rel
    binary = image > thresh
    binary = morphology.remove_small_objects(binary, min_size=min_size)

    # --- Label et extraire la plus grande région ---
    labels = measure.label(binary)
    regions = measure.regionprops(labels)
    if len(regions) == 0:
        raise ValueError("Aucun anneau détecté, ajuster threshold_rel")

    region = max(regions, key=lambda r: r.area)
    coords = region.coords  # (y, x)

    y = coords[:, 0]
    x = coords[:, 1]

    # --- Fit ellipse via moments ---
    x_c = x.mean()
    y_c = y.mean()
    
    # Covariance pour axes et angle (optionnel)
    cov = np.cov(x - x_c, y - y_c)
    evals, evecs = np.linalg.eig(cov)
    a, b = np.sqrt(evals)
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])

    # Charger les paramètres initiaux
    ai = load(ponifile)
    if ai.detector.orientation.value == 1: # Topleft Orientation
        cy_corrected = image.shape[0] - y_c
        cx_corrected = image.shape[1] - x_c
    elif ai.detector.orientation.value == 2: # Topright Orientation
        cy_corrected = image.shape[0] - y_c
        cx_corrected = x_c
    elif ai.detector.orientation.value == 3: # Bottomright Orientation
        cy_corrected = y_c
        cx_corrected = x_c
    elif ai.detector.orientation.value == 4: # Bottomleft Orientation
        cy_corrected = y_c
        cx_corrected = image.shape[1] - x_c
    
    # --- Convertir centre en poni ---
    
    poni1_new, poni2_new = center_to_poni(cx_corrected, cy_corrected, ai.pixel1, ai.pixel2,
                                          ai.dist, ai.rot1, ai.rot2, ai.rot3)

    # --- Mettre à jour poni ---
    ai.poni1 = poni1_new
    ai.poni2 = poni2_new
    if output_ponifile:
        ai.write(output_ponifile)
    
    # --- Plotting ---
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Afficher l'image
        vmin, vmax = np.percentile(image, [1, 99])
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        
        # Afficher l'ellipse détectée
        # Convertir angle en degrés pour matplotlib
        angle_deg = np.degrees(angle)
        ellipse = Ellipse(
            xy=(x_c, y_c),
            width=2*a,
            height=2*b,
            angle=angle_deg,
            fill=False,
            edgecolor='red',
            linewidth=2,
            label=f'Fitted ellipse (a={a:.1f}, b={b:.1f}px)'
        )
        ax.add_patch(ellipse)
        
                
        # Afficher les coordonnées corrigées
        ax.plot(x_c, y_c, 'g+', markersize=12, markeredgewidth=1.5,
                label ='recalculated center')
        
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title(f'Beam Center Recalibration (Orientation: {ai.detector.orientation})', fontsize=14)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plt.show()

    return ai



# --- Exemple d'utilisation ---
# image = ton image TEM numpy 2D
# cx, cy, a, b, angle = fit_ellipse_to_ring(image)
# print("Centre estimé : ", cx, cy)

