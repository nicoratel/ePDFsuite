from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from .filereader import load_data
import fabio
import os
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math

def build_calibration_data_from_cif(
    cif_file,
    wavelength,
    n_peaks=15,
    two_theta_range=(0, 2)
):
    """
    Génère une liste de pics de diffraction à partir d'un fichier CIF.

    Parameters
    ----------
    cif_file : str
        Chemin vers le fichier CIF
    wavelength : float
        Longueur d'onde (en Å)
    n_peaks : int
        Nombre de pics à retourner
    two_theta_range : tuple
        Intervalle en 2θ (degrés)

    Returns
    -------
    peaks : list of dict
        Chaque pic contient :
        - hkl
        - d (Å)
        - two_theta (deg)
        - intensity (relative)
    """

    # Lecture de la structure
    structure = Structure.from_file(cif_file)

    # Calculateur XRD
    xrd = XRDCalculator(wavelength=wavelength)

    pattern = xrd.get_pattern(structure, two_theta_range=two_theta_range)

    peaks = []
    for i in range(min(n_peaks, len(pattern.x))):
        peaks.append({
            "hkl": pattern.hkls[i][0]["hkl"],
            "d": pattern.d_hkls[i],
            "two_theta": pattern.x[i],
            "intensity": pattern.y[i]
        })

    line2write=''
    for p in peaks:
        line2write += str(p['d'])+'\n'
    with open('./distances.txt','w') as f:
        f.write(line2write)
    return peaks


def perform_geometric_calibration(
        cif_file: str,
        image_file: str):
    """
    Docstring pour perform_geometric_calibration
    
    :param cif_file: Path to cif structure file for calibrant
        :type cif_file: str
    :param dm4_file: Path to dm4 file for calibrant diffraction data
        :type dm4_file: str
    """
    
    # load data and metadata
    detector_info, raw_image = load_data(image_file)

    # Define output EDF file name
    edffile = image_file.replace('.dm4', '.edf')

    # Create EDF image and save
    edf_image = fabio.edfimage.EdfImage(data=raw_image, header=detector_info)
    edf_image.write(edffile)

    # Create distance files for use in pyFAI-calib2
    peaks = build_calibration_data_from_cif(
        cif_file,
        wavelength=detector_info['wavelength'],  
        n_peaks=10)
    
    print("=" * 70)
    print('EXPERIMENT SETTINGS TO INPUT IN PYFAI-CALIB2:')
    print('='*70)
    print(f'Camera description={detector_info["description"]}')
    print(f'pixel_size_x={detector_info["pixel_size"]}X{detector_info["pixel_size"]}')  # in µmeters
    print(f'image dimension={detector_info["image_width"]}X{detector_info["image_height"]}') # in pixels 
    print(f'Electron wavelength={detector_info["wavelength"]} Å')

    print('=' * 70 )
    print('Launching pyFAI-calib2')

    os.system(f'conda run -n epdfpy pyFAI-calib2 -c ./distances.txt {edffile}')

    # Clean up distance file and edf file
    os.remove(edffile)
    os.remove('./distances.txt')


def get_calibration_parameters(poni_file: str):
    """
    Docstring pour get_calibration_parameters
    
    Returns
    -------
    calibration_params : dict
        Dictionnaire contenant les paramètres de calibration géométrique
    """
    import json
    calibration_params = {}
    if not os.path.exists(poni_file):
        raise FileNotFoundError(f"Fichier PONI non trouvé: {poni_file}")

    with open(poni_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('Distance:'):
            calibration_params['distance'] = float(line.split(':')[1].strip())
        elif line.startswith('Poni1:'):
            calibration_params['poni1'] = float(line.split(':')[1].strip())
        elif line.startswith('Poni2:'):
            calibration_params['poni2'] = float(line.split(':')[1].strip())
        elif line.startswith('Rot1:'):
            calibration_params['rot1'] = float(line.split(':')[1].strip())
        elif line.startswith('Rot2:'):
            calibration_params['rot2'] = float(line.split(':')[1].strip())
        elif line.startswith('Rot3:'):
            calibration_params['rot3'] = float(line.split(':')[1].strip())
        elif line.startswith('Wavelength:'):
            calibration_params['wavelength'] = float(line.split(':')[1].strip())
        elif line.startswith('Detector_config:'):
            config_str = line.split(':', 1)[1].strip()
            config = json.loads(config_str)
            calibration_params['pixel1'] = config.get('pixel1', None)
            calibration_params['pixel2'] = config.get('pixel2', None)
            calibration_params['max_shape'] = config.get('max_shape', None)

    return calibration_params