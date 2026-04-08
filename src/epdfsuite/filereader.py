import re
import os
import sys
import hyperspy.api as hs
from .camera_library import DETECTOR_LIBRARY
import numpy as np


def extract_camera_type(metadata, detector_lib=DETECTOR_LIBRARY):
    """
    Identify the camera type from HyperSpy metadata using regex alias matching.

    First attempts an exact match of the image title against library keys, then
    falls back to case-insensitive regex search over each detector's alias list.

    Parameters
    ----------
    metadata : HyperSpy metadata object
        Metadata loaded from a DM4 (or other HyperSpy-supported) file.
    detector_lib : dict, optional
        Detector library to search. Defaults to the built-in
        :data:`DETECTOR_LIBRARY`.

    Returns
    -------
    camera_key : str or None
        Key of the matched detector in ``detector_lib``, or ``None`` if not found.
    camera_title : str or None
        Raw title string found in the metadata, or ``None`` if unavailable.
    """
    try:
        if hasattr(metadata, 'General'):
            general = metadata.General
            if hasattr(general, 'title'):
                title = general.title
                
                
                # First, search for exact match
                if title in detector_lib:
                    return title, title
                
                # Then, search by regex on aliases
                title_lower = title.lower()
                
                for camera_key, params in detector_lib.items():
                    if 'aliases' in params:
                        for alias_pattern in params['aliases']:
                            try:
                                # Compile pattern as regex
                                pattern = re.compile(alias_pattern, re.IGNORECASE)
                                if pattern.search(title_lower):                                    
                                    return camera_key, title
                            except re.error:
                                # If not valid regex, search as substring
                                if alias_pattern.lower() in title_lower:                                    
                                    return camera_key, title
                
                # No match found
                print(f"  ⚠ No match found in aliases")
                return None, title
    except Exception as e:
        print(f"Error extracting camera type: {e}")
    
    return None, None

def _search_metadata_recursive(obj, target_keys, depth=0, max_depth=10):
    """
    Recursively search a HyperSpy metadata object for target field names.

    Traverses attributes and dictionary-like entries at each level, performing
    case-insensitive substring matching of attribute names against ``target_keys``.

    Parameters
    ----------
    obj : object
        Root metadata object (HyperSpy ``DictionaryTreeBrowser`` or any object
        with ``__dict__`` or dict-like interface).
    target_keys : list of str
        Field names to search for (case-insensitive substring match).
    depth : int, optional
        Current recursion depth. Default is 0.
    max_depth : int, optional
        Maximum recursion depth to prevent infinite loops. Default is 10.

    Returns
    -------
    found : dict
        Mapping ``{attribute_name: value}`` for all matched scalar fields
        (int, float, or str).
    """
    found = {}
    
    if depth > max_depth or obj is None:
        return found
    
    try:
        # Search in attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                attr_lower = attr_name.lower()
                
                # Check if this attribute matches any target key
                for target_key in target_keys:
                    if target_key.lower() in attr_lower:
                        if isinstance(attr_value, (int, float, str)) and attr_value is not None:
                            found[attr_name] = attr_value
                        break
                
                # Recursively search in nested objects
                if hasattr(attr_value, '__dict__'):
                    nested_found = _search_metadata_recursive(attr_value, target_keys, depth + 1, max_depth)
                    found.update(nested_found)
        
        # Search in dictionary-like objects
        if hasattr(obj, '__getitem__') and hasattr(obj, 'keys'):
            for key in obj.keys():
                key_lower = key.lower()
                value = obj[key]
                
                # Check if key matches any target
                for target_key in target_keys:
                    if target_key.lower() in key_lower:
                        if isinstance(value, (int, float, str)) and value is not None:
                            found[key] = value
                        break
                
                # Recursively search in nested objects
                if hasattr(value, '__dict__') or (hasattr(value, '__getitem__') and hasattr(value, 'keys')):
                    nested_found = _search_metadata_recursive(value, target_keys, depth + 1, max_depth)
                    found.update(nested_found)
    except Exception as e:
        pass  # Silently skip objects that can't be searched
    
    return found

def extract_wavelength(metadata=None, voltage_kv=None):
    """
    Determine the relativistic electron wavelength from metadata or voltage.

    Searches the metadata recursively for a wavelength, beam energy, or
    accelerating voltage field. If a direct wavelength is found (0.001–1 Å)
    it is returned immediately. Otherwise the wavelength is computed from the
    voltage using the relativistic de Broglie relation:

    .. math::

        \\lambda = \\frac{h}{\\sqrt{2 m_e e V \\left(1 + \\frac{eV}{2 m_e c^2}\\right)}}

    Parameters
    ----------
    metadata : HyperSpy metadata object, optional
        Metadata from a DM4 file. If ``None``, ``voltage_kv`` must be provided.
    voltage_kv : float, optional
        Accelerating voltage in kV. Overrides the voltage found in ``metadata``
        if both are provided.

    Returns
    -------
    wavelength : float or None
        Electron wavelength in Å, or ``None`` if neither voltage nor wavelength
        could be determined.
    """
    
    wavelength_angstrom = None
    
    # Deep search in metadata for relevant fields
    if metadata is not None:
        try:
            # Search for wavelength, energy, and voltage fields
            search_keys = ['wavelength', 'energy', 'beam_energy', 'accelerating_voltage', 'acceleration_voltage']
            found_values = _search_metadata_recursive(metadata, search_keys)
            
            # Priority: wavelength > beam_energy > energy > accelerating_voltage
            for key, value in found_values.items():
                key_lower = key.lower()
                
                try:
                    # Try to convert to float
                    val_float = float(value)
                    
                    # Check for wavelength
                    if 'wavelength' in key_lower:
                        # Assume it's in Ångströms if small enough (typically 0.01 - 0.1 Å for electrons)
                        if 0.001 < val_float < 1:
                            wavelength_angstrom = val_float
                            #print(f"  ✓ Found wavelength in metadata: {val_float:.6f} Å")
                            break
                    
                    # Check for energy fields
                    elif any(term in key_lower for term in ['energy', 'accelerat']):
                        if voltage_kv is None:
                            # Energy in keV if < 10000, else in eV
                            if val_float < 10000:
                                voltage_kv = val_float  # Already in keV
                                #print(f"  ✓ Found energy in metadata: {val_float:.1f} keV")
                            else:
                                voltage_kv = val_float / 1000  # Convert from eV to keV
                                #print(f"  ✓ Found energy in metadata: {val_float:.1f} eV ({voltage_kv:.1f} keV)")
                
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values
            
        except Exception as e:
            print(f"Warning: Could not search metadata: {e}")
    
    # If wavelength found directly, return it
    if wavelength_angstrom is not None:
        return wavelength_angstrom
    
    # Otherwise, calculate from voltage
    if voltage_kv is None:
        print("⚠ Voltage (kV) or wavelength not found in metadata")
        return None
    
    # Ensure voltage_kv is a number
    try:
        voltage_kv = float(voltage_kv)
    except (ValueError, TypeError):
        print("⚠ Could not convert voltage to float")
        return None
    
    # Constants (SI units)
    h = 6.62607015e-34  # Planck constant (J·s)
    m_e = 9.1093837015e-31  # Electron mass (kg)
    e = 1.602176634e-19  # Elementary charge (C)
    c = 299792458  # Speed of light (m/s)
    
    V = voltage_kv * 1000  # Convert kV to V
    
    # De Broglie wavelength with relativistic correction
    # λ = h / √(2*m_e*e*V*(1 + e*V/(2*m_e*c²)))
    rest_energy = m_e * c**2 / e  # Rest energy in eV
    kinetic_energy = V  # Kinetic energy in eV
    
    relativistic_factor = 1 + kinetic_energy / (2 * rest_energy)
    wavelength_m = h / np.sqrt(2 * m_e * e * V * relativistic_factor)
    
    # Convert to Ångströms
    wavelength_angstrom = wavelength_m * 1e10
    
    #print(f"  ✓ Calculated wavelength from {voltage_kv:.1f} kV: {wavelength_angstrom:.6f} Å")
    return wavelength_angstrom

def get_detector_params(camera_key, detector_lib=DETECTOR_LIBRARY):
    """
    Return the parameters of a detector from the library, without alias entries.

    Parameters
    ----------
    camera_key : str
        Key name of the detector in ``detector_lib``.
    detector_lib : dict, optional
        Detector library to query. Defaults to :data:`DETECTOR_LIBRARY`.

    Returns
    -------
    params : dict or None
        Copy of the detector parameter dictionary (``pixel_size``,
        ``image_width``, ``image_height``, ``binning``, ``description``),
        or ``None`` if ``camera_key`` is not found.
    """
    if camera_key in detector_lib:
        params = detector_lib[camera_key].copy()
        params.pop('aliases', None)  # Remove aliases from result
        return params
    else:
        print(f"⚠ Camera type '{camera_key}' not found in library")
        print(f"Available cameras: {list(detector_lib.keys())}")
        return None
    
def load_data(file, normalize=True, verbose=True):
    """
    Load a DM4 image file and return detector metadata and the image array.

    Identifies the camera type and wavelength from the file metadata,
    optionally normalises the raw counts by the exposure time, and prints
    a summary of the detected instrument parameters.

    Parameters
    ----------
    file : str
        Path to the DM4 image file.
    normalize : bool, optional
        If ``True`` (default), divide the image by the exposure time (s)
        to obtain a count-rate image. Has no effect if the exposure time
        cannot be found in the metadata.
    verbose : bool, optional
        If ``True`` (default), print loaded file info and detector parameters.

    Returns
    -------
    detector_info : dict
        Dictionary with keys: ``camera_type``, ``camera_title``,
        ``pixel_size`` (µm), ``image_width`` (px), ``image_height`` (px),
        ``binning``, ``description``, ``wavelength`` (Å),
        ``exposure_time`` (s, if found).
    raw_image : ndarray
        2D float array of the image, normalised by exposure time if requested.
    """
    
    
    # Load image
    image = hs.load(file)
    metadata = image.metadata
    
    raw_image = image.data
    
    # Extract detector info automatically
    camera_key, camera_title = extract_camera_type(metadata)
    
    # Extract wavelength information
    wavelength_info = extract_wavelength(metadata)
    
    if camera_key:
        detector_info = get_detector_params(camera_key)
        detector_info['camera_type'] = camera_key
        detector_info['camera_title'] = camera_title
        
    else:
        # If detector not found, create dict with default values
        detector_info = {
            'camera_type': None,
            'camera_title': camera_title,
            'wavelength' : wavelength_info,
            'pixel_size': None,
            'image_width': raw_image.shape[1],
            'image_height': raw_image.shape[0],
            'binning': 1,
            'description': 'Unknown detector',
            'wavelength': wavelength_info if wavelength_info is not None else None,           
            'note': 'Dimensions from raw image'
        }
    
    # Add wavelength information if available
    if wavelength_info is not None:
        detector_info['wavelength'] = wavelength_info
    
    # Extract exposure time
    exposure_time = None   
    found_values = _search_metadata_recursive(metadata, ['exposure_time','exposure', 'acquisition_time', 'dwell_time'])
    for key, value in found_values.items():
        
        exposure_time = float(value)
        detector_info['exposure_time'] = exposure_time
        break
        
    
    if verbose:
        print(f"Loaded file: {file}")
        print("Sample information:")
        for key, value in detector_info.items():
            print(f"  {key}: {value}")
    
    if normalize:
        if 'exposure_time' in detector_info and detector_info['exposure_time'] is not None:
            raw_image = raw_image / detector_info['exposure_time']
            #if verbose:
            #print(f"  ✓ Normalized image by exposure time: {detector_info['exposure_time']} s")
        else:
            print("  ⚠ Exposure time not found, image not normalized")
    
    return detector_info, raw_image

def add_detector(camera_key, pixel_size, image_width, image_height, binning=1, description='', aliases=None):
    """
    Add a new detector entry to the in-memory library and persist it to disk.

    The entry is appended to the global :data:`DETECTOR_LIBRARY` dictionary
    and immediately written back to ``camera_library.py``.

    Parameters
    ----------
    camera_key : str
        Unique identifier for the detector (e.g. ``'K3_300kV'``).
    pixel_size : float
        Physical pixel size in micrometres (µm).
    image_width : int
        Full-frame image width in pixels.
    image_height : int
        Full-frame image height in pixels.
    binning : int, optional
        Hardware binning factor applied at acquisition. Default is 1.
    description : str, optional
        Human-readable description of the detector.
    aliases : list of str, optional
        List of regex patterns (case-insensitive) used to match this detector
        from image title strings in metadata. Default is ``[]``.

    Returns
    -------
    success : bool
        ``True`` if the detector was added, ``False`` if ``camera_key``
        already exists in the library.
    """
    if camera_key in DETECTOR_LIBRARY:
        print(f"⚠ Camera type '{camera_key}' already exists in library")
        return False
    
    if aliases is None:
        aliases = []
    
    # Add to in-memory dictionary
    DETECTOR_LIBRARY[camera_key] = {
        'pixel_size': pixel_size,
        'image_width': image_width,
        'image_height': image_height,
        'binning': binning,
        'description': description,
        'aliases': aliases
    }
    
    # Save to camera_library.py file
    _save_detector_library()
    
    print(f"✓ Added detector: '{camera_key}' with pixel_size={pixel_size}µm")
    print(f"✓ Saved to camera_library.py")
    return 

def _save_detector_library():
    """
    Persist the current state of :data:`DETECTOR_LIBRARY` to ``camera_library.py``.

    Overwrites the file with a pretty-printed Python assignment so that changes
    made via :func:`add_detector` survive across sessions.
    """
    import pprint
    
    # Get the path to camera_library.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    camera_lib_path = os.path.join(current_dir, 'camera_library.py')
    
    # Generate the content
    content = "# TEM detector library with their specifications, can be extended\nDETECTOR_LIBRARY = "
    content += pprint.pformat(DETECTOR_LIBRARY, indent=4)
    content += "\n"
    
    # Write to file
    try:
        with open(camera_lib_path, 'w') as f:
            f.write(content)
    except Exception as e:
        print(f"⚠ Error saving detector library: {e}")