"""
Lobato parametrization for electron and X-ray scattering factors.

Based on:
Lobato, I., & Van Dyck, D. (2014). An accurate parameterization for scattering 
factors, electron densities and electrostatic potentials for neutral atoms that 
obey all physical constraints. Acta Crystallographica Section A, 70(6), 636-649.

This is a simplified standalone implementation extracted from abTEM to avoid
heavy dependencies on visualization libraries.
"""

import numpy as np
import json
from pathlib import Path


# Lobato parameters for all elements (H to Lr)
_JSON_PATH = Path(__file__).parent / "lobato_parameters.json"
with _JSON_PATH.open() as _f:
    LOBATO_PARAMETERS = json.load(_f)



def electron_scattering_factor(k2, params):
    """
    Compute electron scattering factor using Lobato parametrization.
    
    Parameters
    ----------
    k2 : float or ndarray
        Squared scattering vector magnitude (in 1/Angstrom^2)
    params : ndarray
        Lobato parameters [2 x 5] array where params[0] are 'a' coefficients
        and params[1] are 'b' coefficients
        
    Returns
    -------
    f : float or ndarray
        Electron scattering factor
        
    Notes
    -----
    Based on Lobato & Van Dyck (2014), Acta Cryst. A70, 636-649
    f_e(k^2) = sum_i [ a_i * (2 + b_i * k^2) / (1 + b_i * k^2)^2 ]
    """
    p = np.array(params, dtype=np.float64)
    # Vectorized computation over all 5 terms
    k2 = np.asarray(k2)
    result = np.zeros_like(k2, dtype=np.float64)
    for i in range(5):
        result += (p[0, i] * (2.0 + p[1, i] * k2) / 
                   (1.0 + p[1, i] * k2) ** 2)
    return result


def x_ray_scattering_factor(k, params):
    """
    Compute X-ray scattering factor using Lobato parametrization.
    
    Parameters
    ----------
    k : float or ndarray
        Scattering vector magnitude (in 1/Angstrom)
    params : ndarray
        Lobato parameters [2 x 5] array
        
    Returns
    -------
    f : float or ndarray
        X-ray scattering factor
        
    Notes
    -----
    Uses Bohr radius conversion: 1 Bohr = 0.529177 Angstrom
    Based on Lobato & Van Dyck (2014), Acta Cryst. A70, 636-649
    f_xray(k) = sum_i [ 2*pi^2*a0 * a_i / (b_i * (1 + b_i * k^2)^2) ]
    where a0 is the Bohr radius in Angstrom
    """
    BOHR_TO_ANGSTROM = 0.529177210903
    p = np.array(params, dtype=np.float64)
    k = np.asarray(k)
    k2 = k**2
    result = np.zeros_like(k, dtype=np.float64)
    for i in range(5):
        result += (2 * np.pi**2 * BOHR_TO_ANGSTROM * p[0, i] / 
                   (p[1, i] * (1 + p[1, i] * k2) ** 2))
    return result


def compute_scattering_profile(elements, s_values, xray=False):
    """
    Compute scattering factor profiles for multiple elements.
    
    Parameters
    ----------
    elements : list of str
        List of element symbols (e.g., ['Au', 'C'])
    s_values : ndarray
        Scattering vector magnitudes in 1/Angstrom (s = sin(theta)/lambda)
    xray : bool, optional
        If True, compute X-ray scattering factors. If False (default),
        compute electron scattering factors.
        
    Returns
    -------
    profiles : ndarray
        Array of shape (n_elements, n_points) with scattering profiles
    """
    n_elements = len(elements)
    n_points = len(s_values)
    profiles = np.zeros((n_elements, n_points), dtype=np.float64)
    k2_values = s_values ** 2 if not xray else None
    
    for i, element in enumerate(elements):
        if element not in LOBATO_PARAMETERS:
            raise ValueError(f"Element '{element}' not found in Lobato parameters")
        
        params = LOBATO_PARAMETERS[element]
        
        if xray:
            # For X-rays: f(k) where k = s
            profiles[i] = x_ray_scattering_factor(s_values, params)
        else:
            # For electrons: f(k^2) where k^2 = s^2
            # Keep a single vectorized evaluation (no per-point loop)
            profiles[i] = electron_scattering_factor(k2_values, params)
    
    return profiles


class LobatoScatteringCalculator:
    """
    Calculator for electron and X-ray scattering factors using Lobato parametrization.
    
    This provides a simplified interface similar to abtem's LobatoParametrization.
    """
    
    def __init__(self):
        self.parameters = LOBATO_PARAMETERS
    
    def line_profiles(self, elements, cutoff, sampling, name="scattering_factor"):
        """
        Compute scattering factor line profiles.
        
        Parameters
        ----------
        elements : list of str
            Element symbols
        cutoff : float
            Maximum s value (1/Angstrom)
        sampling : float
            Sampling interval in s (1/Angstrom)
        name : str, optional
            Type of scattering factor: "scattering_factor" for electrons,
            "x_ray_scattering_factor" for X-rays
            
        Returns
        -------
        result : SimpleNamespace
            Object with 'array' attribute containing profiles of shape 
            (n_elements, n_points)
        """
        from types import SimpleNamespace
        
        # Generate s grid
        n_points = int(np.ceil(cutoff / sampling))
        s_values = np.arange(n_points) * sampling
        
        # Determine if X-ray or electron
        xray = (name == "x_ray_scattering_factor")
        
        # Compute profiles
        profiles = compute_scattering_profile(elements, s_values, xray=xray)
        
        # Return in a format compatible with abtem's return structure
        result = SimpleNamespace(array=profiles)
        return result
    
    def get_parameters(self, element):
        """Get Lobato parameters for a specific element."""
        if element not in self.parameters:
            raise ValueError(f"Element '{element}' not found")
        return np.array(self.parameters[element])
