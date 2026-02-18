"""ePDFsuite package"""

# Imports absolus pour charger les modules
from ePDFsuite.ePDFsuite import SAEDProcessor
from ePDFsuite.filereader import load_data

__version__ = "0.1.0"

# Ce que les utilisateurs peuvent importer
__all__ = [
    'SAEDProcessor',
    'load_data',
    
]