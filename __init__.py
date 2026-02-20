"""ePDFsuite package"""

# Imports absolus pour charger les modules
from .ePDFsuite import SAEDProcessor
from .ePDFsuite import extract_ePDF_from_mutliple_files
from .filereader import load_data
# from .pdfanalysis import perform_automatic_pdf_analysis  # Module not found

__version__ = "0.1.1"

# Ce que les utilisateurs peuvent importer
__all__ = [
    'SAEDProcessor',
    'load_data',
	'extract_ePDF_from_multiple_files',
	'perform_automatic_pdf_analysis',
    
]
