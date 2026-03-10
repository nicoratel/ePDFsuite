from .filereader import load_data
from .recalibration import recalibrate_no_beamstop, recalibrate_with_beamstop, recalibrate_with_beamstop_noponi
from .pdf_extraction import compute_ePDF
from pyFAI import load
import fabio
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import hyperspy.api as hs


class SAEDProcessor:
    def __init__(self, image_file, poni_file = None, mask=None, mtf_file=None, verbose=False):
        """
        Initialize a SAED data processor.
        
        Parameters
        ----------
        image_file : str
            SAED data file in DM4, DM3, tif, tiff format
        poni_file : str, optional
            Geometric calibration file in .poni format
        mask : str, optional
            Mask file for the image
        verbose : bool
            If True, prints metadata information
        """
        self.dm4_file = image_file
        self.poni_file = poni_file
        self.beamstop = True
        self.initial_center = None  # To be set by user after inspection via plot()
        metadata, img = load_data(image_file,verbose=verbose)
        self.metadata = metadata
        self.img = img
        if mask is not None:
            mask_img = fabio.open(mask)
            self.mask = mask_img.data
        else:
            self.mask = np.zeros_like(self.img)
        
        if poni_file is not None:
            self.ai = load(poni_file)
            self.use_pyfai=True
            if mask is not None:
                mask_img = fabio.open(mask)
                self.mask = mask_img.data.astype(bool)
            else:
                self.mask = np.zeros(self.img.shape, dtype=bool)
        else:
            img = hs.load(image_file)
            self.use_pyfai=False
            self.scale = img.axes_manager[0].scale# in nm/pixel
            self.units = img.axes_manager[0].units
            print(f'scale = {self.scale}, unit = {self.units}')
        if mtf_file is not None:
            self.ismtf = True
            self.q_mtf, self.mtf_values = np.loadtxt(mtf_file, unpack=True)
        else:
            self.ismtf = False

            


    def integrate(self, dm4_file=None, npt=2500, initial_center=None, plot=False):
        """
        Integrate SAED pattern to 1D.
        
        Parameters
        ----------
        dm4_file : str, optional
            File to integrate. If None, uses self.dm4_file
        npt : int
            Number of points in output
        initial_center : tuple, optional
            Initial center (x, y) in pixels. If None, uses self.initial_center
        plot : bool
            If True, displays the integrated pattern
        
        Returns
        -------
        q : array
            Scattering vector
        I : array
            Integrated intensity
        """
        if dm4_file is None:
            dm4_file = self.dm4_file
        
        # Use provided initial_center, or fall back to self.initial_center
        center = initial_center if initial_center is not None else self.initial_center

        if self.use_pyfai:
            # Load the image data for the specified file
            _, img_data = load_data(dm4_file, verbose=False)
            
            if self.beamstop:
                self.ai = recalibrate_with_beamstop(dm4_file, self.poni_file, initial_center=center) # seek beamcentre
            else: # recalibrate with beamstop correction
                self.ai = recalibrate_no_beamstop(dm4_file, self.poni_file)
            
            q, I = self.ai.integrate1d(img_data, npt, mask = self.mask, unit="q_A^-1", polarization_factor=0.99)
            
        else: # intégration personnalisée sans pyFAI, pour les cas où il n'y a pas de fichier de calibration ou que les images ont des résolutions différentes
            # Charger l'image
            img = hs.load(dm4_file)
            
            # Recalibrer le centre
            center_x, center_y = recalibrate_with_beamstop_noponi(img.data, threshold_rel=0.5, min_size=50, initial_center=center, plot=False)
            
            # Calculer le profil radial
            y, x = np.indices(img.data.shape)
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Arrondir les distances pour créer des bins
            r_int = r.astype(int)
            
            # Calculer le profil radial (moyenne azimutale)
            radial_bins = np.bincount(r_int.ravel(), weights=img.data.ravel())
            radial_counts = np.bincount(r_int.ravel())
            I = radial_bins / radial_counts
    
            # Axe q (distances radiales)
            q = np.arange(len(I))#pixel array
            
            q = q * self.scale  # Convertir les distances en unités physiques (ex: nm)
            q *= 2*np.pi  # Convertir en q (Å^-1) si les distances sont en nm
            if self.units == '1/nm':
                q /= 10  # Convertir en Å^-1 si les distances sont en nm

        # perform mtf correction if mtf data is available
        if self.ismtf:
            mtf_interp = np.interp(q, self.q_mtf, self.mtf_values, left=1.0, right=1.0)
            mask_mtf = mtf_interp > 0.1
            I[mask_mtf] /= mtf_interp[mask_mtf]

        if plot:
            plt.figure()
            plt.semilogy(q, I)
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Azimuthally Integrated SAED Pattern')
            plt.grid()
            plt.show()
        
        return q, I
    
    def plot(self,vmin=-4, vmax=0,cmap='gray'):
        plt.figure()
        plt.imshow(self.img/np.max(self.img), cmap=cmap,norm = LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
    
    def plot_recalibrated_image(self, initial_center=None):
        """
        Plot the diffraction image with detected center and rings.
        
        Parameters
        ----------
        initial_center : tuple, optional
            Initial center (x, y) in pixels. If None, uses self.initial_center
        """
        center = initial_center if initial_center is not None else self.initial_center
        
        if self.use_pyfai:
            if self.beamstop:
                _ = recalibrate_with_beamstop(self.dm4_file, self.poni_file, initial_center=center, plot=True)
            else:
                _ = recalibrate_no_beamstop(self.dm4_file, self.poni_file, plot=True)
        else:
            _ = recalibrate_with_beamstop_noponi(self.img, initial_center=center, plot=True)

    def extract_epdf(self,
                     ref_diffraction_image=None,
                     ref_poni_file=None,
                     composition = 'Au',                     
                     rmin=0.1,
                     rmax=50.0,
                     rstep=0.01,
                     outputfile=None,
                     interactive = True,
                     plot = False,
                     mtf_file=None,
                     bgscale=1,
                     qmin=1.5,
                     qmax=24,
                     qmaxinst=24,
                     rpoly=1.4,
                     initial_center=None,
                     initial_center_ref=None):
        """
        Extract ePDF from SAED data (legacy method, prefer using extract_epdf function).
        
        This method creates a temporary SAEDProcessor for the reference and calls
        the standalone extract_epdf function.
        
        Parameters
        ----------
        ref_diffraction_image : str, optional
            Path to reference diffraction image
        ref_poni_file : str, optional
            Path to reference poni file (if different from sample)
        composition : str
            Chemical composition
        rmin, rmax, rstep : float
            PDF r-range parameters
        outputfile : str, optional
            Output filename
        interactive : bool
            If True, shows interactive GUI
        plot : bool
            If True, plots results (non-interactive mode)
        bgscale, qmin, qmax, qmaxinst, rpoly : float
            PDF computation parameters
        initial_center : tuple, optional
            Override self.initial_center for this call
        initial_center_ref : tuple, optional
            Initial center for reference image
        
        Returns
        -------
        results : PDFResultsReference or tuple
            PDF results
        """
        # Create reference processor if provided
        ref_processor = None
        if ref_diffraction_image is not None:
            ref_processor = SAEDProcessor(
                ref_diffraction_image,
                poni_file=ref_poni_file if ref_poni_file is not None else self.poni_file,
                beamstop=self.beamstop,
                verbose=False
            )
            # Set initial center for reference
            if initial_center_ref is not None:
                ref_processor.initial_center = initial_center_ref
            elif initial_center is not None:
                ref_processor.initial_center = initial_center
        
        # Temporarily override initial_center if provided
        original_center = self.initial_center
        if initial_center is not None:
            self.initial_center = initial_center
        
        try:
            # Call standalone function
            return extract_epdf(
                sample_processor=self,
                ref_processor=ref_processor,
                composition=composition,
                rmin=rmin,
                rmax=rmax,
                rstep=rstep,
                outputfile=outputfile,
                interactive=interactive,
                plot=plot,
                bgscale=bgscale,
                qmin=qmin,
                qmax=qmax,
                qmaxinst=qmaxinst,
                rpoly=rpoly
            )
        finally:
            # Restore original center
            self.initial_center = original_center
        # retrive wavelength from metadata
        wavelength = self.metadata['wavelength']
        camera = self.metadata['camera_title']
        sample_diffraction_image = self.dm4_file

        # add attributes to class for further use in PDFinteractive
        self.ref_diffraction_image = ref_diffraction_image
        self.composition = composition
        # load sample and reference images
        info , sample_data = load_data(sample_diffraction_image, verbose=False)
        if ref_diffraction_image:
            _, ref_data = load_data(ref_diffraction_image, verbose=False)
        else:
            ref_data = None    
        

        # Recalibrate centre
        if self.use_pyfai:
            # Use the already calibrated AzimuthalIntegrator from self.ai if available,
            # otherwise recalibrate
            if hasattr(self, 'ai') and self.ai is not None:
                ai = self.ai
            else:
                # Initialize Azimuthal Integrator from poni file and recalibrate
                if not self.beamstop:
                    ai = recalibrate_no_beamstop(
                    dm4file=sample_diffraction_image,
                    ponifile=self.poni_file,
                    )
                else:
                    ai = recalibrate_with_beamstop(
                    dm4file=sample_diffraction_image,
                    ponifile=self.poni_file,
                    threshold_rel=0.5,
                    min_size=80,
                    initial_center=initial_center,
                    plot=False
                    )
                # Store the calibrated ai for future use
                self.ai = ai
            
            # Integrate sample image
            q_sample, intensity_sample = ai.integrate1d(
                sample_data,
                npt=2500,
                unit="q_A^-1")
            
            # Integrate reference image
            if ref_data is not None:
                # If ref_poni_file is provided or images have different resolutions
                if ref_poni_file is not None or ref_data.shape != sample_data.shape:
                    # Recalibrate separately for reference
                    poni_for_ref = ref_poni_file if ref_poni_file is not None else self.poni_file
                    if not self.beamstop:
                        ai_ref = recalibrate_no_beamstop(
                            dm4file=ref_diffraction_image,
                            ponifile=poni_for_ref
                        )
                    else:
                        ai_ref = recalibrate_with_beamstop(
                            dm4file=ref_diffraction_image,
                            ponifile=poni_for_ref,
                            threshold_rel=0.5,
                            min_size=80,
                            initial_center=initial_center_ref if initial_center_ref is not None else initial_center,
                            plot=False
                        )
                else:
                    # Use same ai if resolutions match
                    ai_ref = ai
                
                q_ref, intensity_ref = ai_ref.integrate1d(
                    ref_data,
                    npt=2500,
                    unit="q_A^-1")
        else:# no poni file, use custom integration
            q_sample, intensity_sample = self.integrate(self.dm4_file, initial_center=initial_center, plot=False)
            q_ref, intensity_ref = self.integrate(dm4_file= ref_diffraction_image, initial_center=initial_center_ref if initial_center_ref is not None else initial_center, plot=False) if ref_data is not None else (None, None)
        
        if outputfile is None:
            # repalce None by default name based on sample image name
            outputfile = sample_diffraction_image.split('.')[0] + '_pdf.gr'

        if interactive:
            # Create PDFInteractive object
            pdf_interactive = PDFInteractive(
                q_sample,
                intensity_sample,
                composition=composition,
                rmin=rmin,
                rmax=rmax,
                rstep=rstep,
                ref_diffraction_image=ref_diffraction_image if ref_diffraction_image is not None else None,
                outputfile=outputfile,
                SAEDProcessor=self,
                initial_center=initial_center,
                initial_center_ref=initial_center_ref,
                xray=False
            )
            # Si une méthode d'export existe, l'appeler ici
            if hasattr(pdf_interactive, 'save_results'):
                pdf_interactive.save_results(outputfile)
            pdf_interactive.show()
            # Store the interactive object for access to results
            self.pdf_interactive = pdf_interactive
            # Return a reference to the results that will be updated by sliders
            return PDFResultsReference(pdf_interactive)
        else:
            print('Compute PDF with given parameters')
            r,G = compute_ePDF(
                q_sample,
                intensity_sample,
                composition,
                Iref=intensity_ref if ref_data is not None else None,
                bgscale=bgscale,
                qmin=qmin,
                qmax=qmax,
                qmaxinst=qmaxinst,
                rmin=rmin,
                rmax=rmax,
                rstep=rstep,
                rpoly=rpoly,
                Lorch=True,
                plot=plot)
            # header should have same architecture as .gr files from pdfgetx3 for compatibility with PDFBatchAnalysis
            header  = '[DEFAULT]\n\nversion = ePDFsuite 1.0\n\n'
            header += '#input and output specifications\n'
            header += 'dataformat = q_A \n'
            header +=f'inputfile = {sample_diffraction_image}\n'
            header +=f'backgroundfile = {ref_diffraction_image}\n'
            header += 'outputtype = gr\n\n'
            header += '#PDF calculation setup\n'
            header += 'mode = electrons\n'        
            header +=f'wavelength = {self.metadata.get("wavelength", "unknown"):.4f}\n'
            header += 'twothetazero = 0\n'        
            header +=f'composition={composition} \n'
            header +=f'bgscale = {1:.2f} \n'
            header +=f'rpoly = {1.4} \n'
            header +=f'qmaxinst = {np.max(q_sample):.2f}\n'
            header +=f'qmin = {np.min(q_sample):.2f} \n'
            header +=f'qmax = {np.max(q_sample):.2f}  \n'
            header +=f'rmin = {0:.2f} \n'
            header +=f'rmax = {50:.2f} \n'
            header +=f'rstep = {0.01:.2f}\n\n'
            header += '# End of config --------------------------------------------------------------\n#### start data\n\n'
            header += '#S 1 \n'
            header += '#L r(Å)  G(Å$^{-2}$)'

            np.savetxt(outputfile, np.column_stack((r, G)),header=header,delimiter=' ',comments='')
            print(f'PDF saved to {outputfile}')
            return r, G



# ------------------
# Standalone ePDF extraction function
# ------------------
def extract_epdf(sample_processor,
                 ref_processor=None,
                 composition='Au',
                 rmin=0.1,
                 rmax=50.0,
                 rstep=0.01,
                 outputfile=None,
                 interactive=True,
                 plot=False,
                 bgscale=1,
                 qmin=1.5,
                 qmax=24,
                 qmaxinst=24,
                 rpoly=1.4):
    """
    Extract electron pair distribution function (ePDF) from SAED data.
    
    This standalone function provides a clean interface for ePDF extraction,
    treating sample and reference data symmetrically via SAEDProcessor instances.
    
    Parameters
    ----------
    sample_processor : SAEDProcessor
        Processor for sample diffraction data. Should have initial_center set if needed.
    ref_processor : SAEDProcessor, optional
        Processor for reference/background diffraction data. If None, no background subtraction.
    composition : str
        Chemical composition (e.g., 'Au', 'Fe2O3')
    rmin, rmax, rstep : float
        PDF r-range parameters (Angstroms)
    outputfile : str, optional
        Path to save PDF results. Auto-generated if None.
    interactive : bool
        If True, shows interactive parameter adjustment GUI
    plot : bool
        If True, plots results in non-interactive mode
    bgscale, qmin, qmax, qmaxinst, rpoly : float
        PDF computation parameters
    
    Returns
    -------
    results : PDFResultsReference or tuple
        Interactive mode: PDFResultsReference with .r and .g properties
        Non-interactive mode: tuple (r, G)
    
    Examples
    --------
    >>> # Setup processors
    >>> sample = SAEDProcessor('sample.dm4', poni_file='calib.poni')
    >>> sample.plot()  # Inspect to determine center
    >>> sample.initial_center = (335, 275)  # Set after inspection
    >>> 
    >>> ref = SAEDProcessor('reference.dm4', poni_file='calib.poni')
    >>> ref.initial_center = (324, 257)
    >>> 
    >>> # Extract PDF
    >>> results = extract_epdf(sample, ref, composition='Au', interactive=True)
    >>> r, g = results  # Access results
    """
    # Integrate sample
    q_sample, intensity_sample = sample_processor.integrate(plot=False)
    
    # Integrate reference if provided
    if ref_processor is not None:
        q_ref, intensity_ref = ref_processor.integrate(plot=False)
    else:
        q_ref, intensity_ref = None, None


    # Generate output filename if not provided
    if outputfile is None:
        outputfile = sample_processor.dm4_file.split('.')[0] + '_pdf.gr'
    
    if interactive:
        # Create PDFInteractive object
        pdf_interactive = PDFInteractive(
            q_sample,
            intensity_sample,
            composition=composition,
            rmin=rmin,
            rmax=rmax,
            rstep=rstep,
            ref_diffraction_image=ref_processor.dm4_file if ref_processor is not None else None,
            outputfile=outputfile,
            SAEDProcessor=sample_processor,
            initial_center=sample_processor.initial_center,
            initial_center_ref=ref_processor.initial_center if ref_processor is not None else None,
            xray=False
        )
        # Si une méthode d'export existe, l'appeler ici
        if hasattr(pdf_interactive, 'save_results'):
            pdf_interactive.save_results(outputfile)
        pdf_interactive.show()
        # Store the interactive object for access to results
        sample_processor.pdf_interactive = pdf_interactive
        # Return a reference to the results that will be updated by sliders
        return PDFResultsReference(pdf_interactive)
    else:
        print('Compute PDF with given parameters')
        r, G = compute_ePDF(
            q_sample,
            intensity_sample,
            composition,
            Iref=intensity_ref if ref_processor is not None else None,
            bgscale=bgscale,
            qmin=qmin,
            qmax=qmax,
            qmaxinst=qmaxinst,
            rmin=rmin,
            rmax=rmax,
            rstep=rstep,
            rpoly=rpoly,
            Lorch=True,
            plot=plot)
        
        # Generate header for .gr file
        header = '[DEFAULT]\n\nversion = ePDFsuite 1.0\n\n'
        header += '#input and output specifications\n'
        header += 'dataformat = q_A \n'
        header += f'inputfile = {sample_processor.dm4_file}\n'
        header += f'backgroundfile = {ref_processor.dm4_file if ref_processor is not None else "None"}\n'
        header += 'outputtype = gr\n\n'
        header += '#PDF calculation setup\n'
        header += 'mode = electrons\n'
        header += f'wavelength = {sample_processor.metadata.get("wavelength", "unknown"):.4f}\n'
        header += 'twothetazero = 0\n'
        header += f'composition={composition} \n'
        header += f'bgscale = {bgscale:.2f} \n'
        header += f'rpoly = {rpoly:.2f} \n'
        header += f'qmaxinst = {qmaxinst:.2f}\n'
        header += f'qmin = {qmin:.2f} \n'
        header += f'qmax = {qmax:.2f}  \n'
        header += f'rmin = {rmin:.2f} \n'
        header += f'rmax = {rmax:.2f} \n'
        header += f'rstep = {rstep:.2f}\n\n'
        header += '# End of config --------------------------------------------------------------\n'
        header += '#### start data\n\n'
        header += '#S 1 \n'
        header += '#L r(Å)  G(Å$^{-2}$)\n'
        
        # Write output file
        with open(outputfile, 'w') as f:
            f.write(header)
            for ri, Gi in zip(r, G):
                f.write(f'{ri:.4f}  {Gi:.6f}\n')
        
        print(f'PDF saved to {outputfile}')
        return r, G


# ------------------
# Results Reference Class
# ------------------
class PDFResultsReference:
    """
    A reference object that allows unpacking of PDF results from interactive mode.
    
    This class acts as a wrapper around PDFInteractive, providing access to the
    most recently computed r and G values through tuple unpacking.
    
    Usage:
        r, g = proc.extract_epdf(interactive=True)
        # After adjusting sliders, r and g will contain the latest values
        print(r, g)  # Access the arrays directly
    """
    
    def __init__(self, pdf_interactive):
        """
        Initialize with a PDFInteractive instance.
        
        Args:
            pdf_interactive: The PDFInteractive object containing the results
        """
        self._pdf_interactive = pdf_interactive
    
    def __iter__(self):
        """
        Allow tuple unpacking: r, g = reference
        
        Returns the latest computed r and G arrays.
        """
        if self._pdf_interactive.last_r is None or self._pdf_interactive.last_G is None:
            print("⚠️ Aucune valeur disponible. Ajustez les paramètres avec les sliders pour générer r et G.")
            return iter([None, None])
        return iter([self._pdf_interactive.last_r, self._pdf_interactive.last_G])
    
    def __repr__(self):
        """String representation of the reference."""
        if self._pdf_interactive.last_r is None:
            return "PDFResultsReference(no data yet - adjust sliders to compute)"
        return f"PDFResultsReference(r: {len(self._pdf_interactive.last_r)} points, " \
               f"r_range=[{self._pdf_interactive.last_r.min():.2f}, {self._pdf_interactive.last_r.max():.2f}] Å)"
    
    @property
    def r(self):
        """Direct access to r array."""
        return self._pdf_interactive.last_r
    
    @property
    def g(self):
        """Direct access to G array."""
        return self._pdf_interactive.last_G


# ------------------
# Interactive GUI Class
# ------------------
class PDFInteractive:
    """
    Interactive widget-based interface for PDF parameter optimization.
    
    This class provides real-time parameter adjustment with immediate visual feedback,
    making it easier to optimize PDF processing parameters interactively.
    """
    
    def __init__(self,
                 q,
                 Iexp,
                 composition,
                 ref_diffraction_image=None,
                 rmin=0,
                 rmax=50,
                 rstep=0.01,
                 xray: bool = False,
                 outputfile: str = './pdf_results.csv',
                 SAEDProcessor=None,
                 initial_center=None,
                 initial_center_ref=None):
        """
        Initialize the interactive PDF interface.
        
        Args:
            q (array): Scattering vector values
            Iexp (array): Experimental intensity data
            composition (str): Chemical formula
            Iref (array, optional): Reference background
            rmin (float): Minimum r for PDF
            rmax (float): Maximum r for PDF
            rstep (float): Step size for r
            xray (bool): If True, use X-ray scattering factors
            outputfile (str): Default output filename for saving results
            SAEDProcessor: SAEDProcessor instance for metadata access
            initial_center (tuple): Initial center coordinates as (x, y) in pixels for sample
            initial_center_ref (tuple): Initial center coordinates as (x, y) in pixels for reference
        """
        # Import widgets here to avoid issues when they're not needed
        import ipywidgets as widgets
        from IPython.display import display
        # Store widgets reference for use in the class
        self.widgets = widgets
        self.display = display
        
        print('Adjust sliders to optimize PDF parameters. Click "Save" to export results.')
        # Retrieve useful metadata from SAEDProcessor if provided
        if SAEDProcessor is not None:
            self.wavelength = SAEDProcessor.metadata.get('wavelength', None)
            self.camera = SAEDProcessor.metadata.get('camera_title', None)
            self.sample_diffraction_image = SAEDProcessor.dm4_file
            self.ref_diffraction_image = ref_diffraction_image if ref_diffraction_image is not None else None
            self.composition = composition
        else:
            self.wavelength = None
            self.camera = None
            self.sample_diffraction_image = None
            self.ref_diffraction_image = None
            self.composition = None
        
        # Store initial_center for later use
        self.initial_center = initial_center
        self.initial_center_ref = initial_center_ref
        
        # integrate reference image if provided
        if ref_diffraction_image is not None and SAEDProcessor is not None:
            _, Iref = SAEDProcessor.integrate(dm4_file=self.ref_diffraction_image, initial_center=initial_center_ref if initial_center_ref is not None else initial_center, plot=False)
        else:
            Iref = None
        
        # Store PDF computation parameters

        self.xray = xray
        self.pdf_config = dict(
            q=q, Iexp=Iexp, Iref=Iref, composition=composition,
            rmin=rmin, rmax=rmax, rstep=rstep,
        )
        
        # Storage for last computed results (for saving)
        self.last_r = None
        self.last_G = None

        # Create parameter control sliders
        self.bgscale_slider = self.widgets.FloatSlider(
            value=1, min=0, max=2, step=0.01, 
            description="bgscale", readout_format=".2f"
        )
        self.qmin_slider = self.widgets.FloatSlider(
            value=1.5, min=np.min(q), max=min(24,np.max(q)), step=0.01,
            description="qmin", readout_format=".2f"
        )
        self.qmax_slider = self.widgets.FloatSlider(
            value=min(24,np.max(q)), min=np.min(q), max=np.max(q), step=0.01,
            description="qmax", readout_format=".2f"
        )
        self.qmaxinst_slider = self.widgets.FloatSlider(
            value=min(24,np.max(q)), min=np.min(q), max=np.max(q), step=0.01,
            description="qmaxinst", readout_format=".2f"
        )
        self.rpoly_slider = self.widgets.FloatSlider(
            value=1.4, min=0.1, max=2.5, step=0.01,
            description="rpoly", readout_format=".2f"
        )
        
        self.lorch_checkbox = self.widgets.Checkbox(
            value=True,
            description="apply Lorch window correction to eliminate termination ripples",
            indent=False)

        # Save button for exporting results
        self.save_button = self.widgets.Button(description="💾 Save", button_style="success")
        self.save_button.on_click(lambda b: self.save_results(b, outputfile))

        # Organize widgets in vje veux quelque xhiose de plus simple. Je vais me débrouillerertical layout
        self.sliders = self.widgets.VBox([
            self.bgscale_slider,
            self.qmin_slider,
            self.qmax_slider,
            self.qmaxinst_slider,
            self.rpoly_slider,
            self.lorch_checkbox,
            self.save_button])

        # Output area for plots
        self.plot_output = self.widgets.Output()

        # Link sliders to update function for real-time feedback
        self.widgets.interactive_output(self.update_plot, {
            "bgscale": self.bgscale_slider,
            "qmin": self.qmin_slider,
            "qmax": self.qmax_slider,
            "qmaxinst": self.qmaxinst_slider,
            "rpoly": self.rpoly_slider,
            "lorch": self.lorch_checkbox})

    def update_plot(self, bgscale, qmin, qmax, qmaxinst, rpoly, lorch):
        """
        Update the PDF calculation and plots when parameters change.
        
        This function is called automatically when any slider value changes.
        """
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            # Recompute PDF with new parameters
            r, G = compute_ePDF(
                **self.pdf_config,
                bgscale=bgscale, qmin=qmin, qmax=qmax,
                qmaxinst=qmaxinst, rpoly=rpoly, plot=True, Lorch=lorch)
            # Store results for potential saving
            self.last_r, self.last_G = r, G

    def save_results(self, b, outputfile='./pdf_results.gr'):
        """
        Save the last computed PDF results to TXT file with metadata.
        
        Args:
            b: Button widget (unused, required by widget callback signature)
            outputfile: Output filename (default: './pdf_results.gr')
        """
        if self.last_r is None or self.last_G is None:
            print("⚠️ Aucun résultat à sauvegarder (génère d'abord un plot).")
            return

        # make header similar to pdfgetx3 for further compatibility with PDFBatchANalayis
        # header should have same architecture as .gr files from pdfgetx3 for compatibility with PDFBatchAnalysis
        header  = '[DEFAULT]\n\nversion = ePDFsuite 1.0\n\n'
        header += '# input and output specifications\n'
        header +=f'camera = {self.camera} \n'
        header +=f'inputfile = {self.sample_diffraction_image}\n'
        header +=f'backgroundfile = {self.ref_diffraction_image}\n'
        header += 'outputtype = gr\n\n'
        header += '#PDF calculation setup\n'
        header += 'mode = electrons\n'        
        header +=f'wavelength = {self.wavelength:.4f}\n'
        header += 'twothetazero = 0\n'        
        header +=f'composition={self.composition} \n'
        header +=f'bgscale = {1:.2f} \n'
        header +=f'rpoly = {1.4} \n'
        header +=f'qmaxinst = {self.qmaxinst_slider.value:.2f}\n'
        header +=f'qmin = {self.qmin_slider.value:.2f} \n'
        header +=f'qmax = {self.qmax_slider.value:.2f}  \n'
        header +=f'rmin = {0:.2f} \n'
        header +=f'rmax = {50:.2f} \n'
        header +=f'rstep = {0.01:.2f}\n\n'
        header += '# End of config --------------------------------------------------------------\n#### start data\n\n'
        header += '#S 1 \n'
        header += '#L r(Å)  G(Å$^{-2}$)'

        np.savetxt(outputfile, np.column_stack((self.last_r, self.last_G)),header=header,delimiter=' ',comments='')
        
        

    def show(self):
        """
        Display the interactive interface.
        
        Creates a horizontal layout with sliders on the left and plots on the right.
        """
        # Generate initial plot with default parameter values BEFORE displaying UI
        # This ensures last_r and last_G are immediately available for unpacking
        self.update_plot(
            self.bgscale_slider.value, self.qmin_slider.value,
            self.qmax_slider.value, self.qmaxinst_slider.value,
            self.rpoly_slider.value, self.lorch_checkbox.value
        )
        
        ui = self.widgets.HBox([self.sliders, self.plot_output])
        self.display(ui)


def extract_ePDF_from_mutliple_files(dm4_files,
                                     ref_diffraction_image=None,
                                     ref_poni_file=None,
                                     composition = 'Au',
                                     rmin=0.1,
                                     rmax=50.0,
                                     rstep=0.01,
                                     qmin=1.5,
                                     qmax=24,
                                     qmaxinst=24,
                                     bgscale=1.0,
                                     rpoly=1.4,
                                     outputfile=None,
                                     interactive = True,
                                     poni_file=None,
                                     beamstop=False,
                                     plot=False,
                                     verbose=False):
        """
        Docstring pour extract_ePDF_from_mutliple_files
        
        
        :param dm4_files: list of file paths to SAED data files in DM4, DM3, tif, tiff format
        :param ref_diffraction_image: file path to reference diffraction image
        :param ref_poni_file: file path to PONI file for reference (if different resolution from sample)
        :param composition: chemical composition of the sample
        :param rmin: minimum r value for PDF calculation
        :param rmax: maximum r value for PDF calculation
        :param rstep: step size for r values in PDF calculation
        :param outputfile: file path to save the output PDF data
        :param interactive: whether to run in interactive mode
        :param poni_file: file path to PONI file for calibration
        :param beamstop: whether to apply beamstop correction
        :param verbose: whether to print detailed information during processing
        """



        I_array = []
        q_array = []
        # Compute average profile from multiple files
        for dm4_file in dm4_files:
            proc = SAEDProcessor(dm4_file, poni_file, beamstop, verbose)
            q,I = proc.integrate(dm4_file, plot=False)
            q_array.append(q)
            I_array.append(I)
        
        # Use the q range from the first file as reference
        q = q_array[0]
        # Interpolate all I arrays to the same q grid
        from scipy.interpolate import interp1d
        I_interpolated = []
        for i, I in enumerate(I_array):
            if len(I) != len(q):
                f = interp1d(q_array[i], I, kind='linear', bounds_error=False, fill_value='extrapolate')
                I_interp = f(q)
            else:
                I_interp = I
            I_interpolated.append(I_interp)
        
        average_radial_profile = np.mean(I_interpolated, axis=0)
        # Integrate reference image
        if ref_diffraction_image is not None:
            if ref_poni_file is not None:
                # Use separate processor for reference if different poni file provided
                proc_ref = SAEDProcessor(ref_diffraction_image, ref_poni_file, beamstop, verbose)
                qref, Iref = proc_ref.integrate(ref_diffraction_image, plot=False)
            else:
                # Use same processor (poni file) for reference
                qref, Iref = proc.integrate(dm4_file=ref_diffraction_image, plot=False)
        else:
            qref, Iref = None, None

        # extract PDF using average profile and reference profile
        if not interactive:
            r,G = compute_ePDF(
                q,
                average_radial_profile,
                composition,
                Iref=Iref if ref_diffraction_image is not None else None,
                bgscale=bgscale,
                qmin=qmin,
                qmax=qmax,
                qmaxinst=qmaxinst,
                rmin=rmin,
                rmax=rmax,
                rstep=rstep,
                rpoly=rpoly,
                Lorch=True,
                plot=True)
            
            # header should have same architecture as .gr files from pdfgetx3 for compatibility with PDFBatchAnalysis
            header  = '[DEFAULT]\n\nversion = ePDFsuite 1.0\n\n'
            header += '#input and output specifications\n'
            header += 'dataformat = q_A \n'
            header +=f'inputfile = {dm4_files}\n'
            header +=f'backgroundfile = {ref_diffraction_image}\n'
            header += 'outputtype = gr\n\n'
            header += '#PDF calculation setup\n'
            header += 'mode = electrons\n'        
            header +=f'wavelength = {proc.metadata.get("wavelength", "unknown"):.4f}\n'
            header += 'twothetazero = 0\n'        
            header +=f'composition={composition} \n'
            header +=f'bgscale = {bgscale:.2f} \n'
            header +=f'rpoly = {rpoly} \n'
            header +=f'qmaxinst = {qmaxinst:.2f}\n'
            header +=f'qmin = {qmin:.2f} \n'
            header +=f'qmax = {qmax:.2f}  \n'
            header +=f'rmin = {rmin:.2f} \n'
            header +=f'rmax = {rmax:.2f} \n'
            header +=f'rstep = {rstep:.2f}\n\n'
            header += '# End of config --------------------------------------------------------------\n#### start data\n\n'
            header += '#S 1 \n'
            header += '#L r(Å)  G(Å$^{-2}$)'

            np.savetxt(outputfile, np.column_stack((r, G)),header=header,delimiter=' ',comments='')
            print(f'PDF saved to {outputfile}')
            if plot:
                plt.figure()
                plt.plot(r, G)
                plt.xlabel('r (Å)')
                plt.ylabel('G(r) (Å$^{-2}$)')
                plt.title('ePDF')
                plt.grid()
                plt.show()
            return r, G
        else:
            # Create PDFInteractive object
            pdf_interactive = PDFInteractive(
                q,
                average_radial_profile,
                composition=composition,
                rmin=rmin,
                rmax=rmax,
                rstep=rstep,
                ref_diffraction_image=ref_diffraction_image if ref_diffraction_image is not None else None,
                outputfile=outputfile,
                SAEDProcessor=proc,
                xray=False
            )
            # Si une méthode d'export existe, l'appeler ici
            if hasattr(pdf_interactive, 'save_results'):
                pdf_interactive.save_results(outputfile)
            pdf_interactive.show()
       
