from filereader import load_data
from recalibration import recalibrate_no_beamstop, recalibrate_with_beamstop
from pdf_extraction import compute_ePDF
from pyFAI import load
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import ipywidgets as widgets
from IPython.display import display


class SAEDProcessor:
    def __init__(self, dm4_file, poni_file,beamstop = False,verbose=False):
        """
        Docstring pour __init__
        
        :param self: Description
        :param dm4_file: SAED data file in DM4, DM3, tif, tiff format
        :param poni_file: geometric calibration file in .poni format
        :param beamstop: Boolean indicating presence of beamstop on the image
        """
        self.dm4_file = dm4_file
        self.poni_file = poni_file
        self.beamstop = beamstop
        metadata, img = load_data(dm4_file,verbose=verbose)
        self.metadata = metadata
        self.img = img
        self.ai = load(poni_file)


    def integrate(self, npt=2500, beamstop=False,plot=False):
        if not beamstop:
            self.ai = recalibrate_no_beamstop(self.dm4_file, self.poni_file)
        else:
            self.ai = recalibrate_with_beamstop(self.dm4_file, self.poni_file, threshold_rel=0.5, min_size=200)
        q, i = self.ai.integrate1d(self.img, npt, unit="q_A^-1", polarization_factor=0.99)
        if plot:
            plt.figure()
            plt.semilogy(q, i)
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Azimuthally Integrated SAED Pattern')
            plt.grid()
            plt.show()
        return q, i
    
    def plot(self,vmin=-4, vmax=0):
        plt.figure()
        plt.imshow(self.img/np.max(self.img), cmap='gray',norm = LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
    
    def extract_epdf(self,
                     ref_diffraction_image=None,
                     composition = 'Au',                     
                     rmin=0.1,
                     rmax=50.0,
                     rstep=0.01,
                     outputfile=None,
                     interactive = True,
                     plot = False):
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
        
        # Initialize Azimuthal Integrator from poni file
        ai=load(self.poni_file)

        # Recalibrate centre
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
            plot=False
            )

        # Integrate sample and reference images
        q_sample, intensity_sample = ai.integrate1d(
            sample_data,
            npt=2500,
            unit="q_A^-1")
        if ref_data is not None:
            q_ref, intensity_ref = ai.integrate1d(
                ref_data,
                npt=2500,
                unit="q_A^-1")
        
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
                Iref=intensity_ref if ref_data is not None else None,
                outputfile=outputfile,
                SAEDProcessor=self,
                xray=False
            )
            # Si une méthode d'export existe, l'appeler ici
            if hasattr(pdf_interactive, 'save'):
                pdf_interactive.save(outputfile)
            pdf_interactive.show()
        else:
            print('Compute PDF with default parameters')
            r,G = compute_ePDF(
                q_sample,
                intensity_sample,
                composition,
                Iref=None,
                bgscale=1.0,
                qmin=1.5,
                qmax=None,
                qmaxinst=None,
                rmin=0.0,
                rmax=50.0,
                rstep=0.01,
                rpoly=1.4,
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
                 Iref=None,
                 rmin=0,
                 rmax=50,
                 rstep=0.01,
                 xray: bool = False,
                 outputfile: str = './pdf_results.csv',
                 SAEDProcessor=None):
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
        """
        print('Adjust sliders to optimize PDF parameters. Click "Save" to export results.')
        # Retrieve useful metadata from SAEDProcessor if provided
        if SAEDProcessor is not None:
            self.wavelength = SAEDProcessor.metadata.get('wavelength', None)
            self.camera = SAEDProcessor.metadata.get('camera_title', None)
            self.sample_diffraction_image = SAEDProcessor.dm4_file
            self.ref_diffraction_image = SAEDProcessor.ref_diffraction_image
            self.composition = composition
        else:
            self.wavelength = None
            self.camera = None
            self.sample_diffraction_image = None
            self.ref_diffraction_image = None
            self.composition = None
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
        self.bgscale_slider = widgets.FloatSlider(
            value=1, min=0, max=2, step=0.01, 
            description="bgscale", readout_format=".2f"
        )
        self.qmin_slider = widgets.FloatSlider(
            value=1.5, min=np.min(q), max=min(24,np.max(q)), step=0.01,
            description="qmin", readout_format=".2f"
        )
        self.qmax_slider = widgets.FloatSlider(
            value=min(24,np.max(q)), min=np.min(q), max=np.max(q), step=0.01,
            description="qmax", readout_format=".2f"
        )
        self.qmaxinst_slider = widgets.FloatSlider(
            value=min(24,np.max(q)), min=np.min(q), max=np.max(q), step=0.01,
            description="qmaxinst", readout_format=".2f"
        )
        self.rpoly_slider = widgets.FloatSlider(
            value=1.4, min=0.1, max=2.5, step=0.01,
            description="rpoly", readout_format=".2f"
        )
        
        self.lorch_checkbox = widgets.Checkbox(
            value=True,
            description="apply Lorch window correction to eliminate termination ripples",
            indent=False)

        # Save button for exporting results
        self.save_button = widgets.Button(description="💾 Save", button_style="success")
        self.save_button.on_click(lambda b: self.save_results(b, outputfile))

        # Organize widgets in vje veux quelque xhiose de plus simple. Je vais me débrouillerertical layout
        self.sliders = widgets.VBox([
            self.bgscale_slider,
            self.qmin_slider,
            self.qmax_slider,
            self.qmaxinst_slider,
            self.rpoly_slider,
            self.lorch_checkbox,
            self.save_button])

        # Output area for plots
        self.plot_output = widgets.Output()

        # Link sliders to update function for real-time feedback
        widgets.interactive_output(self.update_plot, {
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
        
        """
        # Collect metadata from sliders and configuration
        metadata = {
            'Composition': self.pdf_config.get('composition', 'N/A'),
            'bgscale': f"{self.bgscale_slider.value:.4f}",
            'qmin (Å⁻¹)': f"{self.qmin_slider.value:.4f}",
            'qmax (Å⁻¹)': f"{self.qmax_slider.value:.4f}",
            'qmaxinst (Å⁻¹)': f"{self.qmaxinst_slider.value:.4f}",
            'rpoly': f"{self.rpoly_slider.value:.4f}",
            'Lorch correction': str(self.lorch_checkbox.value),
            'rmin (Å)': f"{self.pdf_config.get('rmin', 0):.4f}",
            'rmax (Å)': f"{self.pdf_config.get('rmax', 50):.4f}",
            'rstep (Å)': f"{self.pdf_config.get('rstep', 0.01):.4f}",
        }
        
        # Write file with metadata header
        with open(fname, 'w') as f:
                       
            for key, value in metadata.items():
                f.write(f"{key:.<40} = {value}\n")
            
            f.write("\n" )
            f.write("#DATA: r (Å) vs G(r)\n")
            
            
            # Write data
            for r_val, g_val in zip(self.last_r, self.last_G):
                f.write(f"{r_val:15.6f} {g_val:20.8f}\n")
        
        print(f"Results saved in {fname}")"""

    def show(self):
        """
        Display the interactive interface.
        
        Creates a horizontal layout with sliders on the left and plots on the right.
        """
        ui = widgets.HBox([self.sliders, self.plot_output])
        display(ui)
        
        # Generate initial plot with default parameter values
        self.update_plot(
            self.bgscale_slider.value, self.qmin_slider.value,
            self.qmax_slider.value, self.qmaxinst_slider.value,
            self.rpoly_slider.value, self.lorch_checkbox.value
        )