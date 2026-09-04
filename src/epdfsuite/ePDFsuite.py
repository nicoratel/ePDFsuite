from .filereader import load_data
from .recalibration import recalibrate_from_isocurve
from .pdf_extraction import compute_ePDF
from pyFAI import load
import fabio
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import hyperspy.api as hs

def _mask_as_array(mask):
    """Return a boolean ndarray from a mask that may be a file path or already an ndarray."""
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        return mask.astype(bool)
    
    return fabio.open(mask).data.astype(bool)



class SAEDProcessor:
    def __init__(self,
                image_file,
                poni_file = None,
                mask=None,
                # deconvolution parameters
                mtf_file=None,
                wiener_epsilon=None,
                dqe_file=None,
                verbose=False,
                precession_angle=None,

                ):
        """
        Initialise a SAED data processor.

        Loads the image, identifies the camera type and wavelength from
        metadata, optionally applies MTF deconvolution (Wiener filter), and
        prepares the pyFAI azimuthal integrator if a PONI file is provided.

        Parameters
        ----------
        image_file : str
            Path to the SAED data file (DM4, DM3, tif, tiff).
        poni_file : str, optional
            Path to the pyFAI geometric calibration file (.poni).
            If ``None``, a pixel-scale integrator is used.
        mask : str, optional
            Path to a fabio mask file (EDF or similar). Convention:
            0 = valid pixel, 1 = masked pixel.
        mtf_file : str, optional
            Path to the MTF file used for Wiener deconvolution.
            If ``None``, no deconvolution is applied.
        wiener_epsilon : float, optional
            Regularisation parameter for the Wiener filter.
            If ``None``, read from column 3 of the MTF file.
        dqe_file : str, optional
            Path to the DQE file used for deconvolution.
            If ``None``, Wiener deconvolution if MTF is available, else no deconvolution
        verbose : bool, optional
            If ``True``, print metadata and detector info. Default is ``False``.
        precession_angle : float, optional
            The precession angle of the electron beam in degrees. If ``None``, the angle is not used.
        """
        self.dm4_file = image_file
        self.poni_file = poni_file
        metadata, img = load_data(image_file, verbose=verbose)
        self.metadata = metadata
        self.img = img

        # load mask if provided, otherwise create an empty mask
        if mask is not None:
            mask_img = fabio.open(mask)
            self.mask = mask_img.data
        else:
            self.mask = np.zeros_like(self.img)
        # load poni file if provided, and prepare pyFAI integrator
        if poni_file is not None:
            self.ai = load(poni_file)
            self.use_pyfai = True
            if mask is not None:
                mask_img = fabio.open(mask)
                self.mask = mask_img.data.astype(bool)
            else:
                self.mask = np.zeros(self.img.shape, dtype=bool)
        else:
            _hs_img = hs.load(image_file)
            self.use_pyfai = False
            self.scale = _hs_img.axes_manager[0].scale
            self.units = _hs_img.axes_manager[0].units
            print(f'scale = {self.scale}, unit = {self.units}')

        # load and apply MTF deconvolution (Wiener filter) if MTF file is provided
        if mtf_file is not None:
            self.ismtf = True
            if dqe_file is None:
                from .utilities import deconvolve_mtf_2d
                self.img = deconvolve_mtf_2d(self.img, mtf_file, wiener_epsilon=wiener_epsilon)
                self.isdqe = False
            else:
                from .utilities import deconvolve_mtf_dqe_2d
                self.img = deconvolve_mtf_dqe_2d(self.img, mtf_file, dqe_file)
                self.isdqe = True
        else:
            self.ismtf = False

        # Determine beam centre automatically via iso-intensity contour method.
        # Fall back to the intensity-maximum if isocurve detection fails.
        try:
            cx, cy = recalibrate_from_isocurve(
                self.img, mask=_mask_as_array(self.mask), plot=False
            )
            print(f'Centre estimate from iso-intensity contours: (x={cx:.2f}, y={cy:.2f})')
        except Exception as _e:
            _yx = np.unravel_index(np.argmax(self.img), self.img.shape)
            cy, cx = float(_yx[0]), float(_yx[1])
            print(
                f'Warning: iso-intensity centre detection failed ({_e}). '
                f'Falling back to intensity maximum: (x={cx:.1f}, y={cy:.1f}). '
                'Refine manually in the app.'
            )
        self.center = (cx, cy)
        self.precession_angle = precession_angle
        if precession_angle ==0:
            self.precession_angle=None

            


    def integrate(self, dm4_file=None, npt=2500, center=None, plot=False):
        """
        Azimuthally integrate the SAED pattern to a 1D I(q) profile.

        Beam centre recalibration is always performed with
        :func:`recalibrate_from_isocurve` using ``self.center`` as the
        initial estimate.  If *center* is provided it overwrites
        ``self.center`` and is used directly without re-running isocurve.

        Parameters
        ----------
        dm4_file : str, optional
            Path to an alternative DM4 file to integrate. If ``None``,
            uses ``self.dm4_file``.
        npt : int, optional
            Number of points in the output q profile. Default is 2500.
        center : tuple of float, optional
            Beam centre ``(x, y)`` in pixels.  If provided, overwrites
            ``self.center`` before integrating.  If ``None``, uses
            ``self.center`` as computed at initialisation.
        plot : bool, optional
            If ``True``, display the integrated I(q) pattern. Default is ``False``.

        Returns
        -------
        q : ndarray
            Scattering vector in Å⁻¹.
        I : ndarray
            Azimuthally averaged intensity.
        """
        if center is not None:
            # User-supplied centre: store and use directly
            self.center = center
        cx, cy = float(self.center[0]), float(self.center[1])

        if self.use_pyfai:
            fit2d = self.ai.getFit2D()
            fit2d['centerX'] = cx
            fit2d['centerY'] = cy
            self.ai.setFit2D(**fit2d)
            q, I = self.ai.integrate1d(
                self.img, npt, mask=self.mask, unit="q_A^-1", polarization_factor=0.99
            )
        else:
            mask_bool = np.asarray(self.mask, dtype=bool)
            valid = ~mask_bool.ravel()
            y_idx, x_idx = np.indices(self.img.shape)
            r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
            r_flat = r.ravel()
            img_flat = self.img.ravel()

            # Sub-pixel (linear-interpolation) binning: each pixel's weight is
            # split between its two neighbouring bins proportionally to the
            # fractional distance to each bin centre, mimicking the CSR/LUT
            # approach used by pyFAI and eliminating bin-boundary artefacts.
            r_max = r_flat[valid].max()
            bin_width = r_max / npt
            # Continuous bin index for every valid pixel
            r_norm = r_flat[valid] / bin_width          # in [0, npt]
            k = r_norm.astype(int)                      # lower bin
            frac = r_norm - k                           # fractional part in [0, 1)
            k  = np.clip(k,     0, npt - 1)
            k1 = np.clip(k + 1, 0, npt - 1)
            img_v = img_flat[valid]
            weights_sum = np.zeros(npt)
            counts      = np.zeros(npt)
            np.add.at(weights_sum, k,  (1.0 - frac) * img_v)
            np.add.at(weights_sum, k1, frac          * img_v)
            np.add.at(counts,      k,  (1.0 - frac))
            np.add.at(counts,      k1, frac)
            with np.errstate(invalid='ignore', divide='ignore'):
                I = np.where(counts > 0, weights_sum / counts, 0.0)
            # Bin centres in pixel units (used for q conversion below)
            edges = np.linspace(0.0, r_max, npt + 1)   # kept for r_centers

            # Bin centres in pixels → convert to q using the hyperspy scale
            r_centers = 0.5 * (edges[:-1] + edges[1:])
            if self.units == '1/nm':
                q = r_centers * self.scale * 2 * np.pi / 10
                # Solid angle correction: derive 2θ from q and wavelength
                # q = 4π sin(θ)/λ  →  sin(θ) = qλ/(4π)
                wavelength_A = self.metadata.get('wavelength', None)
                if wavelength_A is not None:
                    sin_theta = np.clip(q * wavelength_A / (4 * np.pi), -1.0, 1.0)
                    two_theta = 2 * np.arcsin(sin_theta)
                    cos3 = np.where(np.cos(two_theta) > 0, np.cos(two_theta) ** 3, 1.0)
                    I = I / cos3
            elif self.units == 'mrad':
                # theta (Bragg half-angle) in radians; detector angle = 2θ
                theta = (r_centers * self.scale * 1e-3) / 2
                q = 4 * np.pi * np.sin(theta) / self.metadata['wavelength']
                
                # Solid angle correction using detector angle 2θ
                two_theta = 2 * theta
                cos3 = np.where(np.cos(two_theta) > 0, np.cos(two_theta) ** 3, 1.0)
                I = I / cos3
            else: # assume units are already in s (1/Å)
                q = r_centers * self.scale * 2 * np.pi
        

        # perform intensity corrections if precession is used
        if self.precession_angle is not None:
            # Convert precession angle from degrees to radians
            alpha_rad = np.radians(self.precession_angle)
            R0 = 2 * np.pi * np.sin(alpha_rad) / self.metadata['wavelength'] 
            # Apply the correction factor: I_corrected = I / q*racine(1-q/2R0)) where R0=2pi*prec_angle/lambda
            I = I / (q * np.sqrt(1 - q / (2 * R0)))        
            print("Intensity corrections induced by precession have been performed.\n")
            # limit on qmax imposed by precession angle:
            qmax_prec = np.sin(alpha_rad)*4*np.pi / self.metadata['wavelength']
            print(f'Intensity corrections impose qmax = {qmax_prec:.2f} $\AA^{-1}$')
            print('See K. Gjonnes Ultramicroscopy 69 1-11 (1997) for more details')
        
        if plot:
            plt.figure()
            plt.semilogy(q, I)
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Azimuthally Integrated SAED Pattern')
            plt.grid()
            plt.show()
        return q, I

    def plot(self,vmin=-4, vmax=0,cmap='jet',display_mask=False):
        plt.figure()
        if display_mask:
            
            # Create a copy of the image for display
            img_display = self.img.copy() / np.max(self.img)
            # Set masked pixels to NaN to display them in white
            img_display[self.mask.astype(bool)] = np.nan
            plt.imshow(img_display, cmap=cmap, norm=LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
            # Set NaN color to white
            current_cmap = plt.get_cmap(cmap).copy()
            current_cmap.set_bad(color='white')
            plt.imshow(img_display, cmap=current_cmap, norm=LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
            plt.plot(self.center[0], self.center[1], 'k+', markersize=8)
        else:            
            plt.imshow(self.img/np.max(self.img), cmap=cmap, norm=LogNorm(vmin=10**(vmin), vmax=10**(vmax)))
        #plot center as wihte cross
            plt.plot(self.center[0], self.center[1], 'w+', markersize=8)

    
    def plot_recalibrated_image(self, **kwargs):
        """
        Display the diffraction image with the detected beam centre.

        Runs :func:`recalibrate_from_isocurve` with ``plot=True``
        to trigger the diagnostic figure, using ``self.center`` as
        the initial estimate.  The result is not stored.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments forwarded to
            :func:`recalibrate_from_isocurve` (e.g. ``n_levels``,
            ``level_range``, ``rms_rel_max``, ``min_arc_deg``,
            ``cluster_window``).
        """
        c_x,c_y = recalibrate_from_isocurve(
            self.img, mask=_mask_as_array(self.mask), plot=True,
            initial_center=self.center, **kwargs
        )
        self.center = (c_x, c_y)

    def inspect_histogram(self, bins=256, log_scale=True, exclude_zero=False,
                          saturation_threshold=0.98, percentile_clip=99.9999):
        """
        Plot the grey-level histogram of the image to assess camera linearity.

        Always displays two side-by-side subplots:

        * **Left** — full image (no mask), full x-axis range, so that
          saturated pixels from e.g. the direct beam are visible.
        * **Right** — valid pixels only (mask applied, if a mask was
          provided), with the x-axis clipped to ``percentile_clip`` to
          reveal the bulk distribution.

        A smooth, monotonically decreasing histogram with no spike at the
        maximum indicates a linear camera response.

        Parameters
        ----------
        bins : int, optional
            Number of histogram bins. Default is 256.
        log_scale : bool, optional
            If ``True`` (default), both y-axes are in log scale.
        exclude_zero : bool, optional
            If ``True``, zero-valued pixels are excluded from both
            histograms. Default is ``False``.
        saturation_threshold : float, optional
            Fraction of the global maximum above which pixels are flagged
            as potentially saturated. A vertical red dashed line is drawn
            at this value on both subplots. Default is 0.98.
        percentile_clip : float, optional
            Percentile used to clip the x-axis of the **masked** (right)
            subplot only. Useful when a few very bright pixels compress the
            distribution towards zero. Default is 99.9999. Set to 100 to
            disable clipping.

        Returns
        -------
        counts_raw : ndarray
            Pixel counts per bin for the unmasked histogram.
        edges_raw : ndarray
            Bin edges for the unmasked histogram.
        counts_masked : ndarray or None
            Pixel counts per bin for the masked histogram, or ``None`` if
            no mask is defined.
        edges_masked : ndarray or None
            Bin edges for the masked histogram, or ``None`` if no mask.
        """
        # Saturation threshold based on the full-image maximum
        all_data = self.img.ravel().astype(float)
        sat_value = saturation_threshold * all_data.max()

        # --- unmasked data (left subplot) ---
        data_raw = all_data.copy()
        if exclude_zero:
            data_raw = data_raw[data_raw > 0]

        # --- masked data (right subplot) ---
        has_mask = self.mask is not None and np.any(self.mask != 0)
        if has_mask:
            data_masked = self.img[self.mask == 0].ravel().astype(float)
            if exclude_zero:
                data_masked = data_masked[data_masked > 0]
        else:
            data_masked = data_raw  # fallback: same data

        counts_raw, edges_raw = np.histogram(data_raw, bins=bins)
        centers_raw = 0.5 * (edges_raw[:-1] + edges_raw[1:])

        counts_masked, edges_masked = np.histogram(data_masked, bins=bins)
        centers_masked = 0.5 * (edges_masked[:-1] + edges_masked[1:])

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        def _draw(ax, centers, counts, edges, title, clip_xaxis):
            ax.bar(centers, counts, width=np.diff(edges), align='center',
                   color='steelblue', edgecolor='none', alpha=0.8)
            ax.axvline(sat_value, color='red', linestyle='--', linewidth=1.2,
                       label=f'Sat. threshold ({saturation_threshold*100:.0f}% = {sat_value:.0f} cts)')
            if clip_xaxis and percentile_clip < 100:
                xmax = np.percentile(data_masked, percentile_clip)
                ax.set_xlim(left=0, right=xmax)
            if log_scale:
                ax.set_yscale('log')
            ax.set_xlabel('Grey level (counts)')
            ax.set_ylabel('Number of pixels')
            ax.set_title(title)
            ax.legend(fontsize=8)

        _draw(axes[0], centers_raw, counts_raw, edges_raw,
              title='Raw image (no mask) — full range',
              clip_xaxis=False)

        masked_title = (f'Masked image — x clipped at {percentile_clip}th percentile'
                        if has_mask else 'No mask provided — same as raw')
        _draw(axes[1], centers_masked, counts_masked, edges_masked,
              title=masked_title,
              clip_xaxis=True)

        plt.suptitle('Image histogram — camera linearity check', fontsize=12, y=1.01)
        plt.tight_layout()
        plt.show()

        # --- warnings (on masked data if available, raw otherwise) ---
        data_check = data_masked if has_mask else data_raw
        n_sat = int(np.sum(data_check >= sat_value))
        label = 'masked image' if has_mask else 'image'
        if n_sat > 0:
            print(f"Warning: {n_sat} pixel(s) in {label} above the saturation threshold "
                  f"({saturation_threshold*100:.0f}% of max = {sat_value:.0f} counts). "
                  "Camera linearity may be compromised.")
        else:
            print(f"No saturated pixels detected in {label} — camera response appears linear.")

        return 


    def extract_epdf(self,
                     ref_diffraction_image=None,
                     ref_poni_file=None,
                     composition='Au',
                     rmin=0.1,
                     rmax=50.0,
                     rstep=0.01,
                     outputfile=None,
                     interactive=True,
                     plot=False,
                     mtf_file=None,
                     bgscale=1,
                     qmin=1.5,
                     qmax=24,
                     qmaxinst=24,
                     rpoly=1.4
                     ):
        """
        Extract the ePDF from the SAED data (convenience wrapper).

        Creates a :class:`SAEDProcessor` for the reference image if provided
        (which automatically computes its own beam centre via isocurve at
        initialisation), then delegates to the standalone :func:`extract_epdf`
        function.

        Parameters
        ----------
        ref_diffraction_image : str, optional
            Path to the reference (background) diffraction image.
        ref_poni_file : str, optional
            Path to a PONI file for the reference image if it differs from
            the sample.
        composition : str, optional
            Chemical formula of the sample (e.g. ``'Au'``, ``'SiO2'``).
            Default is ``'Au'``.
        rmin, rmax, rstep : float, optional
            Real-space range and step for G(r) in Å.
        outputfile : str, optional
            Path for the output ``.gr`` file. Auto-generated if ``None``.
        interactive : bool, optional
            If ``True`` (default), open the interactive slider GUI.
        plot : bool, optional
            If ``True``, display plots in non-interactive mode.
        mtf_file : str, optional
            Not used in this wrapper (MTF applied at initialisation).
        bgscale : float, optional
            Background scaling factor. Default is 1.
        qmin, qmax, qmaxinst : float, optional
            Q-range limits in Å⁻¹ for PDF computation.
        rpoly : float, optional
            Polynomial background degree control (PDFgetX3 convention).
        correct_multiple_scattering : bool, optional
            If ``True``, apply the elastic multiple-scattering correction
            (see :func:`~epdfsuite.multiple_scattering.correct_multiple_scattering`)
            to the sample intensity before PDF extraction, using ``D``,
            ``HV`` and ``density``. Default is ``False``.
        D : float, optional
            Nanoparticle diameter (nm), used to estimate the thickness
            traversed by the beam (``t = 2/3 * D``). Default is 15.
        HV : float, optional
            Accelerating voltage (kV). Default is 200.
        density : float, optional
            Mass density of the sample (g/cm³), used to estimate the
            elastic mean free path. Default is 19.3 (Au).

        Returns
        -------
        results : PDFResultsReference or tuple
            Interactive mode: :class:`PDFResultsReference` supporting
            ``r, g = results`` unpacking.
            Non-interactive mode: ``(r, G)`` tuple of ndarrays.
        """
        ref_processor = None
        if ref_diffraction_image is not None:
            ref_processor = SAEDProcessor(
                ref_diffraction_image,
                poni_file=ref_poni_file if ref_poni_file is not None else self.poni_file,
                verbose=False,
            )

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
                 rpoly=1.4
                 ):
    """
    Extract the electron Pair Distribution Function (ePDF) from SAED data.

    Integrates each :class:`SAEDProcessor` to a 1D I(q) profile, then calls
    :func:`~epdfsuite.pdf_extraction.compute_ePDF`. In interactive mode an
    ipywidgets GUI is shown; in non-interactive mode G(r) is computed once
    and saved to a ``.gr`` file.

    Parameters
    ----------
    sample_processor : SAEDProcessor
        Processor loaded with the sample diffraction image.
        Set ``sample_processor.initial_center`` before calling if needed.
    ref_processor : SAEDProcessor, optional
        Processor loaded with the background/reference image.
        If ``None``, no background subtraction is performed.
    composition : str, optional
        Chemical formula of the sample (e.g. ``'Au'``, ``'Fe2O3'``).
        Default is ``'Au'``.
    rmin, rmax, rstep : float, optional
        Real-space range and step for G(r) in Å.
        Defaults: 0.1, 50.0, 0.01.
    outputfile : str, optional
        Path for the output ``.gr`` file.
        Auto-generated from the sample filename if ``None``.
    interactive : bool, optional
        If ``True`` (default), open the interactive slider GUI via
        :class:`PDFInteractive`.
    plot : bool, optional
        If ``True``, display G(r) and F(Q) in non-interactive mode.
    bgscale : float, optional
        Background scaling factor applied to the reference. Default is 1.
    qmin, qmax, qmaxinst : float, optional
        Q-range limits in Å⁻¹ used for the Fourier transform and polynomial
        background fitting.
    rpoly : float, optional
        Polynomial degree control (PDFgetX3 convention). Default is 1.4.
    

    Returns
    -------
    results : PDFResultsReference or tuple
        Interactive mode: :class:`PDFResultsReference` — supports
        ``r, g = results`` unpacking after slider adjustment.
        Non-interactive mode: ``(r, G)`` tuple of ndarrays.

    Examples
    --------
    >>> sample = SAEDProcessor('sample.dm4', poni_file='calib.poni')
    >>> sample.initial_center = (335, 275)
    >>> ref = SAEDProcessor('ref.dm4', poni_file='calib.poni')
    >>> results = extract_epdf(sample, ref, composition='Au', interactive=False)
    >>> r, G = results
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
        # Création de l'objet PDFInteractive avec la nouvelle interface
        pdf_interactive = PDFInteractive(
            sample_processor,
            ref_processor=ref_processor,
            composition=composition,
            rmin=rmin,
            rmax=rmax,
            rstep=rstep,
            xray=False,
            outputfile=outputfile
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
    Proxy object providing access to the most recent PDF results from interactive mode.

    Wraps a :class:`PDFInteractive` instance and supports tuple unpacking
    (``r, g = reference``) so that the same syntax works whether the
    extraction is interactive or not. Values reflect the **last slider
    state** — call ``r, g = results`` after adjusting the sliders.
    """
    
    def __init__(self, pdf_interactive):
        """
        Parameters
        ----------
        pdf_interactive : PDFInteractive
            The interactive GUI object holding the computed results.
        """
        self._pdf_interactive = pdf_interactive
    
    def __iter__(self):
        """
        Support tuple unpacking: ``r, g = reference``.

        Returns the latest r and G arrays computed by the interactive GUI.
        Prints a warning if no computation has been performed yet.
        """
        if self._pdf_interactive.last_r is None or self._pdf_interactive.last_G is None:
            print("⚠️ Aucune valeur disponible. Ajustez les paramètres avec les sliders pour générer r et G.")
            return iter([None, None])
        return iter([self._pdf_interactive.last_r, self._pdf_interactive.last_G])
    
    def __repr__(self):
        """String representation showing r-range and number of points."""
        if self._pdf_interactive.last_r is None:
            return "PDFResultsReference(no data yet - adjust sliders to compute)"
        return f"PDFResultsReference(r: {len(self._pdf_interactive.last_r)} points, " \
               f"r_range=[{self._pdf_interactive.last_r.min():.2f}, {self._pdf_interactive.last_r.max():.2f}] Å)"
    
    @property
    def r(self):
        """ndarray : Real-space distance axis in Å from the last computation."""
        return self._pdf_interactive.last_r
    
    @property
    def g(self):
        """ndarray : G(r) values in Å⁻² from the last computation."""
        return self._pdf_interactive.last_G


# ------------------
# Interactive GUI Class
# ------------------
class PDFInteractive:
    """
    Jupyter widget GUI for interactive ePDF parameter optimisation.

    Displays ipywidgets sliders for ``bgscale``, ``qmin``, ``qmax``,
    ``qmaxinst``, and ``rpoly``, recomputing G(r) in real time.
    Results can be exported to a ``.gr`` file via the Save button.

    Intended to be created by :func:`extract_epdf` — not directly by users.
    """

    def __init__(self,
                 sample_processor,
                 ref_processor=None,
                 composition='Au',
                 rmin=0,
                 rmax=50,
                 rstep=0.01,
                 xray: bool = False,
                 outputfile: str = './pdf_results.csv'):
        """
        Parameters
        ----------
        sample_processor : SAEDProcessor
            Processor for the sample diffraction data.
        ref_processor : SAEDProcessor, optional
            Processor for the background/reference image. Default is ``None``.
        composition : str, optional
            Chemical formula of the sample. Default is ``'Au'``.
        rmin, rmax, rstep : float, optional
            Real-space range and step for G(r) in Å.
        xray : bool, optional
            If ``True``, use X-ray scattering factors. Default is ``False``.
        outputfile : str, optional
            Default path for the Save button output. Default is
            ``'./pdf_results.csv'``.
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.widgets = widgets
        self.display = display

        print('Slide cursors to ajdust parameters values. Click "Save" to export results.')

        # Stocker les processeurs pour accès ultérieur
        self.sample_processor = sample_processor
        self.ref_processor = ref_processor
        self.composition = composition

        # Intégration des données (sample et ref)
        q, Iexp = sample_processor.integrate(plot=False)
        if ref_processor is not None:
            _, Iref = ref_processor.integrate(plot=False)
        else:
            Iref = None

        # Métadonnées utiles
        self.wavelength = getattr(sample_processor, 'wavelength', None)
        self.camera = getattr(sample_processor, 'camera_title', None)
        self.sample_diffraction_image = getattr(sample_processor, 'dm4_file', None)
        self.ref_diffraction_image = getattr(ref_processor, 'dm4_file', None) if ref_processor is not None else None

        # PDF config
        self.xray = xray
        self.pdf_config = dict(
            q=q, Iexp=Iexp, Iref=Iref, composition=composition,
            rmin=rmin, rmax=rmax, rstep=rstep,
        )

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
        Recompute G(r) and refresh the output plot.

        Called automatically by ``ipywidgets.interactive_output`` whenever
        a slider value changes. Stores the result in ``self.last_r`` and
        ``self.last_G``.

        Parameters
        ----------
        bgscale : float
            Background scaling factor.
        qmin, qmax, qmaxinst : float
            Q-range limits in Å⁻¹.
        rpoly : float
            Polynomial degree control parameter.
        lorch : bool
            Whether to apply the Lorch modification function.
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
        Save the last computed G(r) to a ``.gr`` text file.

        The file format is compatible with PDFgetX3 / PDFBatchAnalysis,
        with a structured header containing all computation parameters.

        Parameters
        ----------
        b : widget button event
            Unused; required by the ipywidgets callback signature.
        outputfile : str, optional
            Output file path. Default is ``'./pdf_results.gr'``.
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
        header +=f'wavelength = {self.sample_processor.metadata.get("wavelength", "unknown"):.4f}\n'
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
        print(f'PDF saved to {outputfile}')
        

    def show(self):
        """
        Render and display the interactive GUI in a Jupyter notebook.

        Computes G(r) with the current slider values before displaying
        the interface, so that ``last_r`` and ``last_G`` are immediately
        available for tuple unpacking.
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
        Extract ePDF by averaging over multiple SAED image files.

        Integrates each file independently, interpolates all profiles onto
        the q-grid of the first file, computes an average I(q), and calls
        :func:`~epdfsuite.pdf_extraction.compute_ePDF`.

        Parameters
        ----------
        dm4_files : list of str
            Paths to the SAED image files (DM4, DM3, tif, tiff).
        ref_diffraction_image : str, optional
            Path to the background/reference diffraction image.
        ref_poni_file : str, optional
            Path to a PONI file for the reference if different from the sample.
        composition : str, optional
            Chemical formula of the sample. Default is ``'Au'``.
        rmin, rmax, rstep : float, optional
            Real-space range and step for G(r) in Å.
        qmin, qmax, qmaxinst : float, optional
            Q-range limits in Å⁻¹ for PDF computation.
        bgscale : float, optional
            Background scaling factor. Default is 1.0.
        rpoly : float, optional
            Polynomial degree control (PDFgetX3 convention). Default is 1.4.
        outputfile : str, optional
            Output ``.gr`` file path. Auto-generated if ``None``.
        interactive : bool, optional
            If ``True`` (default), open the interactive slider GUI.
        poni_file : str, optional
            Path to the pyFAI PONI calibration file.
        beamstop : bool, optional
            Reserved for future use. Default is ``False``.
        plot : bool, optional
            If ``True``, display plots in non-interactive mode.
        verbose : bool, optional
            If ``True``, print processing details. Default is ``False``.
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
       
