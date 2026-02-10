import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import tempfile
import os
from io import BytesIO

# Import the necessary modules from your package
from ePDFsuite import SAEDProcessor, extract_ePDF_from_mutliple_files
from recalibration import recalibrate_with_beamstop_noponi
from filereader import load_data
from pdf_extraction import compute_ePDF
from calibration import perform_geometric_calibration
import hyperspy.api as hs

# Configure Streamlit page
st.set_page_config(
    page_title="ePDFsuite - Interactive GUI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 ePDFsuite - Interactive PDF Analysis")

# Add CSS to style tab labels and reduce content font size
st.markdown("""
    <style>
        button[data-baseweb="tab"] {
            font-size: 16px !important;
            padding: 12px 24px !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 16px;
        }
        /* Reduce font size in tab content */
        .stTabs [role="tabpanel"] {
            font-size: 13px;
        }
        /* Reduce markdown and other text */
        [role="tabpanel"] p {
            font-size: 13px !important;
        }
        /* Reduce heading sizes */
        [role="tabpanel"] h2 {
            font-size: 18px !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        [role="tabpanel"] h3 {
            font-size: 15px !important;
            margin-top: 0.8rem !important;
            margin-bottom: 0.4rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Add stop button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🛑 Stop App", type="secondary"):
    st.info("✋ Stopping application...")
    import sys
    sys.exit(0)

# Create three tabs
tab1, tab2, tab3 = st.tabs(["📊 Geometric Calibration", "📸 Plot Data", "📈 PDF Extraction"])

# ============================================================================
# TAB 1: GEOMETRIC CALIBRATION
# ============================================================================
with tab1:
    st.markdown("# 📊 Geometric Calibration")
    st.markdown("**Perform geometric calibration using pyFAI-calib2. Upload a calibrant diffraction image and the corresponding CIF file.**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 📁 Input Files")
        
        # Upload calibrant diffraction image
        calibrant_image = st.file_uploader(
            "📷 Calibrant diffraction image (DM4, DM3, TIF, TIFF)",
            type=["dm4", "dm3", "tif", "tiff"],
            key="calibrant_image"
        )
        
        # Upload CIF file
        cif_file = st.file_uploader(
            "📄 Calibrant CIF file",
            type=["cif"],
            key="cif_file"
        )
    
    # Display calibrant image if uploaded
    if calibrant_image is not None:
        st.subheader("Image Preview")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp_file:
            tmp_file.write(calibrant_image.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Load and display image
            metadata, img = load_data(tmp_path, verbose=False)
            
            # Create figure for display
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            im = ax.imshow(img / np.max(img), cmap='gray', 
                          norm=LogNorm(vmin=1e-4, vmax=1))
            ax.set_title("Calibrant Diffraction Image")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            plt.colorbar(im, ax=ax, label="Intensity (normalized)")
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Display metadata
            with st.expander("📋 Metadata"):
                st.json(metadata)
        
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    # Calibration button
    if st.button("🔧 Perform Calibration", type="primary"):
        if calibrant_image is None or cif_file is None:
            st.error("❌ Please upload both the image and CIF file")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp_img:
                tmp_img.write(calibrant_image.getbuffer())
                tmp_img_path = tmp_img.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp_cif:
                tmp_cif.write(cif_file.getbuffer())
                tmp_cif_path = tmp_cif.name
            
            try:
                st.info("⏳ Calibration in progress...")
                st.info("📌 A pyFAI-calib2 window will open for interactive adjustment of center and distance.")
                
                # Call the calibration function
                perform_geometric_calibration(
                    cif_file=tmp_cif_path,
                    image_file=tmp_img_path
                )
                
                st.success("✅ Calibration completed! A PONI file has been generated.")
                st.info("📁 The PONI file will be available in the current directory for the 'PDF Extraction' tab")
            
            except Exception as e:
                st.error(f"❌ Error during calibration: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            finally:
                # Clean up temporary files
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)
                if os.path.exists(tmp_cif_path):
                    os.remove(tmp_cif_path)

# ============================================================================
# TAB 2: PLOT DATA
# ============================================================================
with tab2:
    st.markdown("# 📸 Sample Data Visualization")
    st.markdown("**Visualize your sample diffraction image with the recalibrated beam center.**")
    
    st.markdown("### 📁 Input Files")
    
    st.markdown("**Sample Image**")
    sample_image_plot = st.file_uploader(
        "Select sample image",
        type=["dm4", "dm3", "tif", "tiff"],
        key="sample_image_plot"
    )
    
    if sample_image_plot is not None:
        st.subheader("👁️ Sample Image with Recalibrated Center")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp_file:
            tmp_file.write(sample_image_plot.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            metadata, img = load_data(tmp_path, verbose=False)
            
            # Recalibrate center
            center_x, center_y = recalibrate_with_beamstop_noponi(
                img, threshold_rel=0.5, min_size=50, plot=False
            )
            
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(img / np.max(img), cmap='gray',
                          norm=LogNorm(vmin=1e-4, vmax=1))
            # Plot center
            ax.plot(center_x, center_y, 'r+', markersize=12, markeredgewidth=1.5,
                   label=f'Center: ({center_x:.1f}, {center_y:.1f})')
            ax.set_title("Sample Image with Recalibrated Center")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.legend(fontsize=12)
            plt.colorbar(im, ax=ax, label="Intensity (normalized)")
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Display metadata
            with st.expander("📋 Image Metadata"):
                st.json(metadata)
        
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        st.info("📤 Please upload a sample image to visualize")

# ============================================================================
# TAB 3: PDF EXTRACTION
# ============================================================================
with tab3:
    st.markdown("# 📈 PDF Extraction")
    st.markdown("**Calculate the Pair Distribution Function (PDF) from your sample and reference images. Adjust parameters with interactive sliders.**")
    
    # ========== FILE UPLOADS SECTION ==========
    st.markdown("## 📁 Input Files")
    
    col_files1, col_files2 = st.columns(2)
    
    with col_files1:
        st.markdown("**Sample Images** (multiple files allowed)")
        sample_images = st.file_uploader(
            "DM4, DM3, TIF, TIFF",
            type=["dm4", "dm3", "tif", "tiff"],
            accept_multiple_files=True,
            key="sample_images",
            label_visibility="collapsed"
        )
        
        st.markdown("**PONI File** (optional)")
        poni_file = st.file_uploader(
            "Geometric calibration",
            type=["poni"],
            key="poni_file",
            label_visibility="collapsed"
        )
    
    with col_files2:
        st.markdown("**Reference Image** (optional)")
        ref_image = st.file_uploader(
            "DM4, DM3, TIF, TIFF",
            type=["dm4", "dm3", "tif", "tiff"],
            key="ref_image",
            label_visibility="collapsed"
        )
    
    # ========== BEAMSTOP OPTION ==========
    st.markdown("### 🔍 Image Processing")
    beamstop = st.checkbox("Beamstop present on sample images", value=False)
    
    # ========== OUTPUT PARAMETERS SECTION ==========
    st.markdown("## ⚙️ Output Parameters")
    
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.markdown("**R-space Range**")
        rmin = st.number_input("rmin (Å)", value=0.0, step=0.1)
        rmax = st.number_input("rmax (Å)", value=50.0, step=0.1)
    
    with col_out2:
        st.markdown("**Output File**")
        rstep = st.number_input("rstep (Å)", value=0.01, step=0.001)
        samplename = st.text_input("Sample name (optional)", value="", placeholder="Leave empty to use default filename")
        
        # Generate output_file based on samplename
        if samplename:
            output_file = f"{samplename}.gr"
        else:
            output_file = "ePDF_results.gr"
    
    # ========== PROCESSING SECTION ==========
    st.markdown("## 📊 PDF Calculation")
    
    # Default values for interactive sliders
    _default_bgscale = 1.0
    _default_qmin = 1.5
    _default_qmax = 24.0
    _default_qmaxinst = 24.0  # Max limit for qmax slider
    _default_rpoly = 1.4
    _default_lorch = True
    _default_composition = "Au"
    
    # Use defaults for processing (sliders can be adjusted after calculation)
    # Note: beamstop is already defined from the checkbox above
    composition = _default_composition
    
    # Processing button
    if st.button("🚀 Calculate PDF", type="primary"):
        if not sample_images:
            st.error("❌ Please upload at least one sample image")
        elif poni_file is None:
            st.warning("⚠️ No PONI file provided - using automatic recalibration")
        
        # Save uploaded files temporarily
        temp_files = []
        try:
            for idx, sample_img in enumerate(sample_images):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp:
                    tmp.write(sample_img.getbuffer())
                    temp_files.append(tmp.name)
            
            ref_path = None
            if ref_image:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp:
                    tmp.write(ref_image.getbuffer())
                    ref_path = tmp.name
                    temp_files.append(tmp.name)
            
            poni_path = None
            if poni_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".poni") as tmp:
                    tmp.write(poni_file.getbuffer())
                    poni_path = tmp.name
                    temp_files.append(tmp.name)
            
            # Progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⏳ Integrating images...")
            progress_bar.progress(25)
            
            # Process with extract_ePDF_from_multiple_files
            try:
                # Create temporary output file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".gr") as tmp_out:
                    output_path = tmp_out.name
                    temp_files.append(output_path)
                
                # First, integrate the sample images manually to get raw q and I
                status_text.text("⏳ Integrating sample images...")
                I_array = []
                q_array = []
                for dm4_file_path in temp_files[:len(sample_images)]:
                    proc_temp = SAEDProcessor(dm4_file_path, poni_file=poni_path, beamstop=beamstop, verbose=False)
                    q_temp, I_temp = proc_temp.integrate(dm4_file_path, plot=False)
                    q_array.append(q_temp)
                    I_array.append(I_temp)
                
                # Use the q range from the first file as reference
                q_raw = q_array[0]
                
                # Interpolate all I arrays to the same q grid
                from scipy.interpolate import interp1d
                I_interpolated = []
                for i, I in enumerate(I_array):
                    if len(I) != len(q_raw):
                        f = interp1d(q_array[i], I, kind='linear', bounds_error=False, fill_value='extrapolate')
                        I_interp = f(q_raw)
                    else:
                        I_interp = I
                    I_interpolated.append(I_interp)
                
                I_raw = np.mean(I_interpolated, axis=0)  # Average of raw intensities
                
                # Store raw integration results in session state
                st.session_state.q_data = q_raw
                st.session_state.I_data = I_raw
                st.session_state.composition = composition
                st.session_state.rmin = rmin
                st.session_state.rmax = rmax
                st.session_state.rstep = rstep
                
                # Try to integrate reference image if available
                if ref_path:
                    try:
                        # Integrate reference image
                        if poni_path:
                            from pyFAI import load
                            ai = load(poni_path)
                            _, ref_img = load_data(ref_path, verbose=False)
                            q_ref, I_ref = ai.integrate1d(ref_img, npt=2500, unit="q_A^-1")
                            # Interpolate to sample q grid
                            st.session_state.I_ref = np.interp(st.session_state.q_data, q_ref, I_ref)
                        else:
                            # Use custom integration
                            proc_temp = SAEDProcessor(ref_path, poni_file=None, beamstop=beamstop)
                            q_ref, I_ref = proc_temp.integrate(ref_path, plot=False)
                            st.session_state.I_ref = np.interp(st.session_state.q_data, q_ref, I_ref)
                    except Exception as e:
                        st.warning(f"Could not integrate reference image: {e}")
                        st.session_state.I_ref = None
                else:
                    st.session_state.I_ref = None
                
                progress_bar.progress(90)
                st.session_state.data_ready = True
            
            except Exception as e:
                st.error(f"❌ Error during PDF calculation: {e}")
                import traceback
                st.error(traceback.format_exc())
        
        finally:
            # Clean up temporary files
            for tmp_file in temp_files:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
    
    # Display interactive controls if data is ready
    if hasattr(st.session_state, 'data_ready') and st.session_state.data_ready:
        st.subheader("⚙️ Interactive Parameters")
        
        st.markdown("**Adjust these parameters to refine the PDF calculation:**")
        
        # Create 5 columns for sliders to minimize vertical scrolling
        st.markdown("### 🎚️ Interactive Adjustment")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        q_max_data = float(np.max(st.session_state.q_data))
        
        with col1:
            bgscale_int = st.slider("bgscale", 0.0, 2.0, _default_bgscale, 0.01, key="bgscale_slider")
        
        with col2:
            qmin_int = st.slider("qmin (Å⁻¹)", float(np.min(st.session_state.q_data)), q_max_data, _default_qmin, 0.1, key="qmin_slider")
        
        with col3:
            qmax_int = st.slider("qmax (Å⁻¹)", float(np.min(st.session_state.q_data)), q_max_data, _default_qmax, 0.1, key="qmax_slider")
        
        with col4:
            rpoly_int = st.slider("rpoly", 0.1, 10.0, _default_rpoly, 0.1, key="rpoly_slider")
        
        with col5:
            qmaxinst_int = st.slider("qmaxinst (Å⁻¹)", float(np.min(st.session_state.q_data)), q_max_data, _default_qmaxinst, 0.1, key="qmaxinst_slider")
        
        lorch_int = st.checkbox("Lorch window correction", value=_default_lorch, key="lorch_checkbox")
        
        # Call compute_ePDF with plot=False to get data only
        r_pdf, G_pdf = compute_ePDF(
            q=st.session_state.q_data,
            Iexp=st.session_state.I_data,
            composition=st.session_state.composition,
            Iref=st.session_state.I_ref,
            bgscale=bgscale_int,
            qmin=qmin_int,
            qmax=qmax_int,
            qmaxinst=qmaxinst_int,
            rmin=st.session_state.rmin,
            rmax=st.session_state.rmax,
            rstep=st.session_state.rstep,
            rpoly=rpoly_int,
            Lorch=lorch_int,
            plot=False
        )
        
        # Create CSV content for download before displaying plots
        output_data = np.column_stack((r_pdf, G_pdf))
        import io
        csv_buffer = io.StringIO()
        
        # Create header compatible with PDFGetX3/ePDFsuite format
        header = '[DEFAULT]\n\n'
        header += 'version = ePDFsuite 1.0\n\n'
        header += '#input and output specifications\n'
        header += 'dataformat = q_A\n'
        header += f'outputtype = gr\n\n'
        header += '#PDF calculation setup\n'
        header += 'mode = electrons\n'
        header += f'composition = {st.session_state.composition}\n'
        header += f'bgscale = {bgscale_int:.2f}\n'
        header += f'rpoly = {rpoly_int:.2f}\n'
        header += f'qmaxinst = {qmaxinst_int:.2f}\n'
        header += f'qmin = {qmin_int:.2f}\n'
        header += f'qmax = {qmax_int:.2f}\n'
        header += f'rmin = {st.session_state.rmin:.2f}\n'
        header += f'rmax = {st.session_state.rmax:.2f}\n'
        header += f'rstep = {st.session_state.rstep:.2f}\n\n'
        header += '# End of config --------------------------------------------------------------\n'
        header += '#### start data\n\n'
        header += '#S 1\n'
        header += '#L r(Å)  G(r)(Å^{-2})\n'
        
        csv_buffer.write(header)
        for r_val, g_val in zip(r_pdf, G_pdf):
            csv_buffer.write(f"{r_val:.6f} {g_val:.8f}\n")
        csv_content = csv_buffer.getvalue().encode('utf-8')
        
        st.markdown("### 📊 Plots")
        
        # Import functions for intermediate calculations
        from pdf_extraction import compute_f2avg, fit_polynomial_background
        
        q = st.session_state.q_data
        Iexp_orig = st.session_state.I_data  # Original, unmodified
        I_ref = st.session_state.I_ref
        
        # Compute intermediate values
        qstep = q[1] - q[0]
        q_f2, f2avg = compute_f2avg(
            formula=st.session_state.composition,
            x_max=qmax_int,
            x_step=qstep,
            qvalues=True,
            xray=False,
        )
        f2avg_interp = np.interp(q, q_f2, f2avg)
        
        # Modified intensity for plot 2
        Iexp_corrected = Iexp_orig.copy()
        if I_ref is not None:
            Iexp_corrected = Iexp_corrected - bgscale_int * I_ref
        
        mask_inf = q > 0.9 * qmax_int
        I_inf = np.mean(Iexp_corrected[mask_inf])
        
        Inorm = Iexp_corrected / f2avg_interp
        Fm = q * (Inorm / I_inf - 1)
        
        background = fit_polynomial_background(
            q, Fm, rpoly=rpoly_int, qmin=qmin_int, qmax=qmax_int
        )
        Fc = Fm - background
        
        # Create plots with new layout: I(q) and F(q) side by side, G(r) below
        fig = plt.figure(figsize=(11, 6))
        
        # Create grid: 2 rows, 4 columns for flexibility
        # Top row: I(q) and F(q) take 2 columns each
        # Bottom row: G(r) takes all 4 columns
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0:2])  # I(q) - left half
        ax2 = fig.add_subplot(gs[0, 2:4])  # F(q) - right half
        ax3 = fig.add_subplot(gs[1, :])    # G(r) - full width
        
        ax = [ax1, ax2, ax3]
        
        # Plot 1: Raw intensities - show Iexp BEFORE subtraction and bgscale*Iref
        mask_plot = (q >= qmin_int) & (q <= qmax_int)
        ax[0].plot(q, Iexp_orig, 'b-', linewidth=2, label="Iexp (raw)")
        if I_ref is not None:
            ax[0].plot(q, bgscale_int * I_ref, 'r-', linewidth=2, label=f"bgscale*Iref (bgscale={bgscale_int:.2f})")
            ax[0].legend()
        ax[0].set_xlabel("Q (Å⁻¹)", fontsize=10)
        ax[0].set_ylabel("Intensity", fontsize=10)
        ax[0].set_xlim([qmin_int, qmax_int])
        if len(Iexp_orig[mask_plot]) > 0:
            y_min = min(np.min(Iexp_orig[mask_plot]), np.min(bgscale_int * I_ref[mask_plot]) if I_ref is not None else np.min(Iexp_orig[mask_plot]))
            y_max = max(np.max(Iexp_orig[mask_plot]), np.max(bgscale_int * I_ref[mask_plot]) if I_ref is not None else np.max(Iexp_orig[mask_plot]))
            ax[0].set_ylim([y_min, y_max])
        ax[0].set_title("1. Raw Intensities (for bgscale adjustment)", fontsize=11, fontweight='bold')
        ax[0].grid(True, alpha=0.3)
        
        # Plot 2: Corrected structure factor
        ax[1].plot(q, Fc, 'b-', linewidth=2, label=f"F(Q) (rpoly={rpoly_int:.2f})")
        ax[1].set_xlabel("Q (Å⁻¹)", fontsize=10)
        ax[1].set_ylabel("F(Q)", fontsize=10)
        ax[1].set_xlim([qmin_int, qmax_int])
        Fc_valid = Fc[mask_plot][np.isfinite(Fc[mask_plot])]
        if len(Fc_valid) > 0:
            ax[1].set_ylim([np.min(Fc_valid), np.max(Fc_valid)])
        ax[1].set_title("2. Corrected Structure Factor", fontsize=11, fontweight='bold')
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()
        
        # Plot 3: Final PDF
        ax[2].plot(r_pdf, G_pdf, 'b-', linewidth=2, label=f"G(r) (rpoly={rpoly_int:.2f})")
        ax[2].set_xlabel("r (Å)", fontsize=10)
        ax[2].set_ylabel("G(r)", fontsize=10)
        ax[2].set_title("3. Radial Distribution Function (PDF)", fontsize=11, fontweight='bold')
        ax[2].grid(True, alpha=0.3)
        ax[2].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display download button after plots
        st.markdown("### 📋 Results")
        st.download_button(
            label="💾 Download PDF Results",
            data=csv_content,
            file_name=output_file,
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("💡 **ePDFsuite** - Interactive interface for PDF analysis from electron diffraction (SAED) data")
