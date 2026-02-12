import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
import os
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    st.success("👋 Thanks for using ePDFsuite! Session ended.")
    st.stop()

# Create three tabs
# Create two tabs (Geometric Calibration tab removed - requires Qt which doesn't work with Streamlit)
tab1, tab2 = st.tabs(["📸 Plot Data", "📈 PDF Extraction"])

# ============================================================================
# TAB 1: PLOT DATA (FORMERLY TAB 2)
# ============================================================================
with tab1:
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
            
            # Create matplotlib figure for sample image with beam center and LogNorm
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            im = ax.imshow(img / np.max(img), cmap='gray',
                          norm=LogNorm(vmin=1e-4, vmax=1))
            # Plot center marker
            ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2,
                   label=f'Center: ({center_x:.1f}, {center_y:.1f})')
            ax.set_title("Sample Image with Recalibrated Center")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.legend(fontsize=11)
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
# TAB 2: PDF EXTRACTION (FORMERLY TAB 3)
# ============================================================================
with tab2:
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

        st.markdown("---")
        st.info("💡 **PONI files** contain geometric calibration data of the camera. They can be obtained using **pyFAI-calib2** (from the PyFAI Python library) from diffraction data of a polycrystalline standard specimen (e.g., Au, Si).")
    
    # ========== DEFAULT VALUES ==========
    _default_bgscale = 1.0
    _default_qmin = 1.5
    _default_qmax = 24.0
    _default_qmaxinst = 24.0
    _default_rpoly = 1.4
    _default_lorch = True
    _default_composition = "Au"
    
    # ========== BEAMSTOP OPTION ==========
    st.markdown("### 🔍 Image Processing")
    beamstop = st.checkbox("Beamstop present on sample images", value=False)
    
    # ========== OUTPUT PARAMETERS SECTION ==========
    st.markdown("## 📋 Input Parameters")
    
    composition = st.text_input("Chemical composition", value=_default_composition, placeholder="e.g., Au, NaCl, Au3Cu")
    
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
        
        # Create two columns: left for controls, right for plots
        col_controls, col_plots = st.columns([1.2, 2.8], gap="large")
        
        q_max_data = float(np.max(st.session_state.q_data))
        
        # Put all sliders in LEFT column
        with col_controls:
            st.markdown("### 🎚️ Parameters")
            bgscale_int = st.slider("bgscale", 0.0, 2.5, _default_bgscale, 0.01, key="bgscale_slider")
            qmin_int = st.slider("qmin (Å⁻¹)", 0.1, q_max_data, _default_qmin, 0.1, key="qmin_slider")
            qmax_int = st.slider("qmax (Å⁻¹)", float(np.min(st.session_state.q_data)), q_max_data, _default_qmax, 0.1, key="qmax_slider")
            rpoly_int = st.slider("rpoly", 0.1, 10.0, _default_rpoly, 0.1, key="rpoly_slider")
            qmaxinst_int = st.slider("qmaxinst (Å⁻¹)", float(np.min(st.session_state.q_data)), q_max_data, _default_qmaxinst, 0.1, key="qmaxinst_slider")
            lorch_int = st.checkbox("Lorch window correction", value=_default_lorch, key="lorch_checkbox")
            
            st.markdown("---")
            st.markdown("### 📥 Download")
        
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
        
        # Import functions for intermediate calculations
        from pdf_extraction import compute_f2avg, fit_polynomial_background
        
        # Display plots in RIGHT column
        with col_plots:
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
        
            # Create 3 separate figures with individual legends, maintaining original layout
            mask_plot = (q >= qmin_int) & (q <= qmax_int)
            
            # ===== FIGURE 1: Raw Intensities =====
            fig1 = go.Figure()
            
            q_plot1 = q.tolist()
            iexp_plot1 = Iexp_orig.tolist()
            
            fig1.add_trace(
                go.Scatter(x=q_plot1, y=iexp_plot1, mode='lines', name='Iexp (raw)',
                          line=dict(color='blue', width=2),
                          hovertemplate='Q: %{x:.3f}<br>I: %{y:.3e}<extra></extra>')
            )
            
            if I_ref is not None:
                I_ref_bgscaled = (bgscale_int * I_ref).tolist()
                fig1.add_trace(
                    go.Scatter(x=q_plot1, y=I_ref_bgscaled, mode='lines',
                              name=f'bgscale×Iref (scale={bgscale_int:.2f})',
                              line=dict(color='red', width=2),
                              hovertemplate='Q: %{x:.3f}<br>I: %{y:.3e}<extra></extra>')
                )
            
            # Calculate Y-axis limits based on data in [qmin, qmax]
            iexp_in_range = Iexp_orig[mask_plot]
            y_min_plot1 = np.min(iexp_in_range) if len(iexp_in_range) > 0 else 0
            y_max_plot1 = np.max(iexp_in_range) if len(iexp_in_range) > 0 else 1
            if I_ref is not None:
                iref_in_range = bgscale_int * I_ref[mask_plot]
                y_min_plot1 = min(y_min_plot1, np.min(iref_in_range))
                y_max_plot1 = max(y_max_plot1, np.max(iref_in_range))
            
            y_margin = 0.05 * (y_max_plot1 - y_min_plot1)
            
            fig1.update_layout(
                title="1. Raw Intensities (for bgscale adjustment)",
                xaxis_title="Q (Å⁻¹)",
                yaxis_title="Intensity",
                hovermode='x unified',
                showlegend=True,
                legend=dict(x=0.7, y=0.95),
                height=350,
                margin=dict(l=60, r=40, t=60, b=50)
            )
            fig1.update_xaxes(range=[qmin_int, qmax_int])
            fig1.update_yaxes(range=[y_min_plot1 - y_margin, y_max_plot1 + y_margin])
            
            # ===== FIGURE 2: Corrected Structure Factor =====
            fig2 = go.Figure()
            
            q_plot2 = q.tolist()
            fc_plot2 = Fc.tolist()
            
            fig2.add_trace(
                go.Scatter(x=q_plot2, y=fc_plot2, mode='lines', 
                          name=f'F(Q) (rpoly={rpoly_int:.2f})',
                          line=dict(color='darkblue', width=2),
                          hovertemplate='Q: %{x:.3f}<br>F(Q): %{y:.3e}<extra></extra>')
            )
            
            # Calculate Y-axis limits for F(Q) based on data in [qmin, qmax]
            fc_in_range = Fc[mask_plot]
            fc_valid = fc_in_range[np.isfinite(fc_in_range)]
            y_min_plot2 = np.min(fc_valid) if len(fc_valid) > 0 else 0
            y_max_plot2 = np.max(fc_valid) if len(fc_valid) > 0 else 1
            
            y_margin2 = 0.05 * (y_max_plot2 - y_min_plot2)
            
            fig2.update_layout(
                title="2. Corrected Structure Factor",
                xaxis_title="Q (Å⁻¹)",
                yaxis_title="F(Q)",
                hovermode='x unified',
                showlegend=True,
                legend=dict(x=0.7, y=0.95),
                height=350,
                margin=dict(l=60, r=40, t=60, b=50)
            )
            fig2.update_xaxes(range=[qmin_int, qmax_int])
            fig2.update_yaxes(range=[y_min_plot2 - y_margin2, y_max_plot2 + y_margin2])
            
            # Display first two figures side by side
            col_fig1, col_fig2 = st.columns(2)
            with col_fig1:
                st.plotly_chart(fig1, use_container_width=True)
            with col_fig2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # ===== FIGURE 3: Radial Distribution Function =====
            fig3 = go.Figure()
            
            fig3.add_trace(
                go.Scatter(x=r_pdf, y=G_pdf, mode='lines', 
                          name=f'G(r) (rpoly={rpoly_int:.2f})',
                          line=dict(color='darkgreen', width=2),
                          hovertemplate='r: %{x:.3f}<br>G(r): %{y:.3e}<extra></extra>')
            )
            
            fig3.update_layout(
                title="3. Radial Distribution Function (PDF)",
                xaxis_title="r (Å)",
                yaxis_title="G(r)",
                hovermode='x unified',
                showlegend=True,
                legend=dict(x=0.7, y=0.95),
                height=350,
                margin=dict(l=60, r=40, t=60, b=50)
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Put download button in LEFT column
        with col_controls:
            st.download_button(
                label="💾 Download PDF Results",
                data=csv_content,
                file_name=output_file,
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("💡 **ePDFsuite** - Interactive interface for PDF analysis from electron diffraction (SAED) data")
