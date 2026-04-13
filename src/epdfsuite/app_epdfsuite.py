import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import the necessary modules from your package
from epdfsuite import SAEDProcessor, extract_epdf
from epdfsuite.recalibration import recalibrate_with_beamstop_noponi
from epdfsuite.filereader import load_data
from epdfsuite.utilities import draw_mask
from epdfsuite.pdf_extraction import compute_ePDF
import hyperspy.api as hs

# Initialize session state variables
if 'sample_processor' not in st.session_state:
    st.session_state.sample_processor = None
if 'ref_processor' not in st.session_state:
    st.session_state.ref_processor = None

# Configure Streamlit page
st.set_page_config(
    page_title="ePDFsuite - Interactive GUI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 ePDFsuite - Interactive PDF Extraction from SAED Images")

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

# Create two tabs (Define Sample/Ref first, then PDF Extraction)
tab1, tab2 = st.tabs(["📸 Define Sample and Reference", "📈 Extract ePDF"])

# ============================================================================
# TAB 1: DEFINE SAMPLE AND REFERENCE
# ============================================================================
with tab1:
    st.markdown("# 📸 Define Sample and Reference")
    st.markdown("**Upload your diffraction images, inspect them, and define beam centers.**")
    
    # Create two columns for Sample and Reference
    col_sample, col_ref = st.columns(2)
    
    # ========== SAMPLE COLUMN ==========
    with col_sample:
        st.markdown("### 🔵 Sample")
        
        st.markdown("**Sample Image**")
        sample_image = st.file_uploader(
            "Upload sample diffraction image",
            type=["dm4", "dm3", "tif", "tiff"],
            key="sample_image",
            label_visibility="collapsed"
        )
        
        st.markdown("**PONI File**")
        sample_poni = st.file_uploader(
            "Upload sample PONI file (optional)",
            type=["poni"],
            key="sample_poni",
            label_visibility="collapsed"
        )
        
        _col_mask, _col_btn = st.columns([3, 2])
        with _col_mask:
            st.markdown("**Mask File**")
        with _col_btn:
            if st.button("🎨 Click here to draw mask",
                         disabled=(sample_image is None),
                         help="Opens the pyFAI-drawmask GUI on the sample image"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as _tmp:
                    _tmp.write(sample_image.getbuffer())
                    _tmp_path = _tmp.name
                try:
                    draw_mask(_tmp_path)
                finally:
                    if os.path.exists(_tmp_path):
                        os.remove(_tmp_path)
        sample_mask = st.file_uploader(
            "Upload sample mask file (*.edf, optional)",
            type=["edf"],
            key="sample_mask",
            label_visibility="collapsed"
        )
        
        st.markdown("**MTF File**")
        sample_mtf = st.file_uploader(
            "Upload sample MTF file (optional)",
            key="sample_mtf",
            label_visibility="collapsed"
        )
        
        if sample_image is not None:
            # Save to temp file and load
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp_file:
                tmp_file.write(sample_image.getbuffer())
                sample_tmp_path = tmp_file.name
            
            try:
                metadata_sample, img_sample = load_data(sample_tmp_path, verbose=False)
                
                # Display image with Plotly (interactive coordinates)
                img_normalized = np.log10(img_sample / np.max(img_sample) + 1e-4)
                fig_sample = go.Figure(data=go.Heatmap(
                    z=img_normalized,
                    colorscale='Jet',
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2f}<extra></extra>',
                    showscale=False,
                ))
                fig_sample.update_layout(
                    title="Sample Image (hover to see coordinates)",
                    xaxis_title="X (pixels)",
                    yaxis_title="Y (pixels)",
                    height=500,
                    yaxis=dict(autorange='reversed', scaleanchor="x", scaleratio=1),  # Square aspect ratio
                    xaxis=dict(constrain='domain'),
                )
                st.plotly_chart(fig_sample, use_container_width=True)
                
                # Center input
                st.markdown("**Beam Center Coordinates**")
                col_cx, col_cy = st.columns(2)
                with col_cx:
                    sample_center_x = st.number_input("Center X", value=img_sample.shape[1]//2, step=1, key="sample_cx")
                with col_cy:
                    sample_center_y = st.number_input("Center Y", value=img_sample.shape[0]//2, step=1, key="sample_cy")
                
                # Store paths in session state
                st.session_state.sample_tmp_path = sample_tmp_path
                st.session_state.sample_center = (sample_center_x, sample_center_y)
                st.session_state.img_sample = img_sample
                
                # Handle PONI file
                if sample_poni is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".poni") as tmp_poni:
                        tmp_poni.write(sample_poni.getbuffer())
                        st.session_state.sample_poni_path = tmp_poni.name
                else:
                    st.session_state.sample_poni_path = None
                
                # Handle Mask file
                if sample_mask is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_mask:
                        tmp_mask.write(sample_mask.getbuffer())
                        st.session_state.sample_mask_path = tmp_mask.name
                else:
                    st.session_state.sample_mask_path = None
                
                # Handle MTF file
                if sample_mtf is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_mtf:
                        tmp_mtf.write(sample_mtf.getbuffer())
                        st.session_state.sample_mtf_path = tmp_mtf.name
                else:
                    st.session_state.sample_mtf_path = None
                
            except Exception as e:
                st.error(f"Error loading sample image: {e}")
        else:
            st.info("📤 Upload a sample image")
    
    # ========== REFERENCE COLUMN ==========
    with col_ref:
        st.markdown("### 🟠 Reference (optional)")
        
        st.markdown("**Reference Image**")
        ref_image = st.file_uploader(
            "Upload reference diffraction image",
            type=["dm4", "dm3", "tif", "tiff"],
            key="ref_image",
            label_visibility="collapsed"
        )
        
        st.markdown("**PONI File**")
        ref_poni = st.file_uploader(
            "Upload ref PONI file (optional, defaults to sample PONI)",
            type=["poni"],
            key="ref_poni",
            label_visibility="collapsed"
        )
        if ref_poni is None and sample_poni is not None:
            st.caption(f"ℹ️ Using sample PONI: {sample_poni.name}")
        elif ref_poni is not None:
            st.caption(f"✅ Using: {ref_poni.name}")
        
        st.markdown("**Mask File**")
        ref_mask = st.file_uploader(
            "Upload ref mask file (*.edf, optional)",
            type=["edf"],
            key="ref_mask",
            label_visibility="collapsed"
        )
        if ref_mask is None and sample_mask is not None:
            st.caption(f"ℹ️ Using sample mask: {sample_mask.name}")
        elif ref_mask is not None:
            st.caption(f"✅ Using: {ref_mask.name}")
        
        st.markdown("**MTF File**")
        ref_mtf = st.file_uploader(
            "Upload ref MTF file (optional, defaults to sample MTF)",
            key="ref_mtf",
            label_visibility="collapsed"
        )
        if ref_mtf is None and sample_mtf is not None:
            st.caption(f"ℹ️ Using sample MTF: {sample_mtf.name}")
        elif ref_mtf is not None:
            st.caption(f"✅ Using: {ref_mtf.name}")
        
        if ref_image is not None:
            # Save to temp file and load
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp_file:
                tmp_file.write(ref_image.getbuffer())
                ref_tmp_path = tmp_file.name
            
            try:
                metadata_ref, img_ref = load_data(ref_tmp_path, verbose=False)
                
                # Display image with Plotly (interactive coordinates)
                img_normalized = np.log10(img_ref / np.max(img_ref) + 1e-4)
                fig_ref = go.Figure(data=go.Heatmap(
                    z=img_normalized,
                    colorscale='Jet',
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2f}<extra></extra>',
                    showscale=False,
                ))
                fig_ref.update_layout(
                    title="Reference Image (hover to see coordinates)",
                    xaxis_title="X (pixels)",
                    yaxis_title="Y (pixels)",
                    height=500,
                    yaxis=dict(autorange='reversed', scaleanchor="x", scaleratio=1),  # Square aspect ratio
                    xaxis=dict(constrain='domain'),
                )
                st.plotly_chart(fig_ref, use_container_width=True)
                
                # Center input
                st.markdown("**Beam Center Coordinates**")
                col_cx, col_cy = st.columns(2)
                with col_cx:
                    ref_center_x = st.number_input("Center X", value=img_ref.shape[1]//2, step=1, key="ref_cx")
                with col_cy:
                    ref_center_y = st.number_input("Center Y", value=img_ref.shape[0]//2, step=1, key="ref_cy")
                
                # Store paths in session state
                st.session_state.ref_tmp_path = ref_tmp_path
                st.session_state.ref_center = (ref_center_x, ref_center_y)
                st.session_state.img_ref = img_ref
                
                # Handle PONI file (default to sample PONI if not provided)
                if ref_poni is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".poni") as tmp_poni:
                        tmp_poni.write(ref_poni.getbuffer())
                        st.session_state.ref_poni_path = tmp_poni.name
                else:
                    st.session_state.ref_poni_path = st.session_state.get('sample_poni_path', None)
                
                # Handle Mask file (default to sample mask if not provided)
                if ref_mask is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_mask:
                        tmp_mask.write(ref_mask.getbuffer())
                        st.session_state.ref_mask_path = tmp_mask.name
                else:
                    st.session_state.ref_mask_path = st.session_state.get('sample_mask_path', None)
                
                # Handle MTF file (default to sample MTF if not provided)
                if ref_mtf is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_mtf:
                        tmp_mtf.write(ref_mtf.getbuffer())
                        st.session_state.ref_mtf_path = tmp_mtf.name
                else:
                    st.session_state.ref_mtf_path = st.session_state.get('sample_mtf_path', None)
                
            except Exception as e:
                st.error(f"Error loading reference image: {e}")
        else:
            st.info("📤 Upload a reference image (optional)")
            st.session_state.ref_tmp_path = None
    
    # ========== VALIDATION BUTTON ==========
    st.markdown("---")
    
    # MTF deconvolution parameters (actifs uniquement si un fichier MTF est fourni)
    _any_mtf = (sample_mtf is not None) or (ref_mtf is not None)
    if _any_mtf:
        st.markdown("### 🔧 MTF Deconvolution Parameters")
        col_mtf1, col_mtf2 = st.columns(2)
        with col_mtf1:
            mtf_filter = st.selectbox(
                "Filter method",
                options=['rl', 'wiener'],
                format_func=lambda x: 'Richardson-Lucy (iterative)' if x == 'rl' else 'Wiener (inverse filter)',
                index=0,
                help="RL : Richardson-Lucy (iterative method) · Wiener : inverse filter with Wiener regularization (faster but may amplify noise if not careful with epsilon)"
            )
        with col_mtf2:
            if mtf_filter == 'rl':
                mtf_n_iterations = st.number_input("Iterations (RL)", value=50, min_value=1, step=1)
                mtf_wiener_epsilon = None
            else:
                mtf_wiener_epsilon = st.number_input(
                    "Wiener epsilon", value=1e-2, format="%.2e", min_value=1e-10,
                    help="Wiener regularization parameter (epsilon) to prevent noise amplification. Smaller values preserve more high-frequency details but may amplify noise. Adjust based on your data quality."
                )
                mtf_n_iterations = 50
    else:
        mtf_filter = 'rl'
        mtf_n_iterations = 50
        mtf_wiener_epsilon = None

    if st.button("✅ Validate and Create Processors", type="primary"):
        if sample_image is None:
            st.error("❌ Please upload a sample image first")
        else:
            try:
                # Create sample processor
                st.session_state.sample_processor = SAEDProcessor(
                    st.session_state.sample_tmp_path,
                    poni_file=st.session_state.sample_poni_path,
                    mask=st.session_state.get('sample_mask_path', None),
                    mtf_file=st.session_state.get('sample_mtf_path', None),
                    filter=mtf_filter,
                    n_iterations=mtf_n_iterations,
                    wiener_epsilon=mtf_wiener_epsilon,
                    verbose=False
                )
                st.session_state.sample_processor.initial_center = st.session_state.sample_center
                
                # Create reference processor if provided
                if st.session_state.ref_tmp_path is not None:
                    st.session_state.ref_processor = SAEDProcessor(
                        st.session_state.ref_tmp_path,
                        poni_file=st.session_state.ref_poni_path,
                        mask=st.session_state.get('ref_mask_path', None),
                        mtf_file=st.session_state.get('ref_mtf_path', None),
                        filter=mtf_filter,
                        n_iterations=mtf_n_iterations,
                        wiener_epsilon=mtf_wiener_epsilon,
                        verbose=False
                    )
                    st.session_state.ref_processor.initial_center = st.session_state.ref_center
                else:
                    st.session_state.ref_processor = None
                
                st.success("✅ Processors created successfully! Go to 'Extract ePDF' tab.")
                
            except Exception as e:
                st.error(f"❌ Error creating processors: {e}")
                import traceback
                st.error(traceback.format_exc())

# ============================================================================
# TAB 2: PDF EXTRACTION
# ============================================================================
with tab2:
    st.markdown("# 📈 Extract ePDF")
    st.markdown("**Calculate the Pair Distribution Function (PDF) from your processors. Adjust parameters with interactive sliders.**")
    
    # Check if processors are defined
    if st.session_state.sample_processor is None:
        st.warning("⚠️ Please define sample and reference in the 'Define Sample and Reference' tab first")
        st.stop()
    
    # ========== DEFAULT VALUES ==========
    _default_bgscale = 1.0
    _default_qmin = 1.5
    _default_qmax = 24.0
    _default_qmaxinst = 24.0
    _default_rpoly = 1.4
    _default_lorch = True
    _default_composition = "Au"
    
    # ========== INPUT PARAMETERS SECTION ==========
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
        # Progress placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("⏳ Integrating sample...")
            progress_bar.progress(25)
            
            # Integrate sample (MTF correction is applied automatically if mtf_file was provided)
            q_sample, I_sample = st.session_state.sample_processor.integrate(plot=False)
            
            status_text.text("⏳ Integrating reference...")
            progress_bar.progress(50)
            
            # Integrate reference if available (MTF correction is applied automatically if mtf_file was provided)
            if st.session_state.ref_processor is not None:
                q_ref, I_ref = st.session_state.ref_processor.integrate(plot=False)
                # Interpolate to sample q grid
                I_ref_interp = np.interp(q_sample, q_ref, I_ref)
            else:
                I_ref_interp = None
            
            # Store data in session state for interactive controls
            st.session_state.q_data = q_sample
            st.session_state.I_data = I_sample
            st.session_state.I_ref = I_ref_interp
            st.session_state.composition = composition
            st.session_state.rmin = rmin
            st.session_state.rmax = rmax
            st.session_state.rstep = rstep
            
            progress_bar.progress(100)
            status_text.text("✅ Integration complete!")
            st.session_state.data_ready = True
            
        except Exception as e:
            st.error(f"❌ Error during integration: {e}")
            import traceback
            st.error(traceback.format_exc())
    
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
            qmaxinst_int = st.slider("qmaxinst (Å⁻¹)", float(np.min(st.session_state.q_data)), q_max_data, _default_qmaxinst, 0.1, key="qmaxinst_slider")
            rpoly_int = st.slider("rpoly", 0.1, 10.0, _default_rpoly, 0.1, key="rpoly_slider")
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
        from epdfsuite.pdf_extraction import compute_f2avg, fit_polynomial_background
        
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
