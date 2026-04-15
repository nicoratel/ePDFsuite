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
    st.markdown(
        "**Step 1:** upload your files. "
        "**Step 2:** click *Create SAEDProcessor* to auto-detect the beam centre. "
        "**Step 3:** refine the centre manually if needed and click *Update*."
    )

    col_sample, col_ref = st.columns(2)

    # ========== SAMPLE COLUMN ==========
    with col_sample:
        st.markdown("### 🔵 Sample")

        # --- 1. Upload files ---
        st.markdown("#### 1️⃣ Upload Files")

        sample_image = st.file_uploader(
            "Diffraction image",
            type=["dm4", "dm3", "tif", "tiff"],
            key="sample_image",
        )
        sample_poni = st.file_uploader(
            "PONI file (optional)",
            type=["poni"],
            key="sample_poni",
        )

        _col_mlbl, _col_mbtn = st.columns([3, 2])
        with _col_mlbl:
            st.caption("Mask file (.edf, optional)")
        with _col_mbtn:
            if st.button("🎨 Draw mask", disabled=(sample_image is None),
                         key="sample_draw_mask",
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
            "Mask file (.edf, optional)",
            type=["edf"],
            key="sample_mask",
            label_visibility="collapsed",
        )
        sample_mtf = st.file_uploader(
            "MTF file (optional)",
            key="sample_mtf",
        )

        # MTF deconvolution parameters
        if sample_mtf is not None:
            # When the file changes, read epsilon from column 3 of the MTF file
            if st.session_state.get("_sample_mtf_name") != sample_mtf.name:
                st.session_state["_sample_mtf_name"] = sample_mtf.name
                try:
                    import io as _io
                    _mtf_arr = np.loadtxt(_io.BytesIO(sample_mtf.getbuffer()), comments='#')
                    st.session_state["sample_mtf_eps"] = float(max(_mtf_arr[0, 2], 1e-10))
                except Exception:
                    st.session_state.setdefault("sample_mtf_eps", 1e-2)

            with st.expander("🔧 MTF Deconvolution Parameters", expanded=False):
                _s_mtf_eps = st.number_input("Wiener ε", format="%.2e",
                                              min_value=1e-10, key="sample_mtf_eps")
        else:
            _s_mtf_eps = st.session_state.get("sample_mtf_eps", None)

        # Save temp paths to session_state on every rerun
        if sample_image is not None:
            _suffix = "." + sample_image.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix) as _f:
                _f.write(sample_image.getbuffer())
                st.session_state.sample_tmp_path = _f.name
            # Invalidate processor when image file changes
            if st.session_state.get("_sample_image_name") != sample_image.name:
                st.session_state["_sample_image_name"] = sample_image.name
                st.session_state.sample_processor = None
                st.session_state.pop("sample_cx", None)
                st.session_state.pop("sample_cy", None)
        else:
            st.session_state.sample_tmp_path = None

        if sample_poni is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".poni") as _f:
                _f.write(sample_poni.getbuffer())
                st.session_state.sample_poni_path = _f.name
        else:
            st.session_state.sample_poni_path = None

        if sample_mask is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as _f:
                _f.write(sample_mask.getbuffer())
                st.session_state.sample_mask_path = _f.name
        else:
            st.session_state.sample_mask_path = None

        if sample_mtf is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as _f:
                _f.write(sample_mtf.getbuffer())
                st.session_state.sample_mtf_path = _f.name
        else:
            st.session_state.sample_mtf_path = None

        # --- 2. Create SAEDProcessor ---
        st.markdown("#### 2️⃣ Create Processor & Detect Centre")

        if st.button("🚀 Create SAEDProcessor", disabled=(sample_image is None),
                     key="sample_create", type="primary"):
            with st.spinner("Creating processor and detecting beam centre via isocurve…"):
                try:
                    _proc = SAEDProcessor(
                        st.session_state.sample_tmp_path,
                        poni_file=st.session_state.sample_poni_path,
                        mask=st.session_state.sample_mask_path,
                        mtf_file=st.session_state.sample_mtf_path,
                        wiener_epsilon=_s_mtf_eps,
                        verbose=False,
                    )
                    st.session_state.sample_processor = _proc
                    st.session_state.sample_cx = int(round(_proc.center[0]))
                    st.session_state.sample_cy = int(round(_proc.center[1]))
                    st.rerun()
                except Exception as _e:
                    st.error(f"❌ {_e}")
                    import traceback
                    st.error(traceback.format_exc())

        # --- 3. Image + editable centre ---
        if st.session_state.get("sample_processor") is not None:
            _proc = st.session_state.sample_processor
            _cx = st.session_state.get("sample_cx", int(round(_proc.center[0])))
            _cy = st.session_state.get("sample_cy", int(round(_proc.center[1])))

            st.success(
                f"✅ Processor ready — auto-detected centre: "
                f"({int(round(_proc.center[0]))}, {int(round(_proc.center[1]))})"
            )

            _img_norm = np.log10(_proc.img / np.max(_proc.img) + 1e-4)
            _fig_s = go.Figure(data=go.Heatmap(
                z=_img_norm, colorscale="Jet",
                hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>",
                showscale=False,
            ))
            _fig_s.add_trace(go.Scatter(
                x=[_cx], y=[_cy], mode="markers",
                marker=dict(symbol="cross", size=14, color="white",
                            line=dict(color="white", width=2)),
                showlegend=False,
                hovertemplate=f"Centre: ({_cx}, {_cy})<extra></extra>",
            ))
            _fig_s.update_layout(
                title="Sample image — crosshair = current beam centre",
                xaxis_title="X (pixels)", yaxis_title="Y (pixels)",
                height=480,
                yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain="domain"),
                margin=dict(t=50, b=40, l=50, r=10),
            )
            st.plotly_chart(_fig_s, use_container_width=True)

            st.markdown("**Beam Centre — edit if needed, then click Update**")
            _col_sx, _col_sy, _col_supd = st.columns([2, 2, 2])
            with _col_sx:
                st.number_input("Center X", step=1, key="sample_cx")
            with _col_sy:
                st.number_input("Center Y", step=1, key="sample_cy")
            with _col_supd:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Update", key="sample_update_center"):
                    _proc.center = (int(st.session_state.sample_cx),
                                    int(st.session_state.sample_cy))
                    st.success(
                        f"✅ Centre updated to ({_proc.center[0]}, {_proc.center[1]})"
                    )

        elif sample_image is None:
            st.info("📤 Upload a diffraction image above.")
        else:
            st.info("👆 Click **Create SAEDProcessor** to detect the beam centre.")

    # ========== REFERENCE COLUMN ==========
    with col_ref:
        st.markdown("### 🟠 Reference (optional)")

        # --- 1. Upload files ---
        st.markdown("#### 1️⃣ Upload Files")

        ref_image = st.file_uploader(
            "Diffraction image",
            type=["dm4", "dm3", "tif", "tiff"],
            key="ref_image",
        )
        ref_poni = st.file_uploader(
            "PONI file (optional, defaults to sample PONI)",
            type=["poni"],
            key="ref_poni",
        )
        if ref_poni is None and sample_poni is not None:
            st.caption(f"ℹ️ Will use sample PONI: {sample_poni.name}")

        _col_rmlbl, _col_rmbtn = st.columns([3, 2])
        with _col_rmlbl:
            st.caption("Mask file (.edf, optional, defaults to sample mask)")
        with _col_rmbtn:
            if st.button("🎨 Draw mask", disabled=(ref_image is None),
                         key="ref_draw_mask",
                         help="Opens the pyFAI-drawmask GUI on the reference image"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as _tmp:
                    _tmp.write(ref_image.getbuffer())
                    _tmp_path = _tmp.name
                try:
                    draw_mask(_tmp_path)
                finally:
                    if os.path.exists(_tmp_path):
                        os.remove(_tmp_path)
        ref_mask = st.file_uploader(
            "Mask file (.edf, optional)",
            type=["edf"],
            key="ref_mask",
            label_visibility="collapsed",
        )
        if ref_mask is None and sample_mask is not None:
            st.caption(f"ℹ️ Will use sample mask: {sample_mask.name}")

        ref_mtf = st.file_uploader(
            "MTF file (optional, defaults to sample MTF)",
            key="ref_mtf",
        )
        if ref_mtf is None and sample_mtf is not None:
            st.caption(f"ℹ️ Will use sample MTF: {sample_mtf.name}")

        # MTF deconvolution parameters
        if ref_mtf is not None:
            # When the file changes, read epsilon from column 3 of the MTF file
            if st.session_state.get("_ref_mtf_name") != ref_mtf.name:
                st.session_state["_ref_mtf_name"] = ref_mtf.name
                try:
                    import io as _io
                    _mtf_arr = np.loadtxt(_io.BytesIO(ref_mtf.getbuffer()), comments='#')
                    st.session_state["ref_mtf_eps"] = float(max(_mtf_arr[0, 2], 1e-10))
                except Exception:
                    st.session_state.setdefault("ref_mtf_eps",
                        st.session_state.get("sample_mtf_eps", 1e-2))

            with st.expander("🔧 MTF Deconvolution Parameters", expanded=False):
                _r_mtf_eps = st.number_input("Wiener ε", format="%.2e",
                                              min_value=1e-10, key="ref_mtf_eps")
        else:
            _r_mtf_eps = st.session_state.get("ref_mtf_eps",
                                               st.session_state.get("sample_mtf_eps", None))

        # Save temp paths to session_state on every rerun
        if ref_image is not None:
            _suffix = "." + ref_image.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix) as _f:
                _f.write(ref_image.getbuffer())
                st.session_state.ref_tmp_path = _f.name
            if st.session_state.get("_ref_image_name") != ref_image.name:
                st.session_state["_ref_image_name"] = ref_image.name
                st.session_state.ref_processor = None
                st.session_state.pop("ref_cx", None)
                st.session_state.pop("ref_cy", None)
        else:
            st.session_state.ref_tmp_path = None

        if ref_poni is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".poni") as _f:
                _f.write(ref_poni.getbuffer())
                st.session_state.ref_poni_path = _f.name
        else:
            st.session_state.ref_poni_path = st.session_state.get("sample_poni_path")

        if ref_mask is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as _f:
                _f.write(ref_mask.getbuffer())
                st.session_state.ref_mask_path = _f.name
        else:
            st.session_state.ref_mask_path = st.session_state.get("sample_mask_path")

        if ref_mtf is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as _f:
                _f.write(ref_mtf.getbuffer())
                st.session_state.ref_mtf_path = _f.name
        else:
            st.session_state.ref_mtf_path = st.session_state.get("sample_mtf_path")

        # --- 2. Create SAEDProcessor ---
        st.markdown("#### 2️⃣ Create Processor & Detect Centre")

        if st.button("🚀 Create SAEDProcessor", disabled=(ref_image is None),
                     key="ref_create", type="primary"):
            with st.spinner("Creating processor and detecting beam centre via isocurve…"):
                try:
                    _proc = SAEDProcessor(
                        st.session_state.ref_tmp_path,
                        poni_file=st.session_state.ref_poni_path,
                        mask=st.session_state.ref_mask_path,
                        mtf_file=st.session_state.ref_mtf_path,
                        wiener_epsilon=_r_mtf_eps,
                        verbose=False,
                    )
                    st.session_state.ref_processor = _proc
                    st.session_state.ref_cx = int(round(_proc.center[0]))
                    st.session_state.ref_cy = int(round(_proc.center[1]))
                    st.rerun()
                except Exception as _e:
                    st.error(f"❌ {_e}")
                    import traceback
                    st.error(traceback.format_exc())

        # --- 3. Image + editable centre ---
        if st.session_state.get("ref_processor") is not None:
            _proc = st.session_state.ref_processor
            _cx = st.session_state.get("ref_cx", int(round(_proc.center[0])))
            _cy = st.session_state.get("ref_cy", int(round(_proc.center[1])))

            st.success(
                f"✅ Processor ready — auto-detected centre: "
                f"({int(round(_proc.center[0]))}, {int(round(_proc.center[1]))})"
            )

            _img_norm = np.log10(_proc.img / np.max(_proc.img) + 1e-4)
            _fig_r = go.Figure(data=go.Heatmap(
                z=_img_norm, colorscale="Jet",
                hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>",
                showscale=False,
            ))
            _fig_r.add_trace(go.Scatter(
                x=[_cx], y=[_cy], mode="markers",
                marker=dict(symbol="cross", size=14, color="white",
                            line=dict(color="white", width=2)),
                showlegend=False,
                hovertemplate=f"Centre: ({_cx}, {_cy})<extra></extra>",
            ))
            _fig_r.update_layout(
                title="Reference image — crosshair = current beam centre",
                xaxis_title="X (pixels)", yaxis_title="Y (pixels)",
                height=480,
                yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain="domain"),
                margin=dict(t=50, b=40, l=50, r=10),
            )
            st.plotly_chart(_fig_r, use_container_width=True)

            st.markdown("**Beam Centre — edit if needed, then click Update**")
            _col_rx, _col_ry, _col_rupd = st.columns([2, 2, 2])
            with _col_rx:
                st.number_input("Center X", step=1, key="ref_cx")
            with _col_ry:
                st.number_input("Center Y", step=1, key="ref_cy")
            with _col_rupd:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Update", key="ref_update_center"):
                    _proc.center = (int(st.session_state.ref_cx),
                                    int(st.session_state.ref_cy))
                    st.success(
                        f"✅ Centre updated to ({_proc.center[0]}, {_proc.center[1]})"
                    )

        elif ref_image is None:
            st.info("📤 Upload a reference image above (optional).")
        else:
            st.info("👆 Click **Create SAEDProcessor** to detect the beam centre.")

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

            # Sync centre from the number-input fields (Tab 1) before integrating.
            # This ensures the value displayed in the boxes is always the one used,
            # whether it was auto-detected or manually entered.
            _s_proc = st.session_state.sample_processor
            _s_proc.center = (
                int(st.session_state.get("sample_cx", round(_s_proc.center[0]))),
                int(st.session_state.get("sample_cy", round(_s_proc.center[1]))),
            )
            q_sample, I_sample = _s_proc.integrate(plot=False)

            status_text.text("⏳ Integrating reference...")
            progress_bar.progress(50)

            if st.session_state.ref_processor is not None:
                _r_proc = st.session_state.ref_processor
                _r_proc.center = (
                    int(st.session_state.get("ref_cx", round(_r_proc.center[0]))),
                    int(st.session_state.get("ref_cy", round(_r_proc.center[1]))),
                )
                q_ref, I_ref = _r_proc.integrate(plot=False)
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
