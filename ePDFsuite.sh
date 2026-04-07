#!/bin/bash

# ePDFsuite Streamlit App Launcher
# Activates conda environment and launches the interactive GUI

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Conda environment name
CONDA_ENV="epdfpy"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please ensure Anaconda/Miniconda is installed."
    exit 1
fi

# Activate environment
echo "🔧 Activating conda environment: $CONDA_ENV"
source activate $CONDA_ENV

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate environment $CONDA_ENV"
    exit 1
fi

# Launch Streamlit app
echo "🚀 Starting ePDFsuite Streamlit App..."
echo "📍 App URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run "$SCRIPT_DIR/src/epdfsuite/app_epdfsuite.py"

# Deactivate environment on exit
conda deactivate
