# Launching the ePDFsuite GUI

This document describes how to launch the ePDFsuite Streamlit application.

---

## Prerequisites

The conda environment `epdfpy` must be created and the package installed.  
If not already done, install the package in editable mode (once only):

```bash
conda activate epdfpy
pip install -e /path/to/ePDFsuite
```

> In editable mode (`-e`), any change to the source code is immediately reflected without reinstalling.

---

## Option 1 — Python command (recommended)

Once the package is installed, an `epdfsuite-app` command is available in the environment:

```bash
conda activate epdfpy
epdfsuite-app
```

The application opens automatically in the browser at:  
`http://localhost:8501`

To stop the application: **Ctrl+C** in the terminal.

---

## Option 2 — Bash script

A shell script is provided at the root of the project. It automatically activates the `epdfpy` conda environment before launching the application.

```bash
bash /path/to/ePDFsuite/ePDFsuite.sh
```

You can also make it executable for a simpler call:

```bash
chmod +x /path/to/ePDFsuite/ePDFsuite.sh
/path/to/ePDFsuite/ePDFsuite.sh
```

---

## Option 3 — Direct call with Streamlit

From any directory, with the conda environment activated:

```bash
conda activate epdfpy
streamlit run /path/to/ePDFsuite/src/epdfsuite/app_epdfsuite.py
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `command not found: epdfsuite-app` | Reinstall with `pip install -e .` in the `epdfpy` environment |
| `ModuleNotFoundError` | Make sure the `epdfpy` environment is activated |
| Port 8501 already in use | Launch on another port: `streamlit run app_epdfsuite.py --server.port 8502` |
| Blank page in the browser | Wait a few seconds and refresh, or manually open `http://localhost:8501` |
