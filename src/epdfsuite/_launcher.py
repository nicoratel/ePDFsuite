import sys
import os


def main():
    app_file = os.path.join(os.path.dirname(__file__), "app_epdfsuite.py")
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", app_file]
    sys.exit(stcli.main())
