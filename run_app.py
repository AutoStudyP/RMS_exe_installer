"""Entry point script for PyInstaller and direct execution"""

import sys
import os

# Ensure the package directory is in path when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snr_meter.main_window import main

if __name__ == "__main__":
    main()
