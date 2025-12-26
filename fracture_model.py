# fracture_model.py
# This file is a shim to maintain backward compatibility for the standalone release.
# It imports the actual implementation from the local FractureCore library.

import sys
import os

try:
    from FractureCore.fracture_model import FractalBasedFractureGenerator, EllipticalFracture
except ImportError:
    raise ImportError("Could not import FractureCore. Please ensure the 'FractureCore' directory exists in the current directory.")