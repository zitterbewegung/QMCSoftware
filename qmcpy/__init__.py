from .discrete_distribution import *
from .true_measure import *
from .integrand import *
from .stopping_criterion import *
from .util import plot_proj 

import matplotlib.pyplot as plt
import importlib.resources as pkg_resources

import subprocess
import sys

def is_latex_installed():
  """
  Checks if LaTeX is installed on the system.

  Returns:
    True if LaTeX is installed, False otherwise.
  """
  try:
    if sys.platform == "win32":
      # Check for MikTex on Windows
      subprocess.run(["where", "latex"], capture_output=True, check=True)
    elif sys.platform == "darwin":
      # Check for MacTex on macOS
      subprocess.run(["which", "latex"], capture_output=True, check=True)
    else:  # Linux and other Unix-like systems
      subprocess.run(["which", "pdflatex"], capture_output=True, check=True)
    return True
  except (FileNotFoundError, subprocess.CalledProcessError):
    return False

def qmc_apply_style():
    """Apply the qmcpy matplotlib style to allow for a standard style."""
    style_file = 'qmcpy.mplstyle'

    if is_latex_installed:
        print("Please install latex it is required for this style")
        return
    
    try:
        with pkg_resources.path(__package__, style_file) as style_path:
            plt.style.use(style_path)
        print("qmcpy matplotlib style applied.")
    except FileNotFoundError:
        print(f"Style file {style_file} not found in package {__package__}.")

name = "qmcpy"
__version__ = "1.5"
