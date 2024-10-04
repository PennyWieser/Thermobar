__author__ = 'Penny Wieser, Maurizio Petrelli, Jordan Lubbers, Sinan, Eric Wieser'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import ternary
from scipy import interpolate


# This has the core calculations, e.g., molar fractions, cation fractions etc.
from Thermobar.core import *
# This has functions for import, as well as export to other tools like VESIcal
from Thermobar.import_export import *
# This has the functions and equations for liquid-only thermometry
from Thermobar.liquid_thermometers import *
# This calculates lines for rhodes diagrams etc.
from Thermobar.mineral_equilibrium import *
# This has the functions and equations for liquid-olivine and olivine-spinel thermometry
from Thermobar.olivine_liquid_olivine_spinel_thermometry import *
# This has the functions for adding noise, averaging samples by sample name etc.
from Thermobar.noise_averaging import *
# This has functions for orthopyroxene thermobarometry
from Thermobar.orthopyroxene_thermobarometry import *
# This has functions for clinopyroxene thermobarometry
from Thermobar.clinopyroxene_thermobarometry import *
# This has functions for clinopyroxene thermobarometry
from Thermobar.Nimis_1999 import *
# This has functions for two-pyroxene thermobarometry
from Thermobar.two_pyroxene import *
# This has functions for pyroxenes-garnet thermobarometry
from Thermobar.pyroxenes_garnet import *
# Feldspar functions
from Thermobar.feldspar import *
# Amphibole functions
from Thermobar.amphibole import *
# plotting, R2 etc
from Thermobar.plotting import *
# Viscosity
from Thermobar.viscosity import *
# calibration
from Thermobar.calibration_plots import *
# Garnet
from Thermobar.garnet import *
from Thermobar.garnet_plot import*
from Thermobar.garnet_class import*
# Geotherm
from Thermobar.geotherm import *
# Density profiles
from Thermobar.density_profiles import *
# CHOMPI
from Thermobar.chompi import *
# This deals with importing Aztec mineral and melt data
from Thermobar.aztecloading import *

# version
from ._version import __version__
