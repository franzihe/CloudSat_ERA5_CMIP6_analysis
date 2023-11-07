import sys

sys.path.append(
    "/uio/kant/geo-metos-u1/franzihe/Documents/Python/globalsnow/eosc-nordic-climate-demonstrator"
)

import xarray as xr

# xr.set_options(display_style="html")
import intake
import cftime
import cartopy.crs as ccrs
import cartopy as cy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt


import xesmf as xe
from glob import glob
import pandas as pd
import numpy as np

from cmcrameri import cm
from scipy.stats import linregress
import dask 
import dask.array as da
import dask.distributed as distributed

import geocat.comp as gc
from datetime import datetime, timedelta

import h5py

from scipy.optimize import curve_fit


import wget
from matplotlib.colors import LogNorm, LinearSegmentedColormap, BoundaryNorm, TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from sklearn.metrics import r2_score
from urllib.error import HTTPError
import functions as fct


