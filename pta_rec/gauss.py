from __future__ import division

import os
import glob
from itertools import combinations
import copy
import json
import sys

import matplotlib

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import libstempo as T
import libstempo.plot as LP
import libstempo.toasim as LT
from libstempo.libstempo import GWB

import dynesty
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals.parameter import function
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils, blocks

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import scipy.interpolate as interp


####################
# Setup 
####################

# fixed: set thread limits for the current Python process
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
matplotlib.use("Agg")  # fixed: safe backend for server without display

KAPPA = float(sys.argv[1])
KAPPA_TAG = f"{KAPPA:g}"
RUN_NAME = f"open2_gwb_kappa{KAPPA_TAG}"

datadir_in = "./"
cgfgdir = "configs/"
os.makedirs("./data", exist_ok=True)
datadir_out = f"data/psrE_kappa{KAPPA_TAG}/"
outdir = f"{datadir_out}final"                                                                                                        ##############################
chains_dir = f"data/chains_kappa{KAPPA_TAG}/mdc/{RUN_NAME}"


prefix_psr = "J"
Npsr = 20
coord = "cone"

cap_angle =180 * np.pi / 180.0
ra0 = np.pi / 2
dec0 = 0

psrcat_cat = np.genfromtxt(datadir_in + cgfgdir +  "psrcat_data.txt", skip_header=1, dtype="str", unpack=True)
rand_n = np.random.choice(len(psrcat_cat.T), Npsr, replace=False)
num_cat, pmra_cat, pmdec_cat, px_cat, rajd_cat, decjd_cat, f0_cat, f1_cat, dm_cat = psrcat_cat

