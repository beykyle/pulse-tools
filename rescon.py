#!/usr/bin/env python

"""
rescon.py: Code to find the energy resolution of an organic scintillator by iteratively fitting a Gaussian convolution to un-broadened spectra simulated in Monte Carlo until it fits a measured gamma spectra.

"""

import numpy as np
import sys
import waveform
import dataloader
import getwavedata
from scipy import optimize as opt

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

class Experiment:
    pass

class Simulation:
    pass

class resFinder:
    pass

if __name__ == '__main__':
    wavedata = getwavedata.GetWaveData(sys.argv[1])
    print(wavedata)

