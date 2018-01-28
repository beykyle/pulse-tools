#!/usr/bin/env python

"""
rescon.py: Code to find the energy resolution of an organic scintillator by iteratively fitting a Gaussian convolution to un-broadened spectra simulated in Monte Carlo until it fits a measured gamma spectra.

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import waveform
import dataloader
import getwavedata
from scipy.ndimage.filters import gaussian_filter1d as gfilter
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
    def __init__(self , energy):
        self.energy = energy

    def gaussianKernel(self , width):
        gauss = np.exp( -self.energy**2 / width )
        return(gauss / np.sum(gauss) )

    def convolveWithGaussian(self , data , width):
        return(np.convolve(self.gaussianKernel(width) , data)[0:len(self.energy)] )


def test():
    energy   = np.linspace(0 , 2 , num=1000)
    spectrum = np.zeros(len(energy))

    for i , ch in enumerate(spectrum):
        spectrum[i] = + 5*np.random.rand()
        if energy[i] < 0.662:
            spectrum[i] += 50

    res = resFinder(energy)
    convolvedSpec = res.convolveWithGaussian( spectrum , 0.01 )
    visualizeConvolution( energy , spectrum , convolvedSpec )

def visualizeConvolution(energy , original , convolved ):
    plt.plot(energy , original  , label="original" )
    plt.plot(energy , convolved , label="convolved")
    plt.tight_layout()
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Frequency [Arbitrary Units]')
    plt.show()


if __name__ == '__main__':
    #wavedata = getwavedata.GetWaveData(sys.argv[1])
    test()


