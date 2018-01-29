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
from scipy.optimize import curve_fit as fit

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

class Experiment:
    def __init__(self , energy):
        self.energy = energy

    def readFromFile(self , config ):
        wavedata = getwavedata.GetWaveData(config)
        spectrum  = np.zeros(len(self.energy))
        #find spectrum
        #convert to LO
        return(spectrum)

    def makeTestSpectrum(self , comptonErg , a , b , c):
        spectrum = np.zeros( len(self.energy) )
        for i , ch in enumerate(spectrum):
            spectrum[i] += 0.6*np.random.rand()
            if self.energy[i] < comptonErg:
                spectrum[i] += 20 * self.energy[i]**2 -  10 * self.energy[i] + 6

        res = ResolutionFitter(self.energy)
        spectrum = res.convolveWithGaussian( spectrum , a , b , c ) + 0.6 * np.random.rand(len(self.energy))
        return(spectrum)

class Simulation:
    def __init__(self , energy):
        self.energy = energy

    def readFromFile(self , fname ):
        spectrum  = np.zeros(len(self.energy))
        #find spectrum
        return(spectrum)

    def makeTestSpectrum(self , comptonErg , a , b , c):
        spectrum = np.zeros( len(self.energy) )
        for i , ch in enumerate(spectrum):
            spectrum[i] += 0.2*np.random.rand()
            if self.energy[i] < comptonErg:
                spectrum[i] += 20 * self.energy[i]**2 -  10 * self.energy[i] + 6

        return(spectrum)

class ResolutionFitter:
    def __init__(self , energy , **kwargs):
        self.energy = energy
        for key , value in kwargs.items():
            setattr(self , key  , value)

    def kernelFWHM(self , erg , a , b, c):
        return( a + b * np.sqrt( erg + c * erg ** 2) )

    def gaussianKernel(self , erg ,  a , b , c ):
        gauss = np.exp( -self.energy**2 * (1.6651092223) / (self.kernelFWHM(erg , a , b , c) ) )
        return(gauss / np.sum(gauss) )

    def convolveWithGaussian(self , data , a , b , c):
        convolved = np.zeros(1)
        n = len(self.energy)
        for i , dval in enumerate(data):
            shiftGauss = np.append(  np.zeros(len(self.energy) ) , self.gaussianKernel(self.energy[i] , a , b , c) )
            convolved = np.append(convolved ,  np.dot( shiftGauss[n - i:2*n - i] ,  data ) )

        return( convolved[0:n] )


    def findComptonEdge(self , spec):
        diff1 = np.diff(spec)
        k = np.where(diff1 == diff1.min())
        if self.loud == True:
            plt.plot(self.energy[1:-1] , spec[1:-1]      , label="spectrum"       )
            plt.plot(self.energy[1:]   , diff1           , label="1st derivative" )
            plt.plot(self.energy[k]    , spec[k] , '*r'  , label="Compton Edge"   )
            plt.legend()
            plt.show()

        return(k[0][0])

    def cutBaseline(self , experimental , simulated ):
        n = len(self.energy)
        k = self.findComptonEdge(simulated)
        stop = n - n // 3 + k // 3

        self.energy = self.energy[0:stop]
        return(experimental[0:stop] , simulated[0:stop])

    def runFit(self , experimental , simulated , a0 , b0 , c0):
        experimental , simulated = self.cutBaseline(experimental , simulated)

        popt , pcov = fit(self.convolveWithGaussian , simulated , experimental ,
                            p0 = (a0 , b0 , c0) ,
                            bounds = ( (0 , 0 , 0 ) , (1 , 1 , np.inf) )
                          )

        broadened = self.convolveWithGaussian(simulated , *popt)

        print("  a= " , popt[0]  , "+/- " , pcov[0][0] ," ",
              ", b= " , popt[1]  , "+/- " , pcov[0][0] ," ",
              ", c= " , popt[2]  , "+/- " , pcov[0][0] ," ")

        return(self.energy , experimental , simulated , broadened , popt[0] , popt[1] , popt[2])

def test():
    energy   = np.linspace(0 , 2 , num=100)

    exp = Experiment(energy)
    sim = Simulation(energy)
    res = ResolutionFitter(energy , loud=True)

    experimental = exp.makeTestSpectrum(0.662 , 0.04 , 0.015 , 8)
    simulation   = sim.makeTestSpectrum(0.662 , 0.04 , 0.015 , 8)
    energy , experimental , simulation , broadened , a , b , c = res.runFit( experimental , simulation , 0.04 , 0.015 , 8)
    visualizeConvolution( energy , experimental , simulation , broadened )

def visualizeConvolution(energy , experimental , simulation , broadened ):
    plt.plot(energy , experimental  , label="experimental" )
    plt.plot(energy , simulation    , label="simulated")
    plt.plot(energy , broadened     , label="broadened")
   # plt.plot(np.linspace(0 , max(energy) , num=len(convolved)) , convolved , label="convolved")
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Frequency [Arbitrary Units]')
    plt.show()

if __name__ == '__main__':
    test()


