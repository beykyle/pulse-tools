#!/usr/bin/env python

"""
rescon.py: Code to find the energy resolution of an organic scintillator by iteratively fitting a Gaussian convolution to un-broadened spectra simulated in Monte Carlo until it fits a measured gamma spectra.

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import configparser
from scipy.ndimage.filters import gaussian_filter1d as gfilter
from scipy.optimize import curve_fit as fit
from scipy.signal import savgol_filter as savgol
from scipy.signal import find_peaks_cwt as peaks
from lmfit import Model

sys.path.append("../low_level/")

import dataloader
import getwavedata

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

class Spectrum:
    def __init__(self , spec):
        self.spec = spec

    def findComptonEdges(self):
        self.comptonEdges = [0]

    def findValleys(self):
        self.valley = [0]


class Experiment:
    def __init__(self , lines , bins, *args , **kwargs):
        if kwargs is not None:
                for key, value in kwargs.items():
                    if key == "loud" or key == "readSpec":
                        setattr(self , key , booleanize(value))
                    else:
                        setattr(self , key , value)

        self.bins  = int(bins)
        self.lines = lines

    def getData(self , detectors , keys):
        spec = []
        volts = []

        if(self.readSpec == False):
            print("getting experimental data from raw wave data")
            volts , spec = self.readFromFile(self.WaveConfig , detectors , keys)
            if(self.writeSpec == True):
                self.write_spec(volts , spec , keys)
        elif(self.readSpec == True):
            print("getting experimental data from spectra files")
            for key in keys:
                bins , hist = self.specReader(detectors[key].expFilename)
                spec.append(  np.array(hist) )
                volts.append( np.append( [0] , np.array(bins) ) )

        edges = []
        energies = []
        for i , (bins , hist) in enumerate(zip(volts , spec)):
            print("\n Now calibrating " + keys[i])
            edges = self.findComptonEdges(bins , hist , keys[i])
            energy = self.calibrate(bins , edges)
            energies.append(energy)

            if self.loud == True:

                plt.plot(energy[1:], hist)
                plt.title(keys[i])
                plt.xlabel("Light Output [MeV]")
                plt.ylabel("Counts")
                for line in self.lines:
                    lab = str(line) + " MeV"
                    plt.plot([line , line] , [0 , hist.max()] , '--' , label=lab)

                plt.legend()
                plt.show()

            # set experimental values in detector object
            detectors[keys[i]].setExpSpec(hist)
            detectors[keys[i]].setExpEnergy(energy)

        return(detectors)

    def specReader(self , fname):
        print("Reading spectrum from " + fname)
        bins = []
        hist = []
        with open(fname , "r") as inf:
            for line in inf.readlines():
                line = line.rstrip("\r\n").split(",")
                bins.append(float(line[0]))
                hist.append(float(line[1]))
        return(bins , hist)

    def write_spec(self , volts, spec , keys):
        for i , (bins , hist) in enumerate(zip(volts , spec)):
            fname =  "./spec_" + keys[i]+"_.tmp"
            print("writing spectrum to " + fname)
            with open(fname ,'w') as out:
                for b , h in zip(bins , hist):
                    out.write('{:1.5f}'.format(b) + "," + '{:1.8E}'.format(h) + '\r\n' )

    def readFromFile(self , config  , detectors , keys):
        spec = []
        volts = []
        chCount, ph, amp, tailInt, totalInt, cfd, ttt, extras, fullTime, flags, rms = getwavedata.GetWaveData(config , getWaves=False)
        self.numChannels = len(chCount)
        print("\n \n Read pulse integrated spectra from " + str(self.numChannels) + " channels. \n")

        for key in keys:
            print(key + " maps to channels " +  ' , '.join(str(e) for e in detectors[key].channels))
            integral = np.zeros( chCount[ detectors[key].channels[0] ] )
            for channel in detectors[key].channels:
                try:
                    integral =  integral + totalInt[channel][:chCount[channel]]
                except ValueError:
                    print("Detector " + key + " has channels that don't match up!")
                    print("Channels mapping to the same detector must physically read out from the same detector")
                    print("In this case, the number of pulses in channel " + str(channel) + " did not match the number of pulses in one of the other "
                            + str(len(detectors[key].channels) - 1) + " channels mapped to Detector " + key)
                    sys.exit()


            hist , bins = np.histogram(integral  , 200 )
            lastEdge = self.findLastEdge(bins , hist)
            integral = [x for x in integral if x <= bins[lastEdge] ]
            hist , bins = np.histogram( integral, self.bins )
            spec.append(hist)
            volts.append(bins)

        #for chan , bar in zip( range(0 , len(chCount) , 2) , [1,2,3,4,5,6,7,8] ):
        #    #implement arbitrary channel mapping
        #    integral = totalInt[chan+1][:chCount[chan+1]] + totalInt[chan][:chCount[chan]]
        #    hist , bins = np.histogram(integral  , 200 )
        #    lastEdge = self.findLastEdge(bins , hist)
        #    integral = [x for x in integral if x <= bins[lastEdge] ]
        #    hist , bins = np.histogram( integral, self.bins )
        #    spec.append(hist)
        #    volts.append(bins)

        return(volts , spec)

    def calibrate(self , bins , edges):
        if (len(self.lines) != len(edges)):
            print("Unable to find all the edges specified! Will just use first edge!")
            return(bins * (lines[0] / edges[0]) )
        elif len(self.lines) == 1 and len(edges) == 1:
            return(bins * (lines[0] / edges[0]) )
        elif len(self.lines) == 2 and len(edges) == 2:
            b = (self.lines[1] - self.lines[0] ) / (edges[1] - edges[0])
            a = self.lines[0] - b * edges[0]
            return(a + bins * b)
        else:
            # system is overconstrained
            # find a,b with least squares fitting
            # TODO
            return(bins * (lines[0] / edges[0]) )

    def findLastEdge(self , bins , hist):
        nhist = hist/ max(hist)
        diff1 = savgol( np.diff(nhist) , 15 , 3)
        #plt.plot(bins[1:-1] , diff1)
        #plt.plot(bins[1:]   , nhist)
        found = False
        lastEdge = len(hist) - 10
        for i , val in reversed(list(enumerate(hist))):
            if i < len(hist) - 10:
                if (found == False and sum( diff1[i-5:i]) < 0 and sum(nhist[i-5:i]) > 0.008 ):
                    found = True
                    lastEdge = i
        return(lastEdge)

    def findComptonEdges(self , bins , hist  , key):
        print("calibrating spectrum energy lines: " +  ' , '.join(str(e) for e in self.lines) + " MeV")
        print("Finding Compton Edges")
        splineWidth = int(math.ceil(self.bins/15))
        if splineWidth%2 == 0:
            splineWidth = splineWidth + 1
        nhist = savgol( (hist / hist.max())                   , splineWidth      , 3 )
        diff1 = savgol(  np.diff(hist/hist.max())             , splineWidth      , 3 )
        diff2 = savgol(  np.diff(np.diff(hist/hist.max() ))   , splineWidth*2+1  , 3 )
        #k  = np.where( diff1 == diff1.min() )
        #k2 = np.where( diff2 == diff2.min() )

        peakw = int(math.ceil(self.bins/20))
        peakid  = peaks( nhist ,  [peakw , peakw + 1 , peakw+2] )
        peakid2 = peaks( diff2 ,  [peakw , peakw + 1 , peakw+2] )
        #if len(peakid) < 3 or len(peakid2) < 3:
         #   raise NotImplementedError("Compton edges couldn't be found, and manual selection hasn't been implemented. Try again with more data!")
          #  return(0)

        if self.loud == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot( bins[1:]   , nhist            , label="smoothed spectrum"       )

        #find first line
        print("Looking for " + str(self.lines[0]) + " MeV Compton edge.")
        found = False
        for pk in peakid2:
            if found == False and pk > peakid[1]:
                stopEdge = pk

        intermediate =  nhist[peakid[1]:stopEdge] - 0.8 * nhist[peakid[1]]
        k = peakid[1] + np.where( intermediate**2 == (intermediate**2).min() )
        if (self.loud == True):
            ax.plot( bins[k]    , nhist[k-1]  , '*r' , label="Compton Edge" )
        print("Got it!")

        for i, line in reversed(list(enumerate(self.lines))):
            if( i == 0 ):
                pass
            else:
                print("Looking for " + str(line) + " MeV Compton edge.")
                lastEdge = self.findLastEdge(bins , hist)

                try:
                    peak2 = peakid[np.where(peakid < lastEdge)].max()
                except ValueError:
                    raise NotImplementedError("Compton edges couldn't be found, and manual selection hasn't been implemented. Try again with more data!")
                    return(0)

                intermediate = nhist[peak2:lastEdge] - 0.8 * nhist[peak2]
                k2 = peak2 + np.where( intermediate**2 == (intermediate**2).min() )
                if (self.loud == True):
                    ax.plot( bins[k2]    , nhist[k2-1]  , '*r' , label="Compton Edge" )
                nhist = nhist[:peak2]
                print("Got it!")

        if self.loud == True:
            plt.xlabel("Pulse Integral [V ns]")
            plt.ylabel("Relative Frequency")
            plt.title(key)
            plt.legend()
            plt.show()

            correct = input("Are the compton edges marked correctly? [yes/no]")
            if "y" in correct:
                return( [ bins[k[0][0]]  ]  )
            else:
                raise NotImplementedError("Sorry! We haven't added manually selected compton edges yet...")
                return(0)
        else:
            return( [ bins[k[0][0]]  , bins[k2[0][0]] ] )

    def makeTestSpectrum(self , comptonErg , a , b , c):
        spectrum = np.zeros( len(self.energy) )
        for i , ch in enumerate(spectrum):
            spectrum[i] += 0.6*np.random.rand()
            if self.energy[i] < comptonErg[0]:
                spectrum[i] += 20 * self.energy[i]**2 -  10 * self.energy[i] + 6
            if self.energy[i] < comptonErg[1]:
                spectrum[i] += 2 * (self.energy[i] - comptonErg[0])**2 -  1 * (self.energy[i] - comptonErg[0])

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

    def makeTestSpectrum(self , comptonErg ):
        spectrum = np.zeros( len(self.energy) )
        for i , ch in enumerate(spectrum):
            spectrum[i] += 0.2*np.random.rand()
            if self.energy[i] < comptonErg[0]:
                spectrum[i] += 10 * self.energy[i]**2 -  20 * self.energy[i] + 6
            if self.energy[i] < comptonErg[1]:
                spectrum[i] += 10 * (self.energy[i] - comptonErg[0])**2 -  5 * (self.energy[i] - comptonErg[0])+ 3

        return(spectrum)

class ResolutionFitter:
    def __init__(self , lines ,  **kwargs):
        for key , value in kwargs.items():
            setattr(self , key  , value)

        self.lines = lines

    def kernelFWHM(self , erg , a):
        if erg > 0:
            return( a )
        else:
            return 0.1

    def gaussianKernel(self , erg ,  a  , shift):
        gauss = np.exp( -(self.energy - shift)**2 * (1.6651092223) / (self.kernelFWHM(erg , a ) ) )
        return(gauss / np.sum(gauss) )

    def simpleKernel(self ,  a  , shift):
        gauss = np.exp( -(self.energy - shift)**2 * (1.6651092223) / a )
        return(gauss / np.sum(gauss) )

    def convolveWithGaussian(self , x , a , d , shift):
        convolved = np.zeros(1)
        n = len(self.energy)
        for i , dval in enumerate(x):
            shiftGauss = np.append(  np.zeros(len(self.energy)) , self.gaussianKernel(self.energy[i] , a  , shift) )
            convolved = d * np.append(convolved ,  np.dot( shiftGauss[n - i:2*n - i] ,  x ) )

        return( convolved[0:n] )

    def simpleConvolve(self, x , a , d , shift):
      conv = d*np.convolve( x , self.simpleKernel(a , shift))
      print(len(conv))
      print(conv)
      return(conv)

    def runFit(self , simulated , experimental , a0 , d0 , shift0 , key):
        popt , pcov = fit(self.convolveWithGaussian , simulated, experimental, p0 = (a0 , d0 , shift0), bounds = ((0 , 0 , 0) , (10 , 10 ,1) ))
        broadened = self.convolveWithGaussian(simulated , *popt)

        print(key + ":  fwhm= " , popt[0]  , "+/- " , pcov[0][0] ," ",
              ", vertical shift= " , popt[1]  , "+/- " , pcov[1][1] ," ",
              ", horizontal shift= " , popt[2]  , "+/- " , pcov[2][2] ," ")

        return(broadened , popt , pcov)

    def runAll(self , detectors):
        for key , value in detectors.items():
            a , d , shift = 0.001 , 0.8 , 0.1
            value.interpolate(lines)
            self.energy = value.expEnergy
         #   self.runLMfit(value.simSpec , value.expSpec, [a , b , c])
            broadened , (a,d,shift) , covariance =  self.runFit(value.simSpec , value.expSpec, a , d , shift, key)
            visualizeConvolution(value.expEnergy , value.expSpec , value.simSpec , broadened , key)
            #plotRes(value.expEnergy , a , b ,c  , covariance , key)
            #output to a file like 'detector: a +/- stdev, b +/- stddev, c +/- stddev'

        return(detectors)


class Detector:
    def __init__(self , simFilename  , key):
        self.simFilename = simFilename
        self.name = key

    def setExpFilename(self , expFilename):
        self.expFilename = expFilename

    def setExpSpec(self  , expSpec):
        self.expSpec = np.array(expSpec)

    def setExpEnergy(self  , energy):
        self.expEnergy = np.array(energy)[1:]

    def setSimSpec(self , energy , simSpec):
        self.simSpec   = np.array(simSpec)
        self.simEnergy = np.array(energy)

    def setChannels(self , channels):
        self.channels = channels

    def interp(self , lines , bounds=[0.2 , 0.7]):
        self.simSpec = self.simSpec / max(self.simSpec)
        self.expSpec = self.expSpec / max(self.expSpec)
        plt.plot(self.simEnergy , self.simSpec)
        plt.plot(self.expEnergy , self.expSpec)
        plt.ylabel("Counts")
        plt.title(self.name)
        plt.legend()
        plt.show()


    def interpolate(self , lines):
        self.simSpec = self.simSpec / max(self.simSpec)
        self.expSpec = self.expSpec / max(self.expSpec)

        #plt.plot(self.simEnergy , self.simSpec)
        #plt.plot(self.expEnergy , self.expSpec)
        #plt.ylabel("Counts")
        #plt.title(self.name)
        #plt.legend()
        ##plt.show()
        # we can only go from the largest min energy bin to the smallest max energy bin

        #elow = min( self.expEnergy[0] , self.simEnergy[0])
        elow = 0.2
        ehigh = 0.7
        #ehigh = min( self.expEnergy[-1] , self.simEnergy[-1])

        lowExpInd = np.where( np.abs( self.expEnergy - elow) ==  np.abs( self.expEnergy - elow ).min() )
        lowSimInd = np.where( np.abs( self.simEnergy - elow) ==  np.abs( self.simEnergy - elow ).min() )
        highExpInd = np.where( np.abs( self.expEnergy - ehigh) ==  np.abs( self.expEnergy - ehigh ).min() )
        highSimInd = np.where( np.abs( self.simEnergy - ehigh) ==  np.abs( self.simEnergy - ehigh ).min() )

        self.expEnergy = self.expEnergy[lowExpInd[0][0]:highExpInd[0][0]]
        self.expSpec   = self.expSpec[lowExpInd[0][0]:highExpInd[0][0]]
        self.simEnergy = self.simEnergy[lowSimInd[0][0]:highSimInd[0][0]]
        self.simSpec   = self.simSpec[lowSimInd[0][0]:highSimInd[0][0]]

        if self.simEnergy[0] >= self.expEnergy[-1] or self.simEnergy[-1] <= self.simEnergy[0]:
            print("Simulation and experiment not within the same energy range! Go run another simulation with the correct energy range")
            sys.exit()

        for line in lines:
            if line > ehigh or line < elow:
                print("Expected gamma lines are outside of the range of the experimental and simulated energy binning! How did you manage that?")
                sys.exit()

        # match the experimental and simulated energy bining by interpolating the simulated spectrum
        simSpecNew = np.zeros(len(self.expEnergy))
        for i , en in enumerate(self.expEnergy):
            if en > elow:
                matchIndices = np.where( np.abs( self.simEnergy - en) ==  np.abs( self.simEnergy - en ).min() )
                simSpecNew[i] = self.simSpec[ matchIndices[0] ]

        # cut the spectra in the valley before the first compton edge
        plt.plot(self.simEnergy , self.simSpec , label='Simulation' , color='g')
        plt.plot(self.expEnergy , simSpecNew   , label='Interpolated Simulation' , color='b')
        plt.plot(self.expEnergy , self.expSpec , label='Experimental', color='r')

        plt.xlabel("Light Output [MeV]")
        plt.ylabel("Counts")
        plt.title(self.name)
        plt.legend()
        plt.show()

        self.simSpec    = simSpecNew
        self.simEnergy  = self.expEnergy

    def setValues(a , b, c , pcov):
        self.a = a
        self.b = b
        self.c = c
        self.covariance = pcov

def test():
    energy   = np.linspace(0 , 2 , num=1000)

    exp = Experiment(energy)
    sim = Simulation(energy)
    res = ResolutionFitter(energy , loud=True , lines=2)

    experimental = exp.makeTestSpectrum([ 0.511 , 1.275 ] , 0.04 , 0.015 , 8)
    simulation   = sim.makeTestSpectrum([ 0.511 , 1.275 ] )
    energy , experimental , simulation , broadened , a , b , c , pcov = res.runFit( experimental , simulation , 0.04 , 0.015 , 8)
    visualizeConvolution( energy , experimental , simulation , broadened )
    plotRes(energy , a , b , c , pcov)

def plotRes(energy , a , b , c , pcov , title):
    y = a + b*np.sqrt(energy + c*energy**2)
    error = np.sqrt( pcov[0][0]**2 + (energy + c*energy**2) * pcov[1][1]**2 + energy**3 * b**2  / (4 * energy * c + 1) * pcov[2][2] )
    plt.plot(energy , y  , "r--" , label="fitting result")
    plt.fill_between(energy , y-error, y+error , label="error")
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('FWHM [MeV]')
    plt.title(title)
    plt.show()


def visualizeConvolution(energy , experimental , simulation , broadened , title):
    plt.plot(energy , experimental  , label="Experimental"  , color='r')
    plt.plot(energy , simulation    , label="Simulation" , color='b')
    plt.plot(energy , broadened     , label="Broadened" , color='k')
   # plt.plot(np.linspace(0 , max(energy) , num=len(convolved)) , convolved , label="convolved")
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Frequency [Arbitrary Units]')
    plt.title(title)
    plt.show()

def booleanize(value):
    """Return value as a boolean."""
    true_values = ("yes", "true", "1")
    false_values = ("no", "false", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_values:
        return True
    elif value.lower() in false_values:
        return False
    raise TypeError("Cannot booleanize ambiguous value '%s'" % value)

if __name__ == '__main__':
    fi     = sys.argv[1]
    conf = configparser.ConfigParser()
    conf.read(fi)

    # set up dict of detectors with name as key and object as value, constructing with simulation data
    detectors = {}
    keys = []

    # set up simulated spectra read in
    if 'Sim Data' in conf:
        for (key , value) in conf.items('Sim Data'):
            keys.append(key)
            detectors[key] = Detector(value , key)
    else:
        print("No Sim Data section in config file! exiting.")
        sys.exit()

    # set up experimental spectra read in
    if 'Spec Data' in conf:
        for (key , value) in conf.items('Spec Data'):
            if key in detectors:
                detectors[key].setExpFilename(value)
            else:
                print("Could not find detector " + key + " in [Sim Data]")
                print("Detector names must match")
                sys.exit()
    else:
        print("No Spec Data section in config file! Can't use spectral data")
        sys.exit()

    # general setup
    if 'General' in conf:
        write    = booleanize( conf['General']['write'].strip() )
        loudin   = booleanize( conf['General']['loud'].strip()  )
        numBins  = int(conf['General']['bin'])
        lines    = [ float(x.strip()) for x in conf['General']['lines'].split(",") ]

        if 'raw_data_conf' in conf['General']:
            waveConfig = conf['General']['raw_data_conf']
            # set up detector mapping, set detector channels
            if 'Detector Mapping' in conf:
                for (key , value) in conf.items('Detector Mapping'):
                    channels = [ int(x.strip()) for x in value.split(",") ]
                    if key in detectors:
                        detectors[key].setChannels(channels)
                    else:
                        print("Could not find detector " + key + " in [Sim Data]")
                        print("Detector names must match")
                        sys.exit()
            else:
                print("No Detector Mapping section in config file! Can't use raw wave data.")
                sys.exit()

            exp = Experiment(lines , bins=numBins , loud=loudin , readSpec=False , WaveConfig=waveConfig, writeSpec=write)
        else:
            exp = Experiment(lines , bins=numBins , loud=loudin , readSpec=True , writeSpec=write)
    else:
        print("No General section in config file! exiting.")
        sys.exit()

    detectors  = exp.getData(detectors , keys)

    for key , value in detectors.items():
        with open(value.simFilename , "r") as simin:
            bins = []
            hist = []
            for line in simin.readlines():
                b , h  = [ float(x.strip()) for x in line.split(" , ") ]
                bins.append(b)
                hist.append(h)
            value.setSimSpec(bins , hist)

    res = ResolutionFitter(lines)
    detectors = res.runAll(detectors)


