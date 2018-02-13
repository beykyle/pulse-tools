#!/usr/bin/env python

"""
rescon.py: Code to find the energy resolution of an organic scintillator by iteratively fitting a Gaussian convolution to un-broadened spectra simulated in Monte Carlo until it fits a measured gamma spectra.

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector
from matplotlib.patches import Rectangle
import waveform
import dataloader
import getwavedata
from scipy.ndimage.filters import gaussian_filter1d as gfilter
from scipy.optimize import curve_fit as fit
from scipy.signal import savgol_filter as savgol
from scipy.signal import find_peaks_cwt as peaks

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

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

def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    ax.add_patch(rect)

class DraggableRectangle:
    lock = None  # only one can be animated at a time
    def __init__(self ):
        self.rect = Rectangle((0,0) ,1 , 1 )
        self.press = None
        self.background = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return
        if DraggableRectangle.lock is not None: return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        print('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata
        DraggableRectangle.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggableRectangle.lock is not self:
            return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableRectangle.lock is not self:
            return

        self.press = None
        DraggableRectangle.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

class Experiment:
    def __init__(self , lines , bins, *args , **kwargs):
        if kwargs is not None:
                for key, value in kwargs.items():
                    if key == "readSpec":
                        self.readSpec = booleanize(value)

        self.bins  = int(bins)
        self.lines = lines

    def run(self , config):
        if(hasattr(self , 'readSpec')):
            if(self.readSpec == True):
                spec = []
                volts = []
                #iterate through every file with spec*.tmp
                #read each channel of spec and volts
        else:
            volts , spec = self.readFromFile(config)
            self.writeSpec(volts , spec)

        edges = []
        energy = []
        for i , (bins , hist) in enumerate(zip(volts , spec)):
            edges.append(self.findComptonEdges(bins , hist))
            energy.append(bins * 0.478 / edges[i])

        return(energy , spec)

    def writeSpec(self , volts, spec):
        print("writing spectrum to spec.tmp, os if something goes wrong you don't have to read through the waves again.")
        for i , (bins , hist) in enumerate(zip(volts , spec)):
            fname =  "spec_" + str(i)+"_.tmp"
            np.savetxt(fname, np.c_[volts , spec] , delimiter="," , newline="\r\n" , fmt='%1.6e')


    def readFromFile(self , config ):
        spec = []
        volts = []
        chCount, ph, amp, tailInt, totalInt, cfd, ttt, extras, fullTime, flags, rms = getwavedata.GetWaveData(config)
        self.numChannels = len(chCount)
        print("\n \n Read pulse integrated spectra from " + str(self.numChannels) + " channels. \n")

        for chan , bar in zip( range(0 , len(chCount) , 2) , [1,2,3,4,5,6,7,8] ):
             hist , bins = np.histogram(totalInt[chan+1][:chCount[chan+1]] + totalInt[chan][:chCount[chan]]  , bins=self.bins )
             spec.append(hist)
             volts.append(bins)
             #plt.hist( totalInt[chan][:chCount[chan]] , bins=self.bins, label='channel: '+ str(chan) , histtype='step')

        print("calibrating spectrum energy lines: " +  ' , '.join(str(e) for e in self.lines) + " MeV")
        return(volts , spec)


    def findComptonEdges(self , bins , hist):
        print("Finding Compton Edges")
        ce = [1]
        nhist = savgol( (hist / hist.max())   , 11 , 3 )
        diff1 = savgol(  np.diff(nhist)       , 7  , 3 )
        diff2 = savgol(  np.diff(diff1)       , 7  , 3 )
        #k  = np.where( diff1 == diff1.min() )
        #k2 = np.where( diff2 == diff2.min() )

        peakid  = peaks( nhist , np.arange(1,10) )
        peakid2 = peaks( diff2 , np.arange(1,10) )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot( bins[1:]   , nhist            , label="spectrum"       )
        ax.plot( bins[1:-2] , diff2            , label="2nd derivative" )
        for pk , pk2 in zip(peakid , peakid2):
            ax.plot( bins[pk]    , nhist[pk-1]  , '*b' , label="peak"   )
            ax.plot( bins[pk2]   , diff2[pk2+1] , '*g' , label="inflection point"   )

        plt.legend()

        for line in self.lines:
            print("Looking for " + str(line) + " MeV Compton edge.")

        rs = RectangleSelector(ax, line_select_callback,
                       drawtype='box', useblit=False, button=[1],
                       minspanx=5, minspany=5, spancoords='pixels',
                       interactive=True)

        plt.show()

        return(ce)


    def makeTestSpectrum(self , comptonErg , a , b , c):
        spectrum = np.zeros( len(self.energy) )
        for i , ch in enumerate(spectrum):
            spectrum[i] += 0.6*np.random.rand()
            if self.energy[i] < comptonErg[0]:
                spectrum[i] += 20 * self.energy[i]**2 -  10 * self.energy[i] + 6
            if self.energy[i] < comptonErg[1]:
                spectrum[i] += 10 * (self.energy[i] - comptonErg[0])**2 -  5 * (self.energy[i] - comptonErg[0]) + 3

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
                spectrum[i] += 20 * self.energy[i]**2 -  10 * self.energy[i] + 6
            if self.energy[i] < comptonErg[1]:
                spectrum[i] += 10 * (self.energy[i] - comptonErg[0])**2 -  5 * (self.energy[i] - comptonErg[0])+ 3

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
        if self.lines == 1:
            experimental , simulated = self.cutBaseline(experimental , simulated)

        popt , pcov = fit(self.convolveWithGaussian , simulated , experimental ,
                            p0 = (a0 , b0 , c0) ,
                            bounds = ( (0 , 0 , 0 ) , (1 , 1 , np.inf) )
                          )

        broadened = self.convolveWithGaussian(simulated , *popt)

        print("  a= " , popt[0]  , "+/- " , pcov[0][0] ," ",
              ", b= " , popt[1]  , "+/- " , pcov[1][1] ," ",
              ", c= " , popt[2]  , "+/- " , pcov[2][2] ," ")

        return(self.energy , experimental , simulated , broadened , popt[0] , popt[1] , popt[2] , pcov)

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

def plotRes(energy , a , b , c , pcov):
    y = a + b*np.sqrt(energy + c*energy**2)
    error = np.sqrt( pcov[0][0]**2 + (energy + c*energy**2) * pcov[1][1]**2 + energy**3 * b**2  / (4 * energy * c + 1) * pcov[2][2] )
    plt.plot(energy , y  , "r--" , label="fitting result")
    plt.fill_between(energy , y-error, y+error , label="error")
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('FWHM [MeV]')
    plt.show()


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
    fi     = sys.argv[1]
    bins   = sys.argv[2]
    if(len(sys.argv) > 3):
        lines = [float(x) for x in sys.argv[3:]]
    else:
        lines = float(input("Enter gamma line in MeV: "))

    exp = Experiment(lines , bins)
    energy , spec = exp.run(fi)



