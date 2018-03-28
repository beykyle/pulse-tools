#!/usr/bin/env python

"""
psdScatter.py: a method to create a scatter density plot of tail/total integral
vs. total integral from dafca output, and selectively view pulses from regions
of the plot

"""

import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.widgets  import RectangleSelector
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"


##############################################

pywaves_directory = "./"

##############################################

sys.path.extend([pywaves_directory])

from dataloader  import DataLoader
from getwavedata import GetWaveData
from waveform    import Waveform

class Annotate(object):
  def __init__(self):
    self.ax = plt.gca()
    self.rect = Rectangle((0,0), 0, 0)
    self.x0 = 0
    self.y0 = 0
    self.x1 = 0
    self.y1 = 0
    self.ax.add_patch(self.rect)
    self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
    self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

  def on_press(self, event):
    self.x0 = event.xdata
    self.y0 = event.ydata

  def on_release(self, event):
    self.x1 = event.xdata
    self.y1 = event.ydata
    self.rect.set_width(self.x1 - self.x0)
    self.rect.set_height(self.y1 - self.y0)
    self.rect.set_xy((self.x0, self.y0))
    self.ax.figure.canvas.draw()

def CFD(Blnd_Wave, F):
	CFD_Val = F*max(Blnd_Wave)
	counter = np.argmax(Blnd_Wave)

	while Blnd_Wave[counter] > CFD_Val:
		counter -= 1

	Time_Range = range(0,880,2)
	y_2 = Blnd_Wave[counter+1]
	y_1 = Blnd_Wave[counter]
	y = CFD_Val
	x_1 = Time_Range[counter]
	x_2 = Time_Range[counter+1]

	Start_Time = x_1 + (y-y_1)*(x_2-x_1)/(y_2-y_1)

	return Start_Time

def readPSD(fi1):

  with open(fi1) as psdfile:
    lines = psdfile.readlines()

  total = []
  tail = []
  ph = []
  ratio = []

  for line in lines:
    line = line.rstrip("\r\n")
    line = line.split(" ")
    line = list(filter(None, line))
    ph.append(float(line[0]))
    total.append(float(line[1]))
    ratio.append(float(line[2]))
    tail.append(float(line[3]))

  return(total , tail , ph , ratio)

def readPulses(config):
  tail = []
  total = []
  ratio = []
  chCount, ph, amp, tailInt, totalInt, cfd, ttt, extras, fullTime, flags, rms = getwavedata.GetWaveData(config)
  for chan in range(len(chCount)):
    tail.append(tailInt[chan][:chCount[chan]])
    total.append(totalInt[chan][:chCount[chan]])
    ratio.append( x/y for x,y in zip( (tailInt[chan][:chCount[chan]]) , totalInt[chan][:chCount[chan]] ) )

  return(tail , total , ratio)

def getPulsesFromBox(waves , tailLim , totLim , fpath , goodInd):

  tailLow  = min(tailLim)
  tailHigh = max(tailLim)
  totLow   = min(totLim)
  totHigh  = max(totLim)

  print("Found " + str(nwaves) + " waves, with " + str(len(goodInd)) + " good ones.")

  goodWaves = []
  pulses    =  []

  print("Storing all the good pulses in the box you selected in 'pulses.out'. This may take a while. ")
  numpulses = float( input("In the meantime, how many pulses do you want to plot?  ") )

  for i in goodInd[0:nwaves - 1]:
    if i < nwaves:
      goodWaves.append(Waves[i])

  j = 0
  #for  wave in goodWaves:
  for wave in Waves:
  #print(wave)
    baseline  = np.average(wave['Samples'][0:5])
    bswave    = baseline - wave['Samples']
    maxInd    = np.where(bswave == max(bswave) )[0][0]


    total      = np.sum( bswave          )*2*VperLSB
    tail       = np.sum( bswave[maxInd:] )*2*VperLSB
    ratio = tail / total

  #  print("Total: " + str(total) + " ratio: " + str(ratio))

    if( total >= totLow and total <= totHigh and ratio >= tailLow and ratio <= tailHigh ):
      pulses.append(bswave)
      #plot the pulse
      if j < numpulses:
        j = j + 1
        x = np.arange(0,320,2)
        y = bswave
        plt.plot(x,y)
        plt.xlabel("Time [ns]")
        plt.ylabel("Pulse Integral [mV ns]")
        plt.show()
        plt.close()

  return(pulses)

def readGoodInd(fname):
  with open(fname , "r") as g:
    lines = g.readlines()

  good = []
  for line in lines:
    good.append(int( line.rstrip("\r\n") ))

  return(good)

def writePulses(pulses):
  with open("pulses.out" , "w") as out:
    for pulse in pulses:
      for sample in pulse:
        out.write('{:1.5f}'.format(sample) + ',')
      out.write("\r\n")

def scatterDensity(data1 , data2 , labels):
  def line_select_callback(eclick, erelease ):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    ax.add_patch(rect)

  fig , ax = plt.subplots()
  dat  = plt.hist2d(data1, data2, (1000, 1000), cmap=plt.cm.jet_r)
  plt.colorbar()
  plt.xlabel(labels[0])
  plt.ylabel(labels[1])

  #rs = RectangleSelector(ax, line_select_callback,
  #                   drawtype='box', useblit=False, button=[1],
  #                   minspanx=5, minspany=5, spancoords='pixels',
  #                   interactive=True)

  a = Annotate()
  #print(rs)
  print("Select a rectangle corresponding to the region of pulses you want to look at more closely.")
  print("Once you've selected the desired area, close the figure!")
  plt.show()
  return([a.x0 , a.x1] , [a.y0 , a.y1])

if __name__ == '__main__':

  ##############################################

  dataFile            = sys.argv[1]
  dataType            = DataLoader.DAFCA_DPP_MIXED
  nWavesPerLoad       = 1000
  Num_Samples         = 160
  dynamic_range_volts = 0.5
  number_of_bits      = 15
  VperLSB             = dynamic_range_volts/(2**number_of_bits)
  ns_per_sample       = 2
  tailIntegralStart   = 12
  integralEnd         = 100
  totalIntegralStart  = -3
  polarity            = -1

  ##############################################
  total  = []
  ratio  = []


  datloader = DataLoader( dataFile , dataType , Num_Samples )
  nwaves    = datloader.GetNumberOfWavesInFile()
  print( "Found " + str(nwaves) + "pulses in the file. Reading up to 1E5...")
  if nwaves < 1E4:
    Waves = datloader.LoadWaves(nwaves)
  else:
    Waves = datloader.LoadWaves( int(1E4) )

  pulses = []
  for w in Waves:
    wave = Waveform(w['Samples'] , polarity , 1 , 5 )
    tail       = wave.GetIntegralFromPeak(tailIntegralStart  , integralEnd) * VperLSB * ns_per_sample
    wave.total = wave.GetIntegralFromPeak(totalIntegralStart , integralEnd) * VperLSB * ns_per_sample
    wave.ratio = tail / wave.total
    total.append(wave.total)
    ratio.append(wave.ratio)
    pulses.append(wave)

  for pulse in pulses[1:10]:
    #plt.plot( range(0 , len(pulse.samples)) , pulse.blsSamples)
    #plt.show()
    print(pulse.total)

  #xlim , ylim = scatterDensity(total , ratio , ["Total Integral [V ns]" , "Tail/Total"] )
  #xmin , xmax = min(xlim) , max(xlim)
  #ymin , ymax = min(ylim) , max(ylim)

  #xlim = [8 , 10]
  #ylim = [ymin , ymax]
  #print("Selected area: Total: [" + str(xmin) + " , " + str(xmax) + "] V ns" + ", Ratio: [" + str(ymin) + " , " + str(ymax) + "]" )


  #xlim , ylim = [-0.02 , 10.3281931632], [-0.02 , 10]
  #writePulses(pulses)
