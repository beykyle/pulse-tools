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

pywaves_directory = "../low_level/"

##############################################

sys.path.extend([pywaves_directory])
sys.path.extend("./")


from pulseTemplates import readAndAveragePulses , writePulse
from compPulse import compPulses
from dataloader  import DataLoader
from getwavedata import GetWaveData
from waveform    import Waveform
import dataloader
import getwavedata

def booleanize(value):
    """Return value as a boolean."""
    true_values = ("yes", "true", "1" , "y" , "Yes" , "Y")
    false_values = ("no", "false", "0" , "n" , "N" , "No")
    if isinstance(value, bool):
        return value
    if value.lower() in true_values:
        return True
    elif value.lower() in false_values:
        return False
    raise TypeError("Cannot booleanize ambiguous value '%s'" % value)

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

def getPulsesFromBox(waves , tailLim , totLim  ):

  tailLow  = min(tailLim)
  tailHigh = max(tailLim)
  totLow   = min(totLim)
  totHigh  = max(totLim)

  goodWaves = []
  numpulses = float( input("How many pulses from this region do you want to plot?  ") )

  j = 0
  #for  wave in goodWaves:
  for wave in waves:
    if( wave.total >= totLow and wave.total <= totHigh and wave.ratio >= tailLow and wave.ratio <= tailHigh ):
      goodWaves.append(wave)
      #plot the pulse
      if j < numpulses:
        j = j + 1
        x = range(0,len(wave.blsSamples))
        y = wave.blsSamples
        plt.plot(x,y)
        plt.xlabel("Time")
        plt.ylabel("Pulse Height")
        plt.show()
        plt.close()

  print("Got all the waves in the current region")
  return(goodWaves)

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
      for sample in pulse.blsSamples:
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

  print("Welcome to psdPulseSelect.py! \r\n\r\n")

  ##############################################
  loud                = False
  dataFile            = sys.argv[1]
  dataType            = DataLoader.DAFCA_STD
  nWavesPerLoad       = 1000
  Num_Samples         = 160
  dynamic_range_volts = 0.5
  number_of_bits      = 15
  VperLSB             = dynamic_range_volts/(2**number_of_bits)
  ns_per_sample       = 2
  tailIntegralStart   = 20
  integralEnd         = 100
  totalIntegralStart  = -3
  polarity            = -1

  ##############################################
  total  = []
  ratio  = []

  nwaves_ = int(sys.argv[2])
  datloader = DataLoader( dataFile , dataType , Num_Samples )
  nwaves    = int(datloader.GetNumberOfWavesInFile())
  print( "Found " + str(nwaves) + " pulses in the file. Reading up to " + str( int( nwaves_ ) )+ " ...")
  if nwaves < nwaves_:
    nwaves_ = nwaves

  Waves = datloader.LoadWaves( nwaves_ )

  if loud == True:
    plt.ion()

  j = 0
  pulses = []
  for wave in Waves:
    wave = Waveform(wave['Samples'] , polarity, 0 , 3)
    wave.BaselineSubtract()
    tail       = wave.GetIntegralFromPeak(tailIntegralStart  , integralEnd) * VperLSB * ns_per_sample
    wave.total = wave.GetIntegralFromPeak(totalIntegralStart , integralEnd) * VperLSB * ns_per_sample
    wave.ratio = tail / (wave.total +0.000001)
    if wave.ratio >= 0 and wave.ratio < 1 and wave.total >=0 and wave.total < 30:# and wave.isDouble(False) == False:
      total.append(wave.total)
      ratio.append(wave.ratio)
      pulses.append(wave)
      if loud == True:
        if np.mod(j,55) == 0:
          plt.cla()
          wave.isDouble(True)
          plt.plot(range(0,160) , pulses[-1].blsSamples)
          plt.draw()
          plt.pause(0.05)

  if loud == True:
    plt.ioff()
    plt.close()

  again = True
  avPulses = []
  regions  = []
  while again == True:
    xlim , ylim = scatterDensity(total , ratio , ["Total Integral [V ns]" , "Tail/Total"] )
    xmin , xmax = min(xlim) , max(xlim)
    ymin , ymax = min(ylim) , max(ylim)

    xlim = [xmin , xmax]
    ylim = [ymin , ymax]
    print("Selected area: Total: [" + str(xmin) + " , " + str(xmax) + "] V ns" + ", Ratio: [" + str(ymin) + " , " + str(ymax) + "]" )
    region_pulses = getPulsesFromBox(pulses , ylim , xlim )
    region_name = str(raw_input("What would you like to call this region? "))
    regions.append(region_name)

    writem      = booleanize(raw_input("would you like to write all the pulses in "  + region_name+ " to " + region_name + "_pulses.out? [y/n] "))
    getTemplate = booleanize(raw_input("would you like to generate a template pulse for the region? [y/n] "))
    if writem == True:
      writePulses(region_pulses)
    if getTemplate == True:
      avpulse = np.array( readAndAveragePulses( pulses=[p.blsSamples for p in region_pulses] ) * 2 /( 2**( number_of_bits) - 1) )
      avPulses.append(avpulse)
      out = region_name + "template.out"
      writePulse(avpulse , out , 1 , xlim , ylim)

    again = booleanize(raw_input("Would you like to select another region? [y/n]"))
    print("\r\n")

  if regions != [] and avPulses != []:
    print("\r\n Comparing pulse templates in each region")
    compPulses( avPulses , regions , ns_per_sample )


