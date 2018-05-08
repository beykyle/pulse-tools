#!/usr/bin/env python

"""
psdScatter.py: a method to create a scatter density plot of tail/total integral
vs. total integral from dafca output, and selectively view pulses from regions
of the plot

"""
import numpy as np
import sys
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.widgets  import RectangleSelector
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import configparser

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

def getPulsesFromBox(waves , ratioLim , totLim  ):
  ratLow  = min(ratioLim)
  ratHigh = max(ratioLim)
  totLow   = min(totLim)
  totHigh  = max(totLim)

  goodWaves = []
  numpulses = tryIntInput("How many pulses from this region do you want to plot?  ")

  j = 0
  #for  wave in goodWaves:
  for wave in waves:
    if( wave.total >= totLow and wave.total <= totHigh and wave.ratio >= ratLow and wave.ratio <= ratHigh ):
      goodWaves.append(wave)
      #plot the pulse
      if j < numpulses:
        print("plotting pulse with total: " + str(wave.total) +  " and ratio: " + str(wave.ratio) )
        j = j + 1
        x = range(0,len(wave.blsSamples))
        y = wave.blsSamples
        plt.plot(x,y)
        plt.xlabel("Time")
        plt.ylabel("Pulse Height")
        plt.show()
        plt.close()

  print("Got all " + str(len(goodWaves)) + " waves in the current region")
  return(goodWaves)

def readGoodInd(fname):
  with open(fname , "r") as g:
    lines = g.readlines()

  good = []
  for line in lines:
    good.append(int( line.rstrip("\r\n") ))

  return(good)

def writePulses(pulses , region_name):
  with open( outPath + region_name + "_pulses.out" , "w") as out:
    for pulse in pulses:
      for sample in pulse.blsSamples:
        out.write('{:1.5f}'.format(sample) + ',')
      out.write("\r\n")

def writeTiming(pulses , region_name):
  with open( outPath + region_name + "_timing.out" , "w") as out:
    for pulse in pulses:
      out.write(str(pulse.ch) + ',' + '{:1.8E}'.format(pulse.time) + "\r\n" )

def writeShaping(std , kurt , region_name):
  with open( outPath + region_name + "_std.out" , "w") as out:
    for s in std:
      out.write(str(s) +"\r\n" )
  with open( outPath + region_name + "_kurt.out" , "w") as out:
    for k in kurt:
      out.write(str(k) +"\r\n" )

def findDoubleFrac(pulses):
  doubles = 0
  goods   = 0
  for pulse in pulses:
    if pulse.isDouble(False) == True:
      doubles += 1
    else:
      goods += 1

  return(doubles / (goods + doubles))

def getPulseWidth(pulses):
  std  = []
  kurt = []
  for pulse in pulses:
    waveform = pulse.blsSamples / max(pulse.blsSamples)
    std.append(  np.std(waveform)         )
    kurt.append( stats.kurtosis(waveform) )
  return(std,kurt)

def plotPulseWidths(kurt , std , region_name):
  plt.figure()
  plt.subplot(121)
  plt.hist(std , 20)
  plt.xlabel("pulse standard deviation [V]")
  plt.ylabel("counts")
  plt.subplot(122)
  plt.hist(kurt , 20)
  plt.xlabel(r"pulse kurtosis")
  plt.ylabel("counts")
  plt.tight_layout()
  plt.savefig(region_name + "_shapes.png" , dpi=500)
  plt.show()
  plt.close()


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

def tryStrInput(string):
  gotIt = False
  while gotIt == False:
    try:
      var = str(raw_input(string))
      gotIt = True
    except ValueError:
      print("Input could not be interpreted! Try again.")

  return(var)

def tryIntInput(string):
  gotIt = False
  while gotIt == False:
    try:
      var = int(raw_input(string))
      gotIt = True
    except ValueError:
      print("Input could not be interpreted! Try again.")

  return(var)

def tryFloatInput(string):
  gotIt = False
  while gotIt == False:
    try:
      var = float(raw_input(string))
      gotIt = True
    except ValueError:
      print("Input could not be interpreted! Try again.")

  return(var)

def tryBoolInput(string):
  gotIt = False
  while gotIt == False:
    try:
      var = booleanize(raw_input(string))
      gotIt = True
    except ValueError:
      print("Input could not be interpreted! Try again.")

  return(var)



if __name__ == '__main__':
  print("Welcome to psdPulseSelect.py! \r\n\r\n")

  configFileName = sys.argv[1]
  config = configparser.ConfigParser()
  config.read(configFileName)

  global outPath
  outPath             = config['Directories']['output_path']
  dynamic_range_volts = float(config['Digitizer']['dynamic_range_volts'])
  number_of_bits      = int(config['Digitizer']['number_of_bits'])
  ns_per_sample       = int(config['Digitizer']['ns_per_sample'])
  integralEnd         = int(config['Pulse Processing']['integral_end'])
  totalIntegralStart  = int(config['Pulse Processing']['total_integral_start'])
  tailIntegralStart   = int(config['Pulse Processing']['tail_integral_start'])
  VperLSB =  dynamic_range_volts / ( 2**( number_of_bits) - 1)

  if len(sys.argv) > 2:
    loud = booleanize(sys.argv[2])
    chCount, tailInt, totalInt, pulses = GetWaveData(configFileName , getWaves=True , loud=loud)
  else:
    chCount, tailInt, totalInt, pulses = GetWaveData(configFileName , getWaves=True , loud=False)

  total      = []
  ratio      = []
  goodPulses = []
  tail     = tailInt[0]
  oldtotal = totalInt[0]

  print(len(tail))
  print(len(pulses))

  for pulse in pulses:
    tot  =  pulse.GetIntegralFromPeak(totalIntegralStart , integralEnd)*VperLSB*ns_per_sample
    tail =  pulse.GetIntegralFromPeak(tailIntegralStart  , integralEnd)*VperLSB*ns_per_sample
    if tail > 0  and tot > 0:
      pulse.total = tot
      pulse.ratio = tail / (0.000001 + tot)
      goodPulses.append(pulse)
      ratio.append(pulse.ratio )
      total.append(tot)

  again = True
  avPulses = []
  regions  = []

  print("Found " + str(len(pulses)) + " good pulses!"  )

  # look at a region and get it's info
  while again == True:
    xlim , ylim = scatterDensity(total , ratio , ["Total Integral [V ns]" , "Tail/Total"] )
    xmin , xmax = min(xlim) , max(xlim)
    ymin , ymax = min(ylim) , max(ylim)

    totLim = [xmin , xmax]
    ratioLim = [ymin , ymax]
    print("Selected area: Total: [" + str(totLim[0]) + " , " + str(totLim[1]) + "] V ns"
                     + ", Ratio: [" + str(ratioLim[0]) + " , " + str(ratioLim[1]) + "]"
         )

    region_pulses = getPulsesFromBox( goodPulses , ratioLim , totLim )
    region_name = tryStrInput("What would you like to call this region? ")
    regions.append(region_name)

   # writem      = booleanize(raw_input("would you like to write all the pulses in "
  #                           + region_name+ " to " + region_name + "_pulses.out? [y/n] ")
 #                           )

   # timing      = booleanize(raw_input("would you like to write timing data and channel in "
#                             + region_name+ " to " + region_name + "_timing.out? [y/n] ")
    #                        )

    getTemplate   = tryBoolInput("would you like to generate a template pulse for the region? [y/n] ")

    getDoubleFrac = tryBoolInput("would you like to calculate the double pulse fraction for the region? [y/n] ")
    if getDoubleFrac == True:
      frac = findDoubleFrac(region_pulses)
      print("Double fraction for " + region_name + ": " + str(frac) )
      shaping      = tryBoolInput("would you like to write pulse shape data in "
                            + region_name+ " to " + outPath + region_name + "_std.out and " + region_name + "_kurt.out? [y/n]" )

      shapfig      = tryBoolInput("would you like to save pulse shape figs in "
                             + region_name+ " to " + outPath + region_name + "_std.png and " + outPath + region_name + "_kurt.png? [y/n]" )

      if shaping == True:
        regionstd , regionkurt = getPulseWidth(region_pulses)
        writeShaping(regionstd , regionkurt , region_name)

      if shapfig == True:
        regionstd , regionkurt = getPulseWidth(region_pulses)
        plotPulseWidths(regionstd , regionkurt , region_name)

    #if writem == True:
    #  writePulses(region_pulses , region_name)

    #if timing == True:
    #  writeTiming(region_pulses , region_name)

    if getTemplate == True:
      avpulse = np.array( readAndAveragePulses( pulses=[p.blsSamples for p in region_pulses] )
                          * VperLSB * ns_per_sample
                        )

      avPulses.append(avpulse)
      out = outPath + region_name + "template.out"
      writePulse(avpulse , out , 1 , totLim , ratioLim)

    again = tryBoolInput("Would you like to select another region? [y/n]")
    print("\r\n")

  if regions != [] and avPulses != []:
    print("\r\n Comparing pulse templates in each region")
    compPulses( avPulses , regions , ns_per_sample )
