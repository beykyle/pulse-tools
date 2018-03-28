#usr/bin/env python

"""
doublePulseClassifier.py: a code that classifies if a waveform from a scintillation detector has pileup using a simple neural net

"""

from dataloader  import DataLoader
from getwavedata import GetWaveData
from waveform    import Waveform

from keras.models import Sequential
from keras.layers import Dense
from matplotlib   import pyplot as plt

import sys
import numpy as np
np.random.seed(7) # fix random seed for reproducibility

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

# initialize global detector variables because im too lazy to use a config file
##############################################

pywaves_directory = "./"

##############################################

sys.path.extend([pywaves_directory])

class Pulse:
  def __init__(self , samples, isDouble=False):
    self.samples = samples
    self.isDouble = isDouble



def plotPulse(samples , t , v):
  plt.plot( range(0  , len(samples)) * t , samples * v  )
  plt.xlabel("Time [ns]")
  plt.ylabel("Pulse Height [V]" )
  plt.show()
  return( input( "Is this a double pulse"))

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
  tailIntegralStart   = 20
  integralEnd         = 100
  totalIntegralStart  = -3
  polarity            = -1

##############################################


  datloader = DataLoader( dataFile , dataType , Num_Samples )
  nwaves    = datloader.GetNumberOfWavesInFile()
  print( "Found " + str(nwaves) + "pulses in the file. Reading up to 1E5...")
  if nwaves < 1E5:
    Waves = datloader.LoadWaves(nwaves)
  else:
    Waves = datloader.LoadWaves( int(10) )

  single_pulses = []
  double_pulses = []
  for w in Waves:
    wave = Waveform(w['Samples'] , polarity , 1 , 5 )
    wave.BaselineSubtract()
    pulse = Pulse( wave.blsSamples , False)
    if wave.isDouble() == False:
      if plotPulse( wave.blsSamples, ns_per_sample , VperLSB) == False: # verify single pulses with a plot
        single_pulses.append( pulse )

    elif wave.isDouble() == True:
      if plotPulse( wave.blsSamples , ns_per_sample , VperLSB) == True: # verify double pulses with a plot
        double_pulses.append( pulse )




