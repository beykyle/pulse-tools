#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt

def readAndAveragePulses(*args, **kwargs):
  if kwargs is not None:
    for key, value in kwargs.items():
      if key == "pulses":
        pulses = value
        read = False
      elif key == "filename":
        filename = value
        read = True
      else:
        raise ValueError("Either too many or unrecognized keyword arguments in readAndAvergePulses")
        sys.exit()
  else:
    raise ValueError("readAndAveragePulses needs a keyword argument! Either fname=pulse_file_name or pulses=list_of_pulses")
    sys.exit()

  if read == True:
    with open(filename , "r") as  fn:
      lines = fn.readlines()

    plt.ion()
    pulses = []
    for line in lines:
      pulses.append([float(l.strip().rstrip("\r\n") ) for l in line[:-3].split(",")])
      a = ([float(l.strip().rstrip("\r\n") ) for l in line[:-3].split(",")])
      plt.semilogy(range(0,len(a)) ,a)
      plt.draw()
      plt.pause(0.05)

  av = np.array(len(pulses[0]))
  norm = len(pulses)
  for pulse in pulses:
    av = av + np.array(pulse)

  return(av / norm)

def writePulse(pulse, fname , conversionFactor , xlim , ylim):
  with open(fname , "w" ) as out:
    out.write("Template from selected region: Total: [" + str(xlim[0]) + " , " + str(xlim[1]) + "] V ns"
        + ", Ratio: [" + str(ylim[0]) + " , " + str(ylim[1]) + "]" )
    for v in pulse:
      out.write('{:1.8E}'.format(v * conversionFactor) + "\r\n")

def plotPulse(pulse , conversionFactor , ns_per_sample):
  time = np.linspace(0,1,len(pulse)) * ns_per_sample
  plt.plot( time , pulse)
  plt.xlabel("time [ns]")
  plt.ylabel("Pulse Height [V]")
  plt.show()

if __name__ == '__main__':
  dynamic_range_volts = 0.5
  number_of_bits      = 14
  VperLSB             = dynamic_range_volts/(2**number_of_bits)

  pulse = np.array( readAndAveragePulses(filename=sys.argv[1])) * 2 /( 2**( number_of_bits) - 1)
  out = sys.argv[2]

  writePulse(pulse , out , 1)
  plotPulse(pulse , 1 , 2)
