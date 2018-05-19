#! usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt

def read(fi):
  a = []
  with open(fi , "r") as f:
    l = f.readlines()

  for lin in l:
    a.append( float(lin.strip().rstrip("\r\n")))

  return(np.array(a))

def compPulses(pulses , regionNames , ns_per_sample):
  time = np.linspace(0,len(pulses[0]),len(pulses[0]) )*ns_per_sample
  for region , pulse in zip(regionNames , pulses):
    #plt.semilogy(time, pulse / max(pulse) , label=str(region))
    plt.semilogy(time, pulse , label=str(region))
    #plt.plot(time, pulse , label=str(region))

  plt.xlabel("Time [ns]")
  plt.ylabel("Signal [V]")
  plt.legend()
  plt.show()

if __name__ == '__main__':
  neutfi = "av_neutron.out"
  gamfi = "av_gamma.out"
  aphi = "av_alpha.out"

  time = np.linspace(0,160,160)*2

  ne = read(neutfi)
  ga = read(gamfi)
  ap = read(aphi)

  compPulses( [ne , ga , ap]  ,['neutron' , 'gamma' , 'alpha'] , 2)


