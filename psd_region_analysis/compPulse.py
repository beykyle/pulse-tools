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


neutfi = "av_neutron_235.out"
gamfi = "av_gamma_235.out"
aphi = "av_alpha_235.out"

time = np.linspace(0,160,160)*2

ne = read(neutfi)
ga = read(gamfi)
ap = read(aphi)

plt.semilogy(time,ne , 'b' , label='neutron')
plt.semilogy(time,ga , 'r' , label='gamma')
plt.semilogy(time,ap , 'g' , label='alpha')
plt.xlabel("Time [ns]")
plt.legend()
plt.show()



