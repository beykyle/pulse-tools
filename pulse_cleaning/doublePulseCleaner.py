#!/usr/bin/env python

import numpy as np

class pulse:
  def __init__(self, samples ):
    self.samples = samples
    self.height  = max(samples)
    self.maxInd  = np.where(samples == self.height)
    self.total   = sum(samples)
    self.ratio   = sum(samples[]) / self.total
    print(self.maxInd)


  def checkIfSingle():
    delta



