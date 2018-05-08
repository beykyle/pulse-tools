# -*- coding: utf-8 -*-
"""

@author: Marc Ruch
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from numpy import mean, sqrt, square
from matplotlib import pyplot as plt

class Waveform:
    def __init__(self, samples, polarity, baselineOffset, nBaselineSamples , ch=0 , time=0):
        self.samples          = samples
        self.ch               = ch
        self.time             = time
        self.polarity         = polarity
        self.baselineOffset   = baselineOffset
        self.nBaselineSamples = nBaselineSamples
        self.maxIndex         = np.argmax(samples)
        self.baseline         = -1
        self.blsSamples       = -1
        self.height           = -1
        self.badPulse         = False
        self.baselined        = False
        self.total            = 0
        self.ratio            = 0

    def SetSamples(self,newSamples):
        self.samples    = newSamples
        self.maxIndex   = -1
        self.baseline   = -1
        self.blsSamples = -1
        self.badPulse   = False
        self.baselined  = False

    def BaselineSubtract(self):
        if self.polarity > 0: # Positive pulse
            self.maxIndex = np.argmax(self.samples)
            # Check for enough room to calculate baseline
            if(self.maxIndex < self.baselineOffset+self.nBaselineSamples):
                self.badPulse = True
                return
            self.baseline = np.average(self.samples[0:self.nBaselineSamples])
            self.blsSamples = self.samples - self.baseline
        else: # Negative pulse
            self.maxIndex = np.argmin(self.samples)
            # Check for enough room to calculate baseline
            if (self.maxIndex < self.baselineOffset + self.nBaselineSamples):
                self.badPulse = True
                return
            self.baseline = np.average(self.samples[:self.nBaselineSamples])
            self.blsSamples = self.baseline - self.samples
        self.baselined = True

    def ApplyCRRC4(self, samplingTime, shapingTime):
        if not self.baselined:
            self.BaselineSubtract()
        if self.badPulse:
            return -1
        T = samplingTime

        # Pre-compute
        a = 1/shapingTime
        alpha = np.exp(-T/shapingTime)
        b = np.array([0,
                      alpha*T**3*(4-a*T),
                      alpha**2*T**3*(12-11*a*T),
                      alpha**3*T**3*(-12-11*a*T),
                     alpha**4*T**3*(-4-a*T)])
        a = np.array([24, -120*alpha, 240*alpha**2,
                      -240*alpha**3, 120*alpha**4, -24*alpha**5])
        self.blsSamples = signal.lfilter(b, a, self.blsSamples)
        self.maxIndex = np.argmax(self.blsSamples)

    def GetMax(self):
        if not self.baselined:
            self.BaselineSubtract()
        if self.badPulse:
            return -1
        return  self.blsSamples[self.maxIndex]

    def GetRMSbls(self,nBaselineSamples):
        if not self.baselined:
            self.BaselineSubtract()
        if self.badPulse:
            return -1
        return  sqrt(mean(square(self.blsSamples[:nBaselineSamples])))

    def GetIntegralFromPeak(self,startIndex,endIndex):
        if not self.baselined:
            self.BaselineSubtract()
        if self.badPulse:
            return -1
        return np.sum(self.blsSamples[self.maxIndex+startIndex:self.maxIndex+endIndex]+1)

    def GetIntegralToZeroCrossing(self):
        if not self.baselined:
            self.BaselineSubtract()
        if self.badPulse:
            return -1
        negativeSamples = np.nonzero(self.blsSamples[self.maxIndex:]<0)[0]
        if len(negativeSamples)==0:
            return -1
        zeroCrossing = negativeSamples[0]+self.maxIndex
        return np.sum(self.blsSamples[self.maxIndex-self.baselineOffset:zeroCrossing])


    def GetCFDTime(self, CFDFraction, movingAverageLength=1):
        if not self.baselined:
            self.BaselineSubtract()
        if self.badPulse:
            return -1
        # Apply filter if needed
        if movingAverageLength>1:
            tSamples = np.convolve(self.blsSamples,np.ones(movingAverageLength)/movingAverageLength,'same')
        else:
            tSamples = self.blsSamples
        tMax = np.argmax(tSamples)
        targetVal = CFDFraction*tSamples[tMax]
        loopIndex = tMax
        while loopIndex > 0:
            loopIndex -= 1
            if tSamples[loopIndex] < targetVal:
                return (targetVal-tSamples[loopIndex])/(tSamples[loopIndex+1]-tSamples[loopIndex]) + loopIndex
        return -1

    def getStdDev(self):
      return(np.std(self.blsSamples))

    def getKurtosis(self):
      return(stats.kurtosis(self.blsSamples))

    def isDouble(self , show):
      # needs to be optimized
      if not self.baselined:
        self.BaselineSubtract()
      if self.badPulse == True:
        return (True)
      if self.height == -1:
        self.height = np.max(self.blsSamples)

      delta_y = 0.05 * self.height
      delta_x = int( round( len(self.samples) / 81))
      y0 = self.blsSamples[0]
      x0 = 0
      xf = len(self.blsSamples) - 1
      yf = self.blsSamples[xf]
      ym = self.height
      xm = self.maxIndex

      # find linear slope and intercept from beginning to max y = a *x + b
      a = (ym - y0) / (xm - delta_x - x0)
      b = y0 + delta_y

      # find the exponential constants for the falling edge y = c * e^(r(x)) + delta_y
      r = 2*(np.log(xm) - np.log(xf + delta_x)) / (xf - xm - delta_x )
      c =  ym

      x =  np.array( range(0 , len(self.samples))  )
      upperCutoff = c * np.exp( r * (x[xm:] - xm - delta_x)) + delta_y
      lowerCutoff = a * x[:xm] + b

      for xi , y in enumerate(self.blsSamples):
        if xi < xm:
          if y > lowerCutoff[xi]:
            self.badPulse == True
        if xi >= xm:
          if y > upperCutoff[xi - xm]:
            self.badPulse == True

      if show == True:
        plt.plot( x , self.blsSamples)
        plt.plot( x[:xm] , lowerCutoff , label="cutoff")
        plt.plot( x[xm:] , upperCutoff , label="cutoff")
        plt.xlabel("time")
        plt.ylabel("pulse height")
        plt.legend()
        if self.badPulse == True:
          plt.title("Double!!")
        plt.draw()
        plt.pause(0.05)

      return(self.badPulse)






