# -*- coding: utf-8 -*-
"""
GetWaveData is a function that uses Waveform and Data loader, along with a
configuration file to return basic waveform information.
Created on Wed Dec  7 06:19:10 2016

@author: Marc Ruch
Edited by Kyle Beyer
"""

from __future__ import print_function
import numpy as np
import sys
import platform
import time
import configparser

from matplotlib import pyplot as plt
from time import sleep

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r %s |%s| %s%% %s'%(prefix, bar, percent, suffix), end = "\r")
    # Print New Line on Complete
    if iteration == total:
        print()

def getWaveTail(waveform):
  return(waveform.GetIntegralFromPeak(tailIntegralStart,integralEnd)*VperLSB*ns_per_sample)

def getWavePH(waveform):
  return(waveform.GetIntegralToZeroCrossing()*VperLSB*ns_per_sample)

def getWaveAmp(waveform):
  return(waveform.GetMax())

def getWaveTot(waveform):
  return(waveform.GetIntegralFromPeak(totalIntegralStart,integralEnd)*VperLSB*ns_per_sample)

def getWaveCFD(waveform):
  return( waveform.GetCFDTime(cfdFraction)*ns_per_sample )

def getWaveRMS(waveform):
  return( waveform.GetRMSbls(nBaselineSamples))


def GetWaveData(configFileName, getZeroCrossingIntegral=True , getWaves=True, loud=False, getTail=False, getTot=False,
    getPH=False , getAmp=False, getRMS=False, getExtras=False, getFullTime=False, getCFD=False, getflags=False ):

    # --------------------------------------------------------------------------------------------------------------------- #
    #    Configuration
    # --------------------------------------------------------------------------------------------------------------------- #

    config = configparser.ConfigParser()
    config.read(configFileName)

    # Setup data info
    # Directories
    data_directory = config['Directories']['data_directory']
    data_file_name = config['Directories']['data_file_name']
    goodInd = False
    if 'goodind_file_name' in config['Directories']:
      goodInd = True
      goodind_file_name = config['Directories']['goodind_file_name']
    pywaves_directory = config['Directories']['pywaves_directory']

    # Load pywaves
    sys.path.extend([pywaves_directory])
    from dataloader import DataLoader
    from waveform import Waveform

    # Digitizer
    global  dataFormatStr
    global  nSamples
    global  ns_per_sample
    global  number_of_bits
    global  dynamic_range_volts
    global  polarity
    global  baselineOffset
    global  nBaselineSamples
    global  nCh
    global  nWavesPerLoad
    global  nWaves
    global  startFolder
    global  nFolders
    global  unevenFactor
    global  cfdFraction
    global  integralEnd
    global  totalIntegralStart
    global  tailIntegralStart
    global  applyCRRC4
    global  CRRC4Tau

    dataFormatStr         = config['Digitizer']['dataFormat']
    nSamples              = int(config['Digitizer']['samples_per_waveform'])
    ns_per_sample         = int(config['Digitizer']['ns_per_sample'])
    number_of_bits        = int(config['Digitizer']['number_of_bits'])
    dynamic_range_volts   = float(config['Digitizer']['dynamic_range_volts'])
    polarity              = int(config['Digitizer']['polarity'])
    baselineOffset        = int(config['Digitizer']['baseline_offset'])
    nBaselineSamples      = int(config['Digitizer']['baseline_samples'])
    nCh                   = int(config['Digitizer']['number_of_channels'])
    nWavesPerLoad         = int(config['Data Management']['waves_per_load'])
    nWaves                = int(config['Data Management']['waves_per_folder']) # per folder
    startFolder           = int(config['Data Management']['start_folder'])
    nFolders              = int(config['Data Management']['number_of_folders'])
    unevenFactor          = int(config['Data Management']['uneven_factor'])
    cfdFraction           = float(config['Pulse Processing']['cfd_fraction'])
    integralEnd           = int(config['Pulse Processing']['integral_end'])
    totalIntegralStart    = int(config['Pulse Processing']['total_integral_start'])
    tailIntegralStart     = int(config['Pulse Processing']['tail_integral_start'])
    applyCRRC4            = bool(int(config['Pulse Processing']['apply_crrc4']))
    CRRC4Tau              = float(config['Pulse Processing']['crrc4_shaping_time'])


    # Pre-calc
    if dataFormatStr == 'DPP_MIXED':
        dataFormat = DataLoader.DAFCA_DPP_MIXED
    elif dataFormatStr == 'STANDARD':
        dataFormat = DataLoader.DAFCA_STD
    if platform.system() is 'Windows':
        directory_separator  = '\\'
    else:
        directory_separator  = '/'
    nLoads = int(nWaves/nWavesPerLoad)
    chBufferSize = int(nFolders*nWaves * unevenFactor / nCh  )
    VperLSB = dynamic_range_volts/(2**number_of_bits)
    fileTimeGap = 2**43 # Note: no more than 3 hours per measurement!

    # --------------------------------------------------------------------------------------------------------------------- #
    #    Data structure setup
    # --------------------------------------------------------------------------------------------------------------------- #

    dataGetter  = {}
    dataStorage = {}

    if getTail == True:
      dataGetter['tail']   = getWaveTail
      dataStorage['tail']  = np.zeros((nCh,chBufferSize))
    if getTot  == True:
      dataGetter['total']  = getWaveTot
      dataStorage['total'] = np.zeros((nCh,chBufferSize))
    if getPH   == True:
      dataGetter['ph']     = getWavePH
      dataStorage['ph']    = np.zeros((nCh,chBufferSize))
    if getCFD  == True:
      dataGetter['cfd']    = getWaveCFD
      dataStorage['cfd']   = np.zeros((nCh,chBufferSize))
    if getRMS  == True:
      dataGetter['rms']    = getWaveRMS
      dataStorage['rms']   = np.zeros((nCh,chBufferSize))
    if getAmp  == True:
      dataGetter['amp']    = getWaveAmp
      dataSotage['amp']    = np.zeros((nCh,chBufferSize))

    if getExtras == True and dataFormatStr == 'DPP_MIXED':
      dataSotage['extra']       = np.zeros((nCh,chBufferSize), dtype=np.uint32)
    if getFullTime == True and dataFormatStr == 'DPP_MIXED':
      dataSotage['fulltime']    = np.zeros((nCh,chBufferSize), dtype=np.uint64)

    # Setup mandatory channel queues
    ttt = np.zeros((nCh,chBufferSize), dtype=np.uint32)
    chCount = np.zeros(nCh, dtype=np.uint32)
    flags = np.zeros((nCh,chBufferSize), dtype=np.uint32)

    # --------------------------------------------------------------------------------------------------------------------- #
    #    Data aquisition
    # --------------------------------------------------------------------------------------------------------------------- #

    startTime = time.time()
    print("Running GetWaveData!")
    print("Starting at " + time.strftime('%H:%M:%S'))

    # Setup data loader
    waveform = Waveform(np.zeros(nSamples), polarity, baselineOffset, nBaselineSamples)

    pulses = []
    # Queue up waves
    for f in range(startFolder, startFolder+nFolders):
        print('\n Folder {}:'.format(f))
        fullDFileName = data_directory + directory_separator + str(f) + directory_separator + data_file_name
        print(fullDFileName)

        datloader     = DataLoader(fullDFileName,dataFormat,nSamples)
        nWavesInFile  = datloader.GetNumberOfWavesInFile()

        if (nWavesInFile < nWaves):
            print('Warning: requested more waves than exists in file!')
            loadsInFile = int(np.ceil(nWavesInFile/nWavesPerLoad))
            print('Loading all {} waves instead...'.format(nWavesInFile))
            lastLoad = nWavesInFile - (loadsInFile-1)*nWavesPerLoad

        else:
            loadsInFile = nLoads
            lastLoad = nWavesPerLoad

        if goodInd == True:
          goodIndFileName = data_directory + directory_separator + str(f) + directory_separator + goodind_file_name
          goodIndices = []
          with open(goodIndFileName , "r") as goodfi:
            good_lines = goodfi.readlines()
            for line in good_lines:
              goodIndices.append(int(line) - 1)
          goodIndices = np.array(goodIndices)
        else:
          goodIndices = np.arange(0,nWavesInFile - 1)

        waveNum = 0
        printProgressBar(0, loadsInFile , prefix = 'Progress: ' , suffix = 'Complete')
        for load in range(loadsInFile):
            if(load == loadsInFile-1):
                wavesThisLoad = lastLoad
            else:
                wavesThisLoad = nWavesPerLoad
            waves = datloader.LoadWaves(wavesThisLoad)
            for w in range(0 , wavesThisLoad - 1):
                if waveNum == goodIndices[0]:
                  goodIndices = goodIndices[1:]
                  ch = waves[w]['Channel']
                  chCount[ch] += 1

                  # set up waveform structure
                  waveform = Waveform( waves[w]['Samples'] , polarity , baselineOffset ,
                                         nBaselineSamples , ch=waves[w]['Channel'] , time=waves[w]['TimeTag'])
                  waveform.BaselineSubtract()

                  # CCR4
                  if applyCRRC4:
                      waveform.ApplyCRRC4(ns_per_sample, CRRC4Tau)

                  # add timing data to mandatory channel structure
                  ttt[ch][chCount[ch]] = waves[w]['TimeTag']

                  # populate data structure for optional
                  for key , value in dataGetter.items():
                    dataStorage[key][ch][chCount[ch]] = getGetter[key](waveform)

                    ttt[ch][chCount[ch]] = waves[w]['TimeTag']
                  # get the whole wave
                  if getWaves == True:
                    pulses.append(waveform)

                waveNum += 1 # increment the wave counter in the current load
              # end iteration over waves in load - end load

            # update progress bar following new load
            printProgressBar(load + 1, loadsInFile , prefix = 'Progress: ' , suffix = 'Complete')

    endTime = time.time()
    runTime = endTime - startTime
    print("\nGetWaveData took {} s".format(runTime))
    if getWaves == False:
      return chCount, ttt , dataStorage
    else:
      return chCount, ttt , dataStorage , pulses

