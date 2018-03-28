# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 12:17:13 2018

@author: wmst
email: wmst@umich.edu
cell: 757-870-1490
"""
##############################################
##############################################
##############################################


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import LogNorm
from scipy.special import erf
import time
from scipy.optimize import leastsq
import scipy
from scipy.optimize import curve_fit

startTime = time.time()
print("Analyzing Data!")
print("Starting at " + time.strftime('%H:%M:%S'))


##############################################
##############################################
##############################################

'''
##############################################

Step #1: the following variable, pywaves_directory, should be the directory
that contain the 3 python files: dataloader.py, waveform.py, and getwavedata.py.

##############################################
'''

##############################################

pywaves_directory = "./"

##############################################

sys.path.extend([pywaves_directory])

from dataloader import DataLoader
from getwavedata import GetWaveData

##############################################
##############################################
##############################################

def Using_DataLoader():
	'''
	##############################################

	Step #2: Three variables are required to run DataLoader.
	Variable #1, Data_File_Path: A path and data file must be specified.
	Variable #2, dataType: CAEN digitizers will output data in one of three
	formats: DAFCA_STD, DAFCA_DPP_MIXED, or DAFCA_DPP_LIST. This defines the
	data structure for how data is pulled in. We will go through DAFCA_DPP_MIXED
	thoroughly.
	Variable #3, Num_Samples: This is the number of samples in a waveform, not
	the number of waves

	##############################################
	'''

	##############################################

	Data_File_Path = "./Na-22/1/dataFile0.dat"
	dataType = DataLoader.DAFCA_DPP_MIXED
	Num_Samples = 440

	##############################################

	Data = DataLoader(Data_File_Path,dataType,Num_Samples)

	print("DataLoader has two useful functions to new users.")
	print("\n")
	print("The first useful function will tell you how many waves are in the specified file.")
	print("\n")
	time.sleep(3)

	Number_of_data_structures = Data.GetNumberOfWavesInFile()

	print("In : "+str(Data_File_Path)+" , there are "+str(Number_of_data_structures)+" pulses, or more specifically, that many data structures.")
	print("\n")
	time.sleep(1)
	print("The second useful function will load in a specified number of waves or data structures.")
	print("\n")
	time.sleep(1)
	var = input("Lets read in one wave and break down what we are reading in. Sound good? (Y/N): ")
	if var == "Y":
		Waves = Data.LoadWaves(1)
		print("\n")
		time.sleep(1)
		print("This one wave has the following structure:")
		print("\n")
		print('EventSize is the event size in number of bytes: '+str(Waves['EventSize'][0]))
		print('Format is referring to the CAEN datatype: '+str(Waves['Format'][0]))
		print('Channel is referring to the input channel on the digitizer: '+str(Waves['Channel'][0]))
		print('Baseline is the baseline of the pulse: '+str(Waves['Baseline'][0]))
		print('ShortGate is the tail integral of the pulse: '+str(Waves['ShortGate'][0]))
		print('LongGate is the total integral of the pulse: '+str(Waves['LongGate'][0]))
		print('TimeTag is a course absolute time for when a pulse begins (ns): '+str(Waves['TimeTag'][0]))
		print('Extras can be thought of as a finer time that can be combined with TimeTag: '+str(Waves['Extras'][0]))
		print('Samples are the actual wave samples: '+str(Waves['Samples'][0]))

		print("\n")
		time.sleep(1)

		print("Baseline, ShortGate, and LongGate have generally not been used because they do not necessarily accuratley represent what they claim to.")
		print("For more infomration (especially when using Extras), please read the UM2580 DPP PSD User Manual found on CAEN's website.")
		print("\n")
		time.sleep(1)
		print("Now let's plot this pulse. When you are happy with looking at the pulse, close the figure.")

		x = np.arange(0,880,2)
		y = Waves['Samples'][0]
		plt.plot(x,y)
		plt.xlabel("Time (ns)")
		plt.xlabel("Digitizer Units")
		plt.show()
		plt.close()

		print("\n")
		time.sleep(1)
		print("From this pulse, we can attain useful information, but first we must subtract the baseline.")
		print("\n")
		print("In DAFCA, the user can specify how much data of the pulse they want before the rising edge. This way a sufficient amount of data can be acquired to determine the baseline.")
		print("Generally, the baseline is determined by averaging the first 50 samples of the pulse.")
		time.sleep(1)
		Baseline = np.average(Waves['Samples'][0][0:50])
		print("\n")
		print("In this case, the baseline of the pulse is: "+ str(Baseline))
		print("\n")
		print("We can now subtract this baseline and plot the pulse.")
		print("\n")

		Baseline_Subtracted_Wave = Waves['Samples'][0] - Baseline

		x = np.arange(0,880,2)
		y = Baseline_Subtracted_Wave
		plt.plot(x,y)
		plt.xlabel("Time (ns)")
		plt.xlabel("Digitizer Units")
		plt.show()
		plt.close()

		print("We can now attain some useful information about the pulse:")
		print("\n")

		Pulse_Height = np.max(Baseline_Subtracted_Wave)
		Max_Index = np.argmax(Baseline_Subtracted_Wave)
		negativeSamples = np.nonzero(Baseline_Subtracted_Wave[Max_Index:]<0)[0]
		Pulse_Integral = np.sum(Baseline_Subtracted_Wave[:Max_Index + negativeSamples[0]])*2

		dynamic_range_volts = 0.5
		number_of_bits = 15

		VperLSB = dynamic_range_volts/(2**number_of_bits)

		F = 0.2

		Start_Time = CFD(Baseline_Subtracted_Wave, F)

		print("Pulse Height (Digitizer Units): "+str(Pulse_Height))
		print("Pulse Integral (Digitizer Units-ns): "+str(Pulse_Integral))
		print("Start Time (ns): "+str(Start_Time))
		print("\n")
		print("To convert from Digitizer Units to Votlage, it is necessary to multiply by the dynamic range of the digitizer and divide by 2^(number of bits). In this case the conversion is 0.5/2^14. This yields:")
		print("\n")

		Pulse_Height = np.max(Baseline_Subtracted_Wave)*VperLSB
		Pulse_Integral = np.sum(Baseline_Subtracted_Wave[:Max_Index + negativeSamples[0]])*2*VperLSB

		print("Pulse Height (mV): "+str(Pulse_Height))
		print("Pulse Integral (mV-ns): "+str(Pulse_Integral))
		print("\n")
		print("The next code, GetWaveData, does these calculations for us.")
		print("\n")

		x = np.arange(0,880,2)
		y = Baseline_Subtracted_Wave
		plt.plot(x,y)
		plt.xlabel("Time (ns)")
		plt.xlabel("Digitizer Units")
		plt.show()
		plt.close()

	else:
		print("Fine.")

##############################################
##############################################
##############################################

def Using_GetWaveData():
	'''
	##############################################

	Step #3: Two variables are required for GetWaveData
	Variable #1, Config_File_Path: A path and configuration file must be specified.
	Variable #2, getZeroCrossingIntegral: If True, GetWaveData will output the pulse integral,
	otherwise it will output an array of zeros.

	The configuration file should have the following format:

	[Directories]
	data_file_name = dataFile0.dat										### The name of the data file to be analyzed
	data_directory = C:/H2DPI/Data/Na_22_Test_1_10_2018_Overnight/		### The data file's directory
	pywaves_directory = C:/H2DPI/Python_Scripts/						### This is the same path that we specified at the beginning

	[Digitizer]
	dataFormat = DPP_MIXED			### This is the dataformat
	samples_per_waveform = 440		### Number of samples in a wave (specified in DAFCA)
	ns_per_sample = 2				### This is the digitization rate. A 500 MHz digitizer samples ever 2 ns
	number_of_bits = 14				### This is the bit range of the digitizer or the "digitizer untis" that has a range from 0-2^14
	dynamic_range_volts = 0.5		### This is the range of votlage for the digitizer
	polarity = 1					### polarity >0 is a positive pulse, 0 and less is a negative pulse
	baseline_offset = 30			### I beleive this helps determine bad pulses
	baseline_samples = 50			### This is how many samples are averaged to determine the baseline
	number_of_channels = 16			### Number of channels data was collected from

	[Data Management]
	waves_per_load = 10000			### This code will read in this many pulses, analyze them, then bring in this many pulses again until it reads in all pulses
	waves_per_folder = 400000		### Total number of waves that will be brought in and analyzed
	start_folder = 1				### The start folder
	number_of_folders = 5			### How many folders you want it to read through. Note, if start_folder = 1, it will only incrementally read 1,2,3,4,5. If 4 is missing it will give an error and won't read 1,2,3,5,6.
	uneven_factor = 2				### GetWaveData works by predefining arrays of zeros and then filling those arrays. To ensure that there is enough room in the arrays, we arbitrarily icnrease their size by this factor.

	[Pulse Processing]
	cfd_fraction = 0.2				### This is the fraction of the rising edge that denotes the pulse start time. Look up digital constant fraction discrimination for more info.
	total_integral_start = 0		### This is the index from the max index to start the total integral for PSD.
	tail_integral_start = 40		### This is the index from the max index to start the tail integral for PSD.
	integral_end = 150				### This is the index from the max index to end both tail and total integrals.
	apply_crrc4 = 0					### Both of these are for crrc shaping.
	crrc4_shaping_time = 125		###

	If you read all of this, here's a song: https://www.youtube.com/watch?v=tGh4FcZKekA

	##############################################
	'''

	##############################################

	Config_File_and_Path = "./rescon_testdata.ini"

	##############################################

	print("This may take a minute...")
	H2DPI_Data = GetWaveData(Config_File_and_Path,getZeroCrossingIntegral=True)

	###########
	chCount = 0
	ph = 1
	amp = 2
	tailInt = 3
	totalInt = 4
	cfd = 5
	ttt = 6
	extras = 7
	fullTime = 8
	flags = 9
	rms = 10
	###########

	print("\n")
	print("Where do I begin. To tell the story of... ")
	print("\n")
	print("When GetWaveData is called, it will return an array of 11 arrays.")
	print("\n")
	print("The first array,")
	print(H2DPI_Data[0])
	print("is an array with length equal to the number of channels specified. The values of each index correspond to how many waves were recorded in that channel.")
	time.sleep(1)
	print("\n")
	print("All the other arrays have the shape ([number_of_channels, int(number_of_folders*waves_per_folder*unevenFactor/number_of_channels)])")
	print("\n")
	print("Let's first define what all of those arrays are. In this code I've defined the variable H2DPI_Data to be the one calling GetWaveData so I will be using that show what is being brought in.")
	time.sleep(1)
	print("\n")
	print("H2DPI_Data[0] ==> The values of each index correspond to how many waves were recorded in that channel.")
	print("H2DPI_Data[1] ==> The values of each index correspond to the pulse integral (V-ns) of each recorded pulse in that channel.")
	print("H2DPI_Data[2] ==> The values of each index correspond to the pulse height (Digitizer Units) of each recorded pulse in that channel.")
	print("H2DPI_Data[3] ==> The values of each index correspond to the tail integral (Digitizer Units-ns) of each recorded pulse in that channel.")
	print("H2DPI_Data[4] ==> The values of each index correspond to the total integral (Digitizer Units-ns) of each recorded pulse in that channel.")
	print("H2DPI_Data[5] ==> The values of each index correspond to the start time (ns) of each recorded pulse in that channel.")
	print("H2DPI_Data[6] ==> The values of each index correspond to the time tag of each recorded pulse in that channel.")
	print("H2DPI_Data[7] ==> The values of each index correspond to the extras of each recorded pulse in that channel.")
	print("H2DPI_Data[8] ==> The values of each index correspond to the absolute start time of each recorded pulse in that channel. Adding this will CFD will give the complete start time of the pulse.")
	print("H2DPI_Data[9] ==> This is an array of zeros. Ignore it.")
	print("H2DPI_Data[10] ==> The values of each index correspond to the root-mean-squre of the baseline of each pulse.")
	print("\n")
	var = input("Type anything to continue.")
	print("\n")

	print("So let's look at Bar 1 which corresponds to channels 0 and 1.")
	print("\n")
	print("Let's plot a pulse integral spectrum. To do this we must first add the integrals from channels 0 and 1 together like:")
	print("Integrals_Bar_1 = H2DPI_Data[1][0] + H2DPI_Data[1][1]")
	print("\n")

	Integrals_Bar_1 = H2DPI_Data[1][0] + H2DPI_Data[1][1]

	plt.hist(Integrals_Bar_1, bins =300, range = [0,50], histtype='step', label='Bar 1')
	plt.xlabel('Pulse Integral (V-ns)')
	plt.ylabel('Counts')
	plt.legend()
	plt.show()
	plt.close()

	var = input("Wow!! There were a lot of zeros in that for some reason... Press X to continue... Well, X and then enter.")
	print("\n")
	print("That's because of how we define the buffer. However, because of H2DPI_Data[0], we know how many pulses are in channels 0 and 1. So to actually attain the pulse integral spectrum we need something like:")
	print("\n")
	print("Integrals_Bar_1 = H2DPI_Data[1][0][:H2DPI_Data[0][0]] + H2DPI_Data[1][1][:H2DPI_Data[0][1]]")
	print("\n")
	print("We can also define a variable to represent H2DPI_Data[0] such as: H2DPI_Counts = H2DPI_Data[0]. Then:")
	print("\n")
	print("Integrals_Bar_1 = H2DPI_Data[1][0][:H2DPI_Counts[0]] + H2DPI_Data[1][1][:H2DPI_Counts[1]]")
	print("\n")
	print("By doing this we are telling it to only look at data that was actually taken.")

	H2DPI_Counts = H2DPI_Data[0]
	Integrals_Bar_1 = H2DPI_Data[1][0][:H2DPI_Counts[0]] + H2DPI_Data[1][1][:H2DPI_Counts[1]]
	plt.hist(Integrals_Bar_1, bins =300, histtype='step', label='Bar 1')
	plt.xlabel('Pulse Integral (V-ns)')
	plt.ylabel('Counts')
	plt.legend()
	plt.show()
	plt.close()

	print("\n")
	print("We can then loop this and overlay the pulse integrals for all bars.")
	print("\n")

	Counts_per_Channel = H2DPI_Data[0]
	Integral = H2DPI_Data[1]

	Number_of_Channels = len(H2DPI_Counts)
	Bar_List = [1,2,3,4,8,7,6,5]
	for chan, bar in zip(range(0,Number_of_Channels,2), Bar_List):
		Integral_1 = Integral[chan][:Counts_per_Channel[chan]]
		Integral_2 = Integral[chan+1][:Counts_per_Channel[chan+1]]
		Integral_Tot = (Integral_1 + Integral_2)
		plt.hist(Integral_Tot, bins =300, histtype='step', label='Bar '+str(bar))
	plt.xlabel('Pulse Integral (V-ns)')
	plt.ylabel('Counts')
	plt.legend()
	plt.show()
	plt.close()
	print("\n")
	print("That's it for now.")
	print("\n")
	print("Cave Johnson. We're done here.")

##############################################
##############################################
##############################################

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

##############################################
##############################################
##############################################

def Running():
	print("\n")
	print("Alright, let's get started!")
	print("This first test involves something the lab boys call repulsion gel."+"\n")
	print("User specified steps in this script are denoted by '''"+"\n"+"and say 'Step #N' where N denotes the step number.")
	print("\n")
	var = input("Have you gone through this script and completed the specified steps? (Y/N) ")

	if var == "N":
		print("\n")
		print("Complete those steps and then run this code.")
	elif var == "Y":
		print("\n")
		print("##############################################")
		print("##############################################")
		print("##############################################")
		time.sleep(1)
		print("\n")
		print("Let's begin with DataLoader.")
		print("\n")
		time.sleep(1)
		Using_DataLoader()
		print("\n")
		print("##############################################")
		print("##############################################")
		print("##############################################")
		print("\n")
		time.sleep(1)
		Using_GetWaveData()

Running()

