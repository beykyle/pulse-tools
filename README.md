# pulse_tools
A collection of tools for analyzing list mode organic scintillator pulses and characterizing detector systems. 
## psd_region_analysis
An interactive GUI that allows a user to select a region from a pulse shape discrimination plot (scatter density plot of the total pulse integral to tail/total ratio) to view and save pulses from. 
Useful for identifying if a region on your plot is due to a particle interaction, clipped or malformed pulses, double pulses, or what have you.
## pulse_classification
A collection of methods for classifying pulses as good/clipped/double etc.
## spec_broadening
A collection of scripts useful for finding energy resolution and gain (e.g. pulse integral [V-ns] to light output [MeVee] conversion) 
for organic, Compton scatter based, scintillators by iteratively convolving 
a Gaussian into an unbroadened Monte Carlo simulated pulse height spectrum until
it matches the measured pulse height spectrum from an experiment corresponding to the simulation.

It can be run for multi-detector systems, where each detector has more than one channel corresponding to it.

It has one .ini config file, which is specified from the command line, e.g.:
    $ python rescon.py config_filename.ini

The ''' *.ini '''' config file included here has detailed comments on it's use.

The script requires simulation spectra and experimental spectra in the form of comma seperated, 2 column files, e.g.:
'''
    bin1,histogram_value1
    bin2,histogram_value2
      .     .
      .     .
      .     .
'''
If the experimental data is in the form of raw CAEN digitizer output, getWaveData is used to build spectra,
and a second config file is required. In this case, the familiarity of the user with getWaveData configuration
is assumed. If this is not the case, provide experimental spectra in the form of a comma seperated, 2 column file, 
as specified above.
## low_level
Handles the low level reading of CAEN digitizer data files. I am not the author of these.
