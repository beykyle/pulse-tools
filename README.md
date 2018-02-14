# rc-detector
This code finds the energy resolution of organic, Compton scatter based, scintillators by iteratively convolving 
a Gaussian with an energy dependent width into an unbroadened Monte Carlo simulated pulse height spectrum until
it matches the measured pulse height spectrum from an experiment corresponding to the simulation.

It can be run for multi-detector systems, where each detector has more than one channel corresponding to it.

It has one .ini config file, which is specified from the command line, e.g.:
    $ python rescon.py config_filename.ini

The config file included here has detailed comments on it's use.

The script requires simulation spectra and experimental spectra in the form of comma seperated, 2 column files, e.g.:
    bin1,histogram_value1
    bin2,histogram_value2
      .     .
      .     .
      .     .

If the experimental data is in the form of raw CAEN digitizer output, getWaveData is used to build spectra,
and a second config file is required. In this case, the familiarity of the user with getWaveData configuration
is assumed. If this is not the case, provide experimental spectra in the form of a comma seperated, 2 column file, 
as specified above.
