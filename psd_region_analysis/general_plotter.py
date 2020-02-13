__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as pylab
from matplotlib import rc
from matplotlib.colors import LogNorm
import matplotlib.font_manager
from sklearn import mixture
from pylab import *
from scipy.optimize import leastsq

import csv
import sys
import os

"""
This module does pulse shape discrimination
"""

def fig_setup():
    fig = plt.figure(figsize=(12, 10), dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    rc('text', usetex=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    params = {'legend.fontsize': 12,
             'axes.labelsize': 12,
             'xtick.labelsize':12,
             'ytick.labelsize':12}
    pylab.rcParams.update(params)
    return fig , ax

class fit_quad_poly:
    def __init__(self):
        self.x = []
        self.y = []

    def solve(self):
        p = np.polyfit(self.x,self.y,2)
        return p[0], p[1], p[2]

    def onclick(self, event):
        if (event.xdata and event.ydata) and (event.xdata > 0 ) and (event.ydata > 0):
            self.x.append(event.xdata)
            self.y.append(event.ydata)


def scatter_desnity(data , labels, save=False, name="plot.png"):
  print("plotting PSD")
  # pre
  total = data["total"]
  tail = data["tail"]
  fig , ax = fig_setup()
  ratio = np.divide(tail,total)
  # plotting
  fig , ax = fig_setup()
  dat = plt.hist2d(total, ratio, (100, 100), cmap=plt.cm.jet_r, norm=LogNorm())
  p = plt.colorbar()
  p.set_label("Counts/s")
  plt.ylim([0 , 1])
  plt.xlabel(labels[0])
  plt.ylabel(labels[1])
  plt.plot([0,max(total)] , [0.5 , 0.5] , 'k--')
  # discrimnation fit np
  fqp = fit_quad_poly()
  cid = fig.canvas.mpl_connect('button_press_event', fqp.onclick)
  plt.show()
  fig.canvas.mpl_disconnect(cid)
  a , b, c = fqp.solve()
  # replot
  fig , ax = fig_setup()
  dat= plt.hist2d(total, ratio, (100, 100), cmap=plt.cm.jet_r, norm=LogNorm())
  p = plt.colorbar()
  p.set_label("Counts/s")
  plt.ylim([0 , 1])
  plt.xlabel(labels[0])
  plt.ylabel(labels[1])
  plt.plot([0,max(total)] , [0.5 , 0.5] , 'k--')
  x = np.linspace(0,max(total), 10000)
  plt.plot(x , a * x**2 + b *x + c, 'k--')

  if save:
      print("Saving PSD plot to ./" + name)
      plt.savefig(name)
  else:
      plt.show()

  #discriminate(total, ratio)

class dg_fitter:
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def double_gaussian(self, params):
        (c1, mu1, sigma1, c2, mu2, sigma2) = params
        res =   c1 * np.exp( - (self.x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
              + c2 * np.exp( - (self.x - mu2)**2.0 / (2.0 * sigma2**2.0) )
        return res

    def double_gaussian_fit(self, params ):
        fit = self.double_gaussian( params )
        return (fit - self.y)

def get_midpoints(arr):
    out = []
    for i , a in enumerate(arr[:-1]):
        out.append((arr[i] + arr[i+1])*0.5)
    return out

def analyze_slice():
    fig , ax = fig_setup()
    slice_max = bounds[i+1]
    mask = np.logical_and( total < slice_max, np.logical_and(total >= slice_min , ratio <= 0.5))
    slice_ratio = np.extract(mask , ratio)
    slice_title = "Slice: " + str(i) + ": " + "{0:.2f}".format(slice_min,2) + " Vns - " + "{0:.2f}".format(slice_max) + " Vns"
    n, bins, hh = plt.hist(slice_ratio, bins=25)
    mp_bins = get_midpoints(bins)
    dg = dg_fitter(mp_bins, n)
    fit = leastsq( dg.double_gaussian_fit , [10E6, 0.1 , 0.1 , 1E5 , 0.3, 0.1] )
    plt.plot(mp_bins , dg.double_gaussian(fit[0]) , 'r--')
    plt.xlabel("Ratio")
    plt.yscale("log")
    plt.ylabel("Counts/s")
    plt.title(slice_title)
    plt.show()

def discriminate(total, ratio, max_ratio=0.5, slices=10):
    mintotal = 0
    maxtotal=max(total)
    bounds = np.linspace(mintotal, maxtotal, slices)
    for i, slice_min in enumerate(bounds[:-1]):
        analyze_slice(bounds, slice_min, total, ratio)


def try_convert(string):
    if string:
        try:
            return float(string)
        except:
            return

def readPSD(fi1):

  with open(fi1) as psdfile:
    lines = psdfile.readlines()

  arr_total = []
  arr_tail = []
  arr_time = []

  for line in lines:
    line = [ x.strip().rstrip("\r\n") for x in line.split(" ")]
    line = list(filter(None, line))

    total = try_convert(line[1])
    tail = try_convert(line[2])
    time = try_convert(line[0])

    if (total and tail and time) and (total > 0 ) and (tail >= 0 ):
        arr_time.append(time)
        arr_total.append(total)
        arr_tail.append(tail)


  return( { "total": np.array(arr_total) ,
            "tail" : np.array(arr_tail)  ,
            "time" : np.array(arr_time)  } )

def append_channel_data(channel_data, new):
    channel_data["total"] = np.append(channel_data["total"] , new["total"])
    channel_data["tail"] = np.append(channel_data["tail"] , new["tail"])
    channel_data["time"] = np.append(channel_data["time"] , new["time"])
    return channel_data

def get_channel(fname):
    fname = fname.replace(".txt" , "")
    breakstr = fname.split("ch")
    return(int(breakstr[1]))

def plot_PID(data):
    pass

def transform_all_data(data, const):
    pass

def flatten_channels(data):
    tail = np.array([])
    total = np.array([])
    time = np.array([])
    for channel , df in data.items():
        tail = np.append(tail , df["tail"])
        time = np.append(time , df["time"])
        total = np.append(total , df["total"])

    return { "total" : total ,
             "tail"  : tail  ,
             "time"  : time  }

def read_all_channels(path):
    data = {}
    print("Reading all channels in " + path)
    max_fstring_len = 65
    for subdir, dirs, files in os.walk(path):
        if files and ".txt" in files[0]:
            for f in files:
                fullpath = subdir + "/" + f
                channel  = get_channel(f)
                fstring  = " ===> Now reading: " + fullpath
                chanstr  = " >>>> Channel #" + str(channel)
                print( fstring.ljust(max_fstring_len) + "{:<20}".format(chanstr))
                if (channel in data):
                    data[channel] = append_channel_data(data[channel] , readPSD(fullpath))
                else:
                    data[channel] = readPSD(fullpath)
    return data

if __name__ == "__main__":
    path = sys.argv[1]
    meas_time_sec = 1
    #meas_time_sec = float(sys.argv[2])
    data = read_all_channels(path)
    transform_all_data(data , 1/meas_time_sec)

    name = path.split("/")[1].strip().rstrip("\r\n") + ".png"

    all_channels = flatten_channels(data)
    scatter_desnity(all_channels , ["total [Vns]" , "tail [Vns] / total [Vns]"], save=True )
    #scatter_desnity(flat_data["tail"] , flat_data["total"])
