#!/usr/bin/env python

"""
GammaImager.py - a script to project a simulated image of a gamma source from a PoliMi collision file
resulting from a n MCNPX-PoliMi simulation of a mixed Compton system
 - e.g. a system where either double compton scatter, compton scatter -> full energy deposition, or both
   event types are possible

"""

from matplotlib  import pyplot as plt
from collections import namedtuple
import numpy as np
import sys
import configparser

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"


# --------------------------------------------------------------------------------------------------------------------- #
#   Detector class
# --------------------------------------------------------------------------------------------------------------------- #

class Detector:
  def __init__(self  , xdev , ydev , zdev , tdev , a , b , c , k=[] , kdev = [] , E2_cutoff=0 ):
    self.xdev = xdev
    self.ydev = ydev
    self.zdev = zdev
    self.tdev = tdev
    self.a = a
    self.b = b
    self.c = c
    self.k = k
    self.kdev      = kdev
    self.E2_cutoff = E2_cutoff

  def sampleX(self , x):
    return( np.random.normal(x , self.xdev))

  def sampleY(self , y):
    return( np.random.normal(y , self.ydev))

  def sampleZ(self , z):
    return( np.random.normal(z , self.zdev))

  def getEnergyFWHM(self, Ei ):
    return( self.a + self.b * np.sqrt( Ei + self.c * Ei**2 ) )

  def sampleEnergy(self, Ei , Edep ):
    return( np.random.normal( self.Edep , 0.2355 * getEnergyFWHM(Ei) ) )

  def sampleTime(self, t ):
    return( np.random.normal( t , self.tdev ))


# --------------------------------------------------------------------------------------------------------------------- #
#  event processing
# --------------------------------------------------------------------------------------------------------------------- #

class eventData:
  """
  The minimum information needed for backprojection is the unit direction vector between events, and the Compton cone
  angle, as well as the uncertainty in those values. This class acts a simple data structure that holds that information.

  """
  def __init__(self , vec , vecUncertainty , angle , angleUncertainty):
    """
    Constructor
    @params:
        vec                 - Required  : unit vector between events [x,y,z]
        vecUncertainty      - Required  : uncertainty in vec [delta_x , delta_y , delta_z]
        angle               - Required  : backprojected cone angle
        angleUncertainty    - Required  : uncertainty in angle
    """
    self.vec = vec
    self.vecUncertainty = vecUncertainty
    self.angle = angle
    self.angleUncertainty = angleUncertainty

def getPositionFromDetector( detector , x , y , z):
  """
  Given a detector object corresponding to a cell in the collision file, as well as the
  exact interaction position from the collision file, returns the reconstructed interaction position.
  Most Compton imagers can only record the position as the position of the detector, with a characteristic
  uncertainty related to the dimensions of the detector. Some Compton imagers can reconstruct position of
  an interaction within a specific direction. For example, Let a compton imager consist of a detector array
  where each detector is a vertical rectangular bar, where the width in the x and y directions, xdev and ydev,
  are much smaller than the width in the z direction. In a real detector, the x,y position of the event
  is taken to be the x,y positions at the vertical axis of the bar, and it has an uncertainty of the widths of
  the bar in the x and y directions respectively. However, if there is a mechanism to reconstruct the position of
  the interaction in the z direction in the bar, then the detector object will have been constructed without a z argument,
  and the interaction location will instead be sampled from a normal distribution centered at the actual z positon of the
  event, and a standard deviation equal to detector.zdev, the uncertainty of the reconstruction.
  @params:
      detector  - Required  : detector object corresponding to MCNP cell of event
      x         - Required  : actual x location of interaction within detector
      y         - Required  : actual y location of interaction within detector
      z         - Required  : actual z location of interaction within detector
  """
  if hasattr( detector , 'x' ):
    # if the reconstruction is based on
    # the location of the detector, sample
    # around its center
    xout = detector.x
  else:
    # otherwise sample about the position of
    # interaction within the detector
    xout = detector.sampleX( x )

  if hasattr( detector , 'y' ):
    yout =  detector.y
  else:
    yout = detector.sampleY( y )

  if hasattr( detector , 'z' ):
    zout = detector.z
  else:
    zout = detector.sampleZ( z )

  pos = [xout , yout , zout ]
  unc = [ detector.xdev , detector.ydev , detector.zdev]
  return(pos , unc)

def getVector( pos1 , pos2 , un1 , un2):
  """
  Returns normal vector between events in the detector array, as well as the uncertainty
  Uncertainty is propagated through the equation for a normalized displacement between to
  points in R^3
  @params:
      pos1    - Required  : position of 1st interaction
      pos2    - Required  : position of 2nd interaction
      un1     - Required  : uncertainty in 1st position
      un2     - Required  : uncertainty in 2nd position
  """
  xdev =  np.sqrt(  un1[0]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
                                  ( (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 +
                                    (pos2[2] - pos1[2])**2 )**3  ) +
                    un2[0]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
                                  ( (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 +
                                    (pos2[2] - pos1[2])**2 )**3  ) +
                    un1[1]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
                                  ( (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 +
                                    (pos2[2] - pos1[2])**2 )**3  ) +
                    un2[1]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
                                  ( (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 +
                                    (pos2[2] - pos1[2])**2 )**3  )+
                    un1[2]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
                                  ( (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 +
                                    (pos2[2] - pos1[2])**2 )**3  ) +
                    un2[2]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
                                  ( (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 +
                                    (pos2[2] - pos1[2])**2 )**3  )
                    )
  ydev = 0
  zdev = 0
  uncertainty=[ xdev , ydev , zdev]
  vec =  np.array( [ pos2[0] - pos1[0] , pos2[1] - pos1[1] , pos2[2] - pos1[2] ] )
  return(vec / np.linalg.norm(vec) , uncertainty)


def getDoubleScatterConeAngleCos(  E1 , E2 , devE1 , devE2 , k=2 , kdev=0):
  """
  Returns the cosine of the backprojection cone angle in a double Compton scatter event, given
  observables  and their uncertainties( E1, E2, devE1, devE1), as well as the optimal
  kinematic approximation and it's uncertainty (k , kdev).
  @params:
      pos1    - Required  : position of 1st interaction
      pos2    - Required  : position of 2nd interaction
      un1     - Required  : uncertainty in 1st position
      un2     - Required  : uncertainty in 2nd position
  """
  angle = 1 - (0.511 / (k*E2 ) - 0.511 / (E1 + k * E2)  )
  # propagate uncertainty in energy and k
  uncertainty = np.sqrt( devE1**2 * 0.261121 / (E1 + k * E2)**4 +
                         devE2**2 * ( 0.511 / (k * E2**2) - 0.511 / (E1 + k * E2)**2 )**2 +
                         kdev**2  * ( 0.511 / (k**2 * E2) - 0.511 / (E1 + k * E2)**2 )**2
                       )
  return angle , uncertainty

def getFullDepConeAngleCos( E1 , E2 , devE1 , devE2):
  angle =  1 - (0.511 / E2 - 0.511 / (E1 + E2)  )
  # propagate uncertainty in energy
  uncertainty = np.sqrt( devE1**2 * 0.261121 / (E1 + E2)**4 +
                         devE2**2 * ( 0.511 / (E2**2) - 0.511 / (E1 + E2)**2 )**2
                       )
  return(angle , uncertainty)



# --------------------------------------------------------------------------------------------------------------------- #
#  Collision file parsing function
# --------------------------------------------------------------------------------------------------------------------- #

def getImageDataFromCollisonFile( collisionFile , detectors , numLines=1000 , loud=False):
  """
  Iterates through a collision file from MXNPX-PoliMi in linear time
  Selects subsequent gamma events with the same particle ID that
  scatter from one cell to another, and use the event processing
  functions to map from observables and detector characteristics
  to a tuple of ( theta, theta_uncertainty , vector , vector-uncertainty)

  See MCNPX-PoliMi manual for details oin collision file structure.

  Each cell in the collision file must have a corresponding detector
  in the detector dictionary (from cell to detector object). The detector
  object is used to randomly sample intrinsic uncertainties into the
  imaging observables, as well as provide options for optimizing the
  kinematic approzimations in imaging.

  @params:
      collisionFile    - Required  : path to collision file (str)
      detectors        - Required  : dictionary from cell to Detector object (see description)
      numLines         - Optional  : number of lines in the collisio file to read
      loud             - Optional  : boolean - if true print progress bar (default False)
  """
  # initialize empty list of image data
  imageData = []

  lines = []
  # read in the entire collision file
  with open(collisionFile  , "r") as filestream:
    for i , line in enumerate(filestream):
      if i <= numLines:
        lines.append(line)

  if loud == True:
    if numLines > len(lines):
      numLines = len(lines)
      print("Only " + str(numLines) + " lines in file" )
    print( "Now parsing " + str(numLines) +  " collisions from " + collisionFile )
    print_progress(0, numLines, prefix='Progress', suffix='', decimals=1, bar_length=100)

  for i ,  line in enumerate(lines[:-1]):
    # get a block of two lines
    line     = [x for x in line.split(" ")       if x != '']
    nextline = [x for x in lines[i+1].split(" ") if x != '']
    # if it is a photon that compton scatters in a cell, and the next line is the same photon=
    # and it scatters into a different cell, where it is either scattered again or captured
    if (  (line[2] == "2") and (line[3] == "1") and (line[0] == nextline[0]) and (int(nextline[5]) != int(line[5]))
                           and (nextline[3] == "1" or nextline[3] == "3" or nextline[3] == "4" )  ):

      # set temporary detector objects
      tmpDet1 = detectors[ line[5] ]
      tmpDet2 = detectors[ nextline[5] ]

      # get vector between scattering events
      pos1 , unc1 = getPositionFromDetector( tmpDet1       , float(line[8] ) ,
                                            float(line[9]) , float(line[10]) )
      pos2 , unc2 = getPositionFromDetector( tmpDet2            , float(nextline[8] ) ,
                                             float(nextline[9]) , float(nextline[10]) )
      vec , vecUncertainty = getVector( pos1 , pos2 , unc1 , unc2)

      # check if it is a scatter -> scatter or scatter -> capture event
      if nextline[3] == "1":
        # double scatter event
        cos , unc = getDoubleScatterConeAngleCos( float( line[6] )     ,
                                                    float( nextline[6] ) ,
                                                    0.2355 * tmpDet1.getEnergyFWHM( float(line[-1]) )     ,
                                                    0.2355 * tmpDet2.getEnergyFWHM( float(nextline[-1]) ) ,
                                                    k=tmpDet1.k[    int(nextline[5]) ] ,
                                                    kdev=tmpDet1.kdev[ int(nextline[5]) ]
                                                  )

      else:
        cos , unc = getFullDepConeAngleCos( float( line[6] )     ,
                                              float( nextline[6] ) ,
                                              0.2355 * tmpDet1.getEnergyFWHM( float(line[-1]) )     ,
                                              0.2355 * tmpDet2.getEnergyFWHM( float(nextline[-1]) ) ,
                                            )

      #append to imageData
      imageData.append( eventData( vec , vecUncertainty , cos  , unc ) )

    # update progress bar following new load
    if loud == True:
        print_progress(i, numLines, prefix='Progress', suffix='', decimals=1, bar_length=100)

  # divide the amount of imageable events (double scatter or scatter -> absorption) by the
  # number of histories used to get image efficiency
  imageEfficiency = len(imageData) / float(nextline[0])
  efficiencyUncertainty = np.sqrt(  len(imageData) / float( nextline[0] )  -
                                    (1 / float(nextline[0])**2 ) * len(imageData)**2
                                 )

  if loud == True:
    print("\n\n Found " + str(len(imageData)) + " imageable events, out of " + nextline[0] +
          " source particle histories checked, for an image efficiency of " + str(imageEfficiency) +
          " +/- " +  str(efficiencyUncertainty) + "\n\n"
         )

  return( imageData , imageEfficiency , efficiencyUncertainty )


# --------------------------------------------------------------------------------------------------------------------- #
#   Image plotting and writing functions
# --------------------------------------------------------------------------------------------------------------------- #

def plotZ(theta, Phi, Z, counter):
    plt.pcolormesh(Theta,Phi,Z, cmap='jet') #http://matplotlib.org/users/colormaps.html
    plt.xlabel(r"Azimuthal Angle [$\Theta$]")
    plt.ylabel(r"Altitude Angle [$\Phi$]")
    plt.title("Cones: "+str(counter))
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.colorbar()
    plt.savefig(sys.argv[1]+str(counter)+".png", ext="png", close=True, verbose=False)
    plt.close()

def writeOut(Z , fname):
    np.savetxt(fname , Z)

# --------------------------------------------------------------------------------------------------------------------- #
#  Image creation functions
# --------------------------------------------------------------------------------------------------------------------- #

def addConeToImage( eventData , Z):

  return(Z)

def createImage( imageableEvents , resolution=[1000,1000] ):
  Z = np.zeros(resolution[0] , resolution[1])
  for event in imageableEvents:
    Z  = addConeToImage(event , resolution)

  return(Z)

# --------------------------------------------------------------------------------------------------------------------- #
#  Image evaluation functions
# --------------------------------------------------------------------------------------------------------------------- #

def fitGaussianToImage( imageMatrix ):
  pass

def getImageEntropy( imageMatrix ):
  pass

# --------------------------------------------------------------------------------------------------------------------- #
#  Detector setup file reader
# --------------------------------------------------------------------------------------------------------------------- #

def setDetectors( dataFile ):
  """
  Reads a .ini detector file, stores detector data in Detector objects
  returns a dictionary mapping MCNP cell to corresponding detector object
  dataFile must point to an ini file that has a section for each detector
  in the problem (e.g. each cell apperaing in the collision file).
  @params:
      dataFile    - Required  : path to .ini detector setup file

  Each detector field in the input file must have the following fields
   -  cell               = cell # in MCNPX-PoliMi
   -  x                  = x at center of cell in cm
   -  y                  = y at center of cell in cm
   -  z                  = z at center of cell in cm
      ---> x,y,z should be from the middle of the whole detector system
      ---> if you leave out x,y, or z, that signifies that there's a
           reconstruction process. In this case the x,y, or z from the
           collision file is used. The appropriate xdev, ydev or zdev can
           be used here to sample an uncertainty inherent to the reconstruction
           xdev, ydev, zdev MUST always be present
   -  x_dev              = uncertainty in the x direction in cm
   -  y_dev              = uncertainty in the y direction in cm
   -  z_dev              = uncertainty in the z direction in cm
   -  time_resolution    = time resolution in seconds
   -  a [MeV]
   -  b [MeV ^ (1/2)]
   -  c [MeV ^ (-1/2)]
     --> FWHM [MeV] = a + b * sqrt( E + c * E^2 )

   And optional fields (comment them out if you dont want them):

   --> If any one of cellMap, kmap or kdev are present, they ALL must be present

   - cellMap = ci , cj , ck , cl , ...
      Where c is a cell number. This gives the mapping from cell to k or kdev for a
      specific detector. e.g. ci corresponds to k1 +/- s1, cj -> k2 +/- s2 , ...

   - kmap = k1 , k2 , k3 , ... , kj-1 , kj+1 ,  ... , kn
       --> for the jth detector out of n detectors, kmap gives the optimal k value for
           the kinematic approximation in double compton scatter imaging.
           The optimal k is for each detector-detector pair, where the 1st detector is
           the detector its being specified for, and the second is the one in the cell
           specified by the item in cellMap at the same element as k

   - kdev = s1 , s2 , s3 , ... , sj-1 , sj+1 ,  ... , sn
       --> the uncertainty of the k distribution for each detector to detector scattering mode
           Every detector that has a k defined must have an uncertainty defined

        If kmap and kdev are not used, default k = 2 with 0 uncertainty

   - E2_cutoff = the cutoff energy for second scatter events in organics. Default is 0 MeV.

   These last 3 fields can apply to any detector, and are concerned with double Compton
   scatter events, where the first scatter happend in the detector they are specified for,
   and the second scatter happens in eany other detector. For compton scatter -> full energy
   depsoiton events.
   kmap, kdev, and E2_cutoff are not used.

   - solidAngle = the solid angle subtended by the detector from the source in collision file.
      --> If this field is included, the GammaImage.py can calculate an intrinsic image
          efficiency
  """
  config = configparser.ConfigParser()
  config.read( dataFile )

  detectors = {}

  for key , detector in config.items():
    if 'Detector' in key:
      # parse optional kinematic arguments
      kmap = {}
      kdev = {}
      e2 = 0
      if 'kmap' in detector:
        cellmap  = [ int(x.strip()) for x in detector['cellmap'].split(",") ]
        k        = [ int(x.strip()) for x in detector['kmap'].split(",") ]
        kd       = [ int(x.strip()) for x in detector['kdev'].split(",") ]
        for i , cell in enumerate(cellmap):
          kmap[cell] = k[i]
          kdev[cell] = kd[i]

      if 'E2_cutoff' in detector:
        e2 =float( detector['E2_cutoff'])

      # construct detector object at element in the dictionary mapping from the cell the detector corresponds to
      detectors[detector['cell']] = Detector( float(detector['x_dev']) , float(detector['y_dev']) ,
                                              float(detector['z_dev'])                            ,
                                              float(detector['time_resolution']) * 1.0E-8         ,
                                              float(detector['a'])     , float(detector['b'])     ,
                                              float(detector['c'])     , k=kmap                   ,
                                              kdev=kdev                , E2_cutoff=e2
                                            )

      # parse optional positonal arguments
      if 'x' in detector:
        detectors[detector['cell']].x= float(detector['x'])

      if 'y' in detector:
        detectors[detector['cell']].y= float(detector['y'])

      if 'z' in detector:
        detectors[detector['cell']].z= float(detector['z'])

  return(detectors)

# --------------------------------------------------------------------------------------------------------------------- #
#  Beautification tools
# --------------------------------------------------------------------------------------------------------------------- #

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# --------------------------------------------------------------------------------------------------------------------- #
#   Main function
# --------------------------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
  # read in collision file
  configFile       = sys.argv[1]

  config = configparser.ConfigParser()
  config.read( configFile )

  # get path to collision file
  colFile = config['Directories']['collision_file_path']

  # read in detector setup file and create dictionary from cells to detector objects
  detectorFile  = config['Directories']['detector_file']
  detectors = setDetectors(detectorFile)

  # read pixel resolution
  resolution = [ int(x.strip()) for x in config['General']['resolution'].split(",") ]

  # set up discrete ordinate coordinate system
  Azimuth = np.linspace(-180,180,resolution[0])
  Altitude = np.linspace(-90,90,resolution[1])
  Theta,Phi = np.meshgrid(Azimuth,Altitude)

  # read in the maximum number of events to use
  maxNumEvents =  int( config['General']['max_num_events'] )

  # read imageable events from collision file
  imageData , efficiency , unc = getImageDataFromCollisonFile(colFile, detectors, numLines=maxNumEvents, loud=True)

  # test a single cone
  Z = np.zeros(resolution)
  Z += addConeToImage(imageData[0] , Z)
  plotZ(Z , 1)

#  if maxNumEvents > len(imageableEvents):
#    maxNumEvents = len(imageableEvents)

#  imageMatrix = createImage(imageableEvents[0:maxNumEvents] , resolution=[200 , 200] )
#  center , covarianceDeterminant , fwhmTheta , fwhmPhi = fitGaussianToImage( imageMatrix )
#  entropy = getImageEntropy( imageMatrix )


