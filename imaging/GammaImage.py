#!/usr/bin/env python

"""
GammaImager.py - a script to project a simulated image of a gamma source from a PoliMi collision file
resulting from a n MCNPX-PoliMi simulation of a mixed Compton system
 - e.g. a system where either double compton scatter, compton scatter -> full energy deposition, or both
   event types are possible

"""

from matplotlib  import pyplot as plt

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
  """
  Each  instance of the Detector class holds the characteristics of a detector defined in the detector
  setup file, corresponding to an MCNP cell in the collision file. If the position of an event within
  the detector is taken just as the center of the detector (or the posiiton along an axis is taken as
  the center of the detector along that axis), the center of the detector x,y,z (or center along that axis
  or axes, e.g. just x or both y and z) will be set in the set detector function. For positions set (e.g. x),
  the uncertainty (e.g. xdev) corresponds to the size of the detector along that axis, as that is the uncertainty.
  Otherwise, if the position of an event is reconstructed somehow within the detector, then the detector object
  does not require a position argument along that axis, and the position uncertainty along that axis is the
  characteristic uncertainty of the reconstruction method, as set in the detector file.
  """
  def __init__(self  , xdev , ydev , zdev , tdev , a , b , c , k={} , kdev={} , E2_cutoff=0 ):
    """
    Constructor
    @params:
        xdev                 - Required  : the uncertainty in the x position of an event in the detector
        ydev                 - Required  : the uncertainty in the y position of an event in the detector
        zdev                 - Required  : the uncertainty in the z position of an event in the detector
        tdev                 - Required  : the time resolution of the detector in seconds
        a                    - Required  : energy fwhm [MeV] = a + b * sqrt( E_i + c * E_i^2 ), where E_i is the
                                           incident particle energy
        b                    - Required  : see a
        c                    - Required  : see a
        k                    - Optional  : kinematic approximation for double scatter events: k = (E_i -E_1)/E_2
                                           Dictionary mapping from cell of intial scatter to optimal k
        kdev                 - Optional  : the uncertainty in the kineamtic approximation
                                           Dictionary mapping from cell of intial scatter to uncertainty
        E2_Cutoff            - Optional  : The cutoff deposited energy for second scatters in the detector,
                                            below which, events are ignored.
    """
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
    """
    Sample location of event in detector - only used for reconstruction
    @params:
        x                 - Required  : the real position within the detector from the collison file
    """
    return( np.random.normal(x , self.xdev))

  def sampleY(self , y):
    """
    Sample location of event in detector - only used for reconstruction
    @params:
        y                 - Required  : the real position within the detector from the collison file
    """
    return( np.random.normal(y , self.ydev))

  def sampleZ(self , z):
    """
    Sample location of event in detector - only used for reconstruction
    @params:
        z                 - Required  : the real position within the detector from the collison file
    """
    return( np.random.normal(z , self.zdev))

  def getEnergyFWHM(self, Ei ):
    """
    energy fwhm [MeV] = a + b * sqrt( E_i + c * E_i^2 ), where E_i is the incident particle energy
    @params:
        Ei                 - Required  : incident particle energy
    """
    return( self.a + self.b * np.sqrt( Ei + self.c * Ei**2 ) )

  def sampleEnergy(self, Ei , Edep ):
    """
    Sample deposited energy
    @params:
        Ei                 - Required  : incident particle energy
        Edep               - Required  : real deposited energy from collision file
    """
    return( np.random.normal( Edep , 0.2355 * self.getEnergyFWHM(Ei) ) )

  def sampleTime(self, t ):
    """
    Sample event time
    @params:
        t                 - Required  : real time from collision file
    """
    return( np.random.normal( t , self.tdev ))


# --------------------------------------------------------------------------------------------------------------------- #
#  event processing
# --------------------------------------------------------------------------------------------------------------------- #

class eventData:
  """
  The minimum information needed for backprojection is the unit direction vector between events, and the Compton cone
  angle, as well as the uncertainty in those values. This class acts a simple data structure that holds that information.

  """
  def __init__(self , theta , phi , deltaTheta , deltaPhi , angle , angleUncertainty):
    """
    Constructor
    @params:
        theta               - Required  : polar angle of projected unit vector
        phi                 - Required  : azimuthal angle of projected unit vector
        deltaTheta          - Required  : uncertainty in theta
        deltaPhi            - Required  : uncertainty in phi
        angle               - Required  : backprojected cone angle cosine
        angleUncertainty    - Required  : uncertainty in angle cosine
    """
    self.theta      = theta * 90 / (np.pi) - 45
    self.phi        = phi   * 90 / (np.pi)
    self.deltaTheta = deltaTheta * 90 / ( np.pi)
    self.deltaPhi   = deltaPhi   * 90 / (np.pi)
    self.angle      = np.arccos(angle) * 90 / ( np.pi)
    self.deltaAngle = np.sqrt( angleUncertainty**2 / ( 1 - angle**2)  ) * 90 / ( np.pi)

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

def unitProjection( pos1 , pos2 , un1=0 , un2=0 , error=False):
  """
  Given two positions in R^3 and their associated uncertainties, unitProjection
  projects the unit vector along the axis between their uncertainties onto the unit
  sphere {theta , phi}, and analytically propagates uncertainty through the
  vector calculations, normalization, and projection to get uncertainties in
  theta and phi.
  @params:
      pos1    - Required  : position of 1st interaction [x,y,z]
      pos2    - Required  : position of 2nd interaction [x,y,z]
      un1     - Required  : uncertainty in 1st position [delta_x , delta_y , delta_z]
      un2     - Required  : uncertainty in 2nd position [delta_x , delta_y , delta_z]
  """
  # find unit vector
  vec =  np.array( [ pos1[0] - pos2[0] , pos1[1] - pos2[1] , pos1[2] - pos2[2] ] )
  #vec =  vec - sourceLocation
  [x,y,z] = vec / np.linalg.norm(vec)


  # project onto unit sphere
  theta = np.arccos(x)
  phi   = np.arcsin(z)

  if error == True:
    # propagate uncertainty in pos1, pos2 to find uncertainty in x
    varx =        ( un1[0]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
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

    # propagate uncertainty in pos1, pos2 to find uncertainty in z
    varz =        ( un1[0]**2 * ( ( (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2 )**2 /
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

    # propagate uncertainty in x into theta
    devTheta = varx / np.fabs((1 - x**2))
    # propagate uncertainty in z into phi
    devPhi   = varz / np.fabs((1 - z**2))

    return(theta , phi , devTheta , devPhi)

  else:
    return(theta, phi)

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
  angle =  1 - (0.511 / E2 - 0.511 / (E1 + E2) )
  # propagate uncertainty in energy
  uncertainty = np.sqrt( devE1**2 * 0.261121 / (E1 + E2)**4 +
                         devE2**2 * ( 0.511 / (E2**2) - 0.511 / (E1 + E2)**2 )**2
                       )
  return(angle , uncertainty)



# --------------------------------------------------------------------------------------------------------------------- #
#  Collision file parsing function
# --------------------------------------------------------------------------------------------------------------------- #

def getImageDataFromCollisonFile( collisionFile , detectors , numLines=1000 , loud=False, centerVec=[0,1,0] ):
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
    # if it is a photon that compton scatters in a cell, and the next line is the same photon
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

      # get projecion of unit vector bt scattering events onto unti sphere {theta,phi}
      # as well as analytically propagated uncertainties
      theta, phi , deltaTheta , deltaPhi  = unitProjection( pos1 , pos2 , un1=unc1 , un2=unc2 , error=True)

      # check if it is a scatter -> scatter or scatter -> capture event
      # and if so, if the second energy deposited is above the cutoff for its detector
      if nextline[3] == "1" and ( float(nextline[6]) >= tmpDet2.E2_cutoff ):
        # double scatter event
        cos , unc = getDoubleScatterConeAngleCos(   tmpDet1.sampleEnergy( float( line[-1])     , float( line[6] ) )            ,
                                                    tmpDet2.sampleEnergy( float( nextline[-1]) , float( nextline[6] ) )    ,
                                                    0.2355 * tmpDet1.getEnergyFWHM( float(line[-1]) )     ,
                                                    0.2355 * tmpDet2.getEnergyFWHM( float(nextline[-1]) ) ,
                                                    k=tmpDet1.k[    int(nextline[5]) ] ,
                                                    kdev=tmpDet1.kdev[ int(nextline[5]) ]
                                                )

      else:
        cos , unc = getFullDepConeAngleCos(   tmpDet1.sampleEnergy( float( line[-1])     , float( line[6] ) )            ,
                                              tmpDet2.sampleEnergy( float( nextline[-1]) , float( nextline[6] ) )    ,
                                              0.2355 * tmpDet1.getEnergyFWHM( float(line[-1]) )     ,
                                              0.2355 * tmpDet2.getEnergyFWHM( float(nextline[-1]) ) ,
                                          )
      #append to imageData
      if (cos >= -1 and cos <= 1 ):
        imageData.append( eventData(theta , phi , deltaTheta , deltaPhi ,  cos  , unc ) )

    # update progress bar following new load
    if loud == True:
        print_progress(i, numLines, prefix='Progress', suffix='', decimals=1, bar_length=100)

  # divide the amount of imageable events (double scatter or scatter -> absorption) by the
  # number of histories used to get image efficiency
  numImageableEvents = len(imageData)
  numSourceHistories = int(nextline[0])
  # Wilson score interval for binomial estimation - see wikipedia
  imageEfficiency = numImageableEvents / numSourceHistories
  efficiencyUncertainty =  np.sqrt(imageEfficiency * ( 1 - imageEfficiency) / numSourceHistories)

  if loud == True:
    print( "\n")
    print("Found " + str(numImageableEvents) + " imageable events, out of " + nextline[0] +
        " source particle histories checked, for an image efficiency of " + '{:1.3E}'.format(imageEfficiency) +
          " +/- " +  '{:1.3E}'.format(efficiencyUncertainty) + "\n\n"
         )

  return( imageData , imageEfficiency , efficiencyUncertainty )


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

def booleanize(value):
    """Return value as a boolean."""
    true_values = ("yes", "true", "1" , "y" , "Yes" , "Y")
    false_values = ("no", "false", "0" , "n" , "N" , "No")
    if isinstance(value, bool):
        return value
    if value.lower() in true_values:
        return True
    elif value.lower() in false_values:
        return False
    raise TypeError("Cannot booleanize ambiguous value '%s'" % value)

# --------------------------------------------------------------------------------------------------------------------- #
#   Image plotting and writing functions
# --------------------------------------------------------------------------------------------------------------------- #

def plotZ(Theta, Phi, Z, counter , resolution  , sourceLocation=[0,0] , sideHists=False ):
  f = plt.figure(1 , figsize=(8,4))

  if( sideHists == True):
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.03

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h , width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axHistx.xaxis.set_visible(False)
    axHisty.yaxis.set_visible(False)

    axHisty.set_ylim(-90 , 90 )
    axHistx.set_xlim(-180 , 180 )

    axScatter = plt.axes(rect_scatter)
    axScatter.pcolormesh(Theta,Phi,Z, cmap='jet') #http://matplotlib.org/users/colormaps.html
    axScatter.set_xlabel(r"Azimuthal Angle [$\theta$]")
    axScatter.set_ylabel(r"Altitude Angle [$\phi$]")
    axScatter.set_xlim(-180 , 180 )
    axScatter.set_ylim(-90 , 90 )
    axScatter.scatter( sourceLocation[0] , sourceLocation[1]  , s=50 , facecolors='none',
             edgecolors='r' , label='actual source location')
    axScatter.xaxis.labelpad = -1
    axScatter.legend()
    plt.text( 205 , 130 ,  "Cones: "+str(counter)  ,fontsize=16 )

    # integrate Z over each dimension
    Azimuth = np.linspace(-180  , 180  ,resolution[0])
    Altitude = np.linspace(-90 , 90 ,resolution[1])
    thetaDist = np.sum(Z , axis=0)
    thetaDist = thetaDist / np.sum(thetaDist)
    phiDist   = np.sum(Z , axis=1)
    phiDist   = phiDist / np.sum(phiDist)

    adjustment = (max(phiDist) - min(phiDist))*0.05
    axHistx.set_ylim(min(thetaDist) , max(thetaDist)*1.1 )
    axHisty.set_xlim( min(phiDist) + adjustment , max(phiDist)*1.1 )

    # get max theta , phi
    thetaMax , phiMax = Azimuth[ np.argmax(thetaDist) ] , Altitude[ np.argmax(phiDist) ]

    # plot integrated theta, phi distributions
    axHistx.plot( Azimuth   , thetaDist  , linewidth=3 )
    axHisty.plot( phiDist   , Altitude   , linewidth=3 )
    axHistx.fill_between( Azimuth   , thetaDist ,  facecolor='b')
    axHisty.fill_between( np.append( phiDist , 0)   , np.append( Altitude , 0)  ,  facecolor='b')

    # plot crosshairs over centers of the distribution
    axHisty.plot( [0, max(phiDist)] , [ phiMax , phiMax ] , 'r--'   )
    axHistx.plot( [thetaMax , thetaMax ] , [0, max(thetaDist)]   , 'r--'   )
    axScatter.plot( [-180 , 180 ] , [phiMax , phiMax] , 'r--')
    axScatter.plot( [thetaMax , thetaMax] , [-90 , 90] , 'r--')

  else:
    plt.title("Cones: "+str(counter))
    plt.pcolormesh(Theta,Phi,Z, cmap='jet') #http://matplotlib.org/users/colormaps.html
    plt.set_xlabel(r"Azimuthal Angle [$\theta$]")
    plt.set_ylabel(r"Altitude Angle [$\phi$]")
    plt.set_xlim(-180 , 180 )
    plt.set_ylim(-90 , 90 )
    plt.scatter( sourceLocation[0] , sourceLocation[1]  , s=50 , facecolors='none',
             edgecolors='r' , label='actual source location')

    plt.legend()

  plt.show()
  plt.close()

def writeOut(Z , fname):
    np.savetxt(fname , Z)

# --------------------------------------------------------------------------------------------------------------------- #
#  Image creation functions
# --------------------------------------------------------------------------------------------------------------------- #

def addConeToImage( e , Z, theta , phi):
  # analytically calculated variance
  avar  =  4 * e.angle**2 *  e.deltaAngle**2
  newmat = np.exp(  -( e.angle**2- (theta - e.theta)**2 - (phi - e.phi)**2)**2 /
                      ( avar + 4 * (theta - e.theta)**2 * e.deltaTheta**2 +
                              4 * (phi - e.phi)**2 * e.deltaPhi**2 )
                 )
  # analytically calculated normalizing factor
  Z += newmat
  return(Z)

def createImage( imageableEvents , res=[1000,1000] , loud=False ):
  if loud == True:
    print( "Now constructing image from " + str(len(imageableEvents)) +  " events")
    print_progress(0, len(imageableEvents), prefix='progress', suffix='', decimals=1, bar_length=100)

  Z = np.zeros(resolution)
  for i , event in enumerate(imageableEvents):
    Z  = addConeToImage(event , Z , theta , phi)
    if loud == True:
      print_progress(i+1, len(imageableEvents), prefix='progress', suffix='', decimals=1, bar_length=100)

  return(Z)

# --------------------------------------------------------------------------------------------------------------------- #
#  Image evaluation functions
# --------------------------------------------------------------------------------------------------------------------- #

def fitGaussianToImage( imageMatrix ):
  pass

def getImageEntropy( imageMatrix ):
  pass


# --------------------------------------------------------------------------------------------------------------------- #
#   Main function
# --------------------------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
  # read in collision file
  configFile       = sys.argv[1]

  config = configparser.ConfigParser()
  config.read( configFile )

  # get loud boolean
  loudPrint = booleanize(config['General']['loud'])

  # get path to collision file
  colFile = config['Directories']['collision_file_path']

  # read in detector setup file and create dictionary from cells to detector objects
  detectorFile  = config['Directories']['detector_file']
  detectors = setDetectors(detectorFile)

  # read pixel resolution
  resolution = [ int(x.strip()) for x in config['General']['resolution'].split(",") ]

  # read gournd truth source location
  location = [ float(x.strip()) for x in config['General']['source_location'].split(",") ]

  #project source location onto unit sphere
  stheta , sphi  = unitProjection( location , [0,0,0] , error=False )

  # set up discrete ordinate coordinate system
  Azimuth = np.linspace(-180  , 180  ,resolution[0])
  Altitude = np.linspace(-90 , 90 ,resolution[1])
  theta,phi = np.meshgrid(Azimuth,Altitude)

  # read in the maximum number of events to use
  maxNumEvents =  int( config['General']['max_num_events'] )

  # read imageable events from collision file
  imageData , efficiency , unc = getImageDataFromCollisonFile(colFile, detectors, numLines=maxNumEvents, loud=loudPrint)

  # test a single cone
  Z = np.zeros(resolution)
  Z += addConeToImage(imageData[0] , Z , theta , phi)
  plotZ(theta , phi , Z , 1 , resolution,sourceLocation=[stheta , sphi] , sideHists=True)

  imageMatrix = createImage(imageData, res=resolution[:] , loud=loudPrint)
  if loudPrint == True:
    print( "\n")
    print("Now plotting image \n")
  plotZ(theta , phi , imageMatrix ,  len(imageData) , resolution, sourceLocation=[stheta , sphi] , sideHists=True)
#  center , covarianceDeterminant , fwhmTheta , fwhmPhi = fitGaussianToImage( imageMatrix )
#  entropy = getImageEntropy( imageMatrix )


