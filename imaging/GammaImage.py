#!/usr/bin/env python

"""
GammaImager.py - a script to project a simulated image of a gamma source from a PoliMi collision file
resulting from a n MCNPX-PoliMi simulation of a mixed Compton system
 - e.g. a system where either double compton scatter, compton scatter -> full energy deposition, or both
   event types are possible

"""
import numpy as np
import sys

__author__ = "Kyle Beyer"
__version__ = "1.0.1"
__maintainer__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"
__status__ = "Development"

class Detector:
  def __init__(self ,  ZDEV , TDEV , a , b , c , kmap , E2Cutoff , x , y , dimensions):
    Zdev = ZDEV
    Tdev = TDEV
    a = a
    b = b
    c = c
    k = kmap
    E2Cutoff = E2Cutoff
    dimensions = dimensions # rectangular base of detector [xwidth , ywidth ]

def getBestKfromCell( cell1 , cell2 ):
  global detectors
  return( detectors[cell1].k[cell2] )

def getABCfromCell( cell ):
  global detectors
  return( [ detectors[cell].a , detectors[cell].b , detectors[cell].c ] )

def getTDEVfromCell( cell ):
  global detectors
  return( [ detectors[cell].Tdev  ] )

def getZDEVfromCell( cell ):
  global detectors
  return( [ detectors[cell].Zdev  ] )

def getXfromCell( cell ):
  global detectors
  return( [ detectors[cell].x  ] )

def getYfromCell( cell ):
  global detectors
  return( [ detectors[cell].y  ] )

def getConeAngleUncertaintyFromCell( cell1 , cell2 , event ):
  abc1 = getABCfromCell( event.cell[0] )
  abc2 = getABCfromCell( event.cell[1] )
  dev1 = 0.235 * (abc1[0] + abc1[1] * np.sqrt( event.Ei1 + abc1[2] * event.Ei1**2 ))
  dev2 = 0.235 * (abc2[0] + abc2[1] * np.sqrt( event.Ei2 + abc2[2] * event.Ei2**2 ))
  return( np.sqrt( dev1**2 * 0.261121 / event.E1**4 + dev2**2 * 0.261121 / event.E2**4 ) )

def getXYZuncertaintyFromCell( cell ):
  return([ detectors[cell].dimensions[0]/2  , detectors[cell].dimensions[1]/2 , detectors[cell].Zdev ])

class ImageableEvent:
    def __init__(self , eventLog ):
      self.cell1 = eventLog.cell[0]
      self.cell2 = eventLog.cell[1]
      self.x1 = getXfromCell( eventLog.cell[0] )
      self.x2 = getXfromCell( eventLog.cell[1] )
      self.y1 = getYfromCell( eventLog.cell[0] )
      self.y2 = getYfromCell( eventLog.cell[1] )
      self.E1 = eventLog.Edep[0]
      self.E2 = eventLog.Edep[1]
      self.z1 = eventLog.z[0]
      self.z2 = eventLog.z[1]
      self.t1 = eventLog.t[0]
      self.t2 = eventLog.t[1]

    def applyZuncertainty(self):
      self.z1 = np.random.norm( self.z1 , getZDEVfromCell(self.cell[0] ) )
      self.z2 = np.random.norm( self.z2 , getZDEVfromCell(self.cell[1] ) )

    def applyEnergyUncertainty(self ,   Ei1 , Ei2):
      abc1 = getABCfromCell( self.cell[0] )
      abc2 = getABCfromCell( self.cell[1] )
      self.E1 = np.random.norm( self.E1 , abc1[0] + abc1[1] * np.sqrt( self.Ei1 + abc1[2] * self.Ei1**2 ) )
      self.E2 = np.random.norm( self.E2 , abc2[0] + abc2[1] * np.sqrt( self.Ei2 + abc2[2] * self.Ei2**2 ) )

    def applyTimeUncertainty(self ):
      self.t1 = np.random.norm( self.t1 , getTDEVfromCell(self.cell[0] ) )
      self.t2 = np.random.norm( self.t2 , getTDEVfromCell(self.cell[1] ) )

    def getVector(self):
      vec =  np.array( [ self.x2 - self.x1 , self.y2 - self.y1 , self.z2 - self.z1] )
      un1 = getXYZuncertaintyFromCell( self.cell1 )
      un2 = getXYZuncertaintyFromCell( self.cell2 )
      uncertainty = [ np.sqrt( self.x2**2 * un2[0]**2 + self.x1**2 * un1[0]**2  ),
                      np.sqrt( self.y2**2 * un2[1]**2 + self.y1**2 * un1[1]**2  ),
                      np.sqrt( self.z2**2 * un2[2]**2 + self.z1**2 * un1[2]**2  ),
                    ]
      return vec , uncertainty

    def getConeAngle(self):
      #only defined for derived classes
      pass

class DoubleScatterEvent(ImageableEvent):
  def getk(self ,  Ei ):
    return( (Ei - self.E1) / self.E2 )

  def getConeAngleCos(self):
    k = getBestKfromCell( self.cell1 , self.cell2 )
    angle = 1 - (0.511 / (k*self.E2 ) - 0.511 / (self.E1 + k * self.E2)  )
    uncertainty = getConeAngleUncertaintyFromCell( self.cell1 , self.cell2  )
    return angle , uncertainty

class FullDepEvent(ImageableEvent):
  def getConeAngleCos(self):
    angle =  1 - (0.511 / (self.E2 ) - 0.511 / (self.E1 + self.E2)  )
    uncertainty = getConeAngleUncertaintyFromCell( self.cell1 , self.cell2  )
    return angle , uncertainty

def parseEvents( events  , abc_organic , abc_inorganic , zdev , tdev):
  imageableEvents = []

  for gammaID , event  in events.items():
    # if it is a double compton scatter event
    if event.eventType == ["1" , "1"] and event.numScat == [0,1]:
      tmp = DoubleScatterEvent(event)
      tmp.applyZuncertainty()
      tmp.applyTimeUncertainty()
      tmp.applyEnergyUncertainty(  event.Ei[0]  , event.Ei[1] )
      imageableEvents.append( tmp )

    # if it is a compton scatter to a full energy event
    elif event.eventType == ["1" , "3"] or event.eventType == ["1" , "3"]:
      tmp = FullDepEvent(event)
      tmp.applyZuncertainty()
      tmp.applyTimeUncertainty()
      tmp.applyEnergyUncertainty( event.Ei[0]  , event.Ei[1] )
      imageableEvents.append( FullDepEvent(event) )

  return( imageableEvents )

def setDetectors( dataFile ):
  # TODO read .ini data file as config and setup each detector
  pass

def getImageableEventsFromCollisonFile( fi , E2Cutoff , detectors ):
  lines =[]
  with open(fi  , "r") as filestream:
    for line in filestream:
      currentline = line.split(" ")
      currentline2 = [x for x in currentline if x != '']
      lines.append(currentline2)

  numTotalEvents = len(lines)
  events = {} # relevant gamma events
  Event = namedtuple("Event" , "Ei Edep t z cell numScat eventType")

  for i,line in enumerate(lines):

    # if its a compton scattered photon
    if line[2] == "2" and line[3] == "1":

      # if particle hasn't been recorded and there are no previous scatters
      if (line[13]=="0") and (line[0] not in events):
        # add the event as the value in the event dictionary for that particle
        events[line[0]] = Event( [ float(line[15]) ] , [float(line[6 ])]  , [ float(line[7]) ] , [ float(line[10]) ] ,
                                 [  int(line[5])]    , [  int(line[13])]  , [   int(line[3]) ]
                               )

      # append the new event to the event dictionary for the particle
      elif (line[0] in events):
        if int(line[13]) == events[line[0]].numScat[-1] + 1:
          events[ line[0] ].Ei.append(      float( line[15] ) )
          events[ line[0] ].Edep.append(    float( line[6]  ) )
          events[ line[0] ].t.append(       float( line[7]  ) )
          events[ line[0] ].z.append(       float( line[10] ) )
          events[ line[0] ].cell.append(      int( line[5]  ) )
          events[ line[0] ].numScat.append(   int( line[13] ) )
          events[ line[0] ].eventType.append( int( line[3]  ) )

    # if its a full energy deposited photon that has previously compton scattered
    elif line[2] == "2" and ( line[3] == "2" or line[3] == "4") and (line[0] in events):
          events[ line[0] ].Ei.append(      float( line[15] ) )
          events[ line[0] ].Edep.append(    float( line[6]  ) )
          events[ line[0] ].z.append(       float( line[10] ) )
          events[ line[0] ].t.append(       float( line[7]  ) )
          events[ line[0] ].cell.append(      int( line[5]  ) )
          events[ line[0] ].numScat.append(   int( line[13] ) )
          events[ line[0] ].eventType.append( int( line[3]  ) )

  # ignore all single scatters and same-cell events
  events =[ e for key , e in events.items() if ( len(e.E) >= 2 and e.cell[0] != e.cell[1] )]

  # calculate imageable event efficiency
  numImageableEvents = len(events)
  imageEfficiency = numImageableEvents / numTotalEvents

  # sort and parse events from log of imageable events
  imageableEvents = parseEvents(events  , E2Cutoff , detectors)

  return( imageEfficiency , imageableEvents )

def createImage( imageableEvents , resolution=[1000,1000] ):
  # TODO return a matrix of the image
  pass

def fitGaussianToImage( imageMatrix ):
  pass

def getImageEntropy( imageMatrix ):
  pass


if __name__ == '__main__':
  colFile = sys.argv[1]
  detectorDataFi = sys.argv[2]

  detectors = setDetectors( detectorDataFi )
  maxNumEvents = sys.argv[3]

  imageEfficiency , imageableEvents  =  getImageableEventsFromCollisonFile( colFile , E2Cutoff , detectors )

  if maxNumEvents > len(imageableEvents):
    maxNumEvents = len(imageableEvents)

  imageMatrix = createImage(imageableEvents[0:maxNumEvents] , resolution=[200 , 200] )
  center , covarianceDeterminant , fwhmTheta , fwhmPhi = fitGaussianToImage( imageMatrix )
  entropy = getImageEntropy( imageMatrix )


