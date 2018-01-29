import numpy as np
import os

class DataLoader:
    DAFCA_STD = 0
    DAFCA_DPP_MIXED = 1
    DAFCA_DPP_LIST = 2
    
    def __init__(self, fileName, dataType, numSamples):#num Samples is the number of samples in a waveform, not number of waves
        self.fileName = fileName
        self.dataType = dataType
        self.numSamples = numSamples
        if dataType == DataLoader.DAFCA_STD:
            self.blockType = np.dtype([('EventSize',(np.int32,1)),
                                        ('BoardID',(np.uint32,1)),
                                        ('Pattern',(np.uint32,1)),
                                        ('Channel',(np.int32,1)),
                                        ('EventCounter',(np.uint32,1)),
                                        ('TimeTag',(np.uint32,1)),
                                        ('Samples',np.uint16,numSamples)])
        elif dataType == DataLoader.DAFCA_DPP_MIXED:
           self.blockType = np.dtype([('EventSize',(np.int32,1)),
                                       ('Format',(np.int32,1)),
                                       ('Channel',(np.int16,1)),
                                       ('Baseline',(np.int16,1)),
                                       ('ShortGate',(np.int16,1)),
                                       ('LongGate',(np.int16,1)),
                                       ('TimeTag',(np.int32,1)),
                                       ('Extras',(np.int32,1)),
                                       ('Samples',np.uint16,numSamples)])
        elif dataType == DataLoader.DAFCA_DPP_LIST:
           self.blockType = np.dtype([('EventSize',(np.int32,1)),
                                       ('Format',(np.int32,1)),
                                       ('Channel',(np.int16,1)),
                                       ('Baseline',(np.int16,1)),
                                       ('ShortGate',(np.int16,1)),
                                       ('LongGate',(np.int16,1)),
                                       ('TimeTag',(np.int32,1)),
                                       ('Extras',(np.int32,1))])
        self.location = 0
    
    def GetNumberOfWavesInFile(self):
        return int(os.path.getsize(self.fileName) / self.blockType.itemsize)
        
    def LoadWaves(self, numWaves):
        """Loads numWaves waveforms. If numWaves == -1, loads all waveforms in the file"""
        fid = open(self.fileName, "rb")
        fid.seek(self.location, os.SEEK_SET)
        self.location += self.blockType.itemsize * numWaves
        return np.fromfile(fid, dtype = self.blockType, count=numWaves)
    
    def Rewind(self):
        self.location = 0