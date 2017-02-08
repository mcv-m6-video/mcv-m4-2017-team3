import cv2
import week5configuration as finalConf
import numpy as np

def getSingleFrame(dataFile,frameNumber,convertColor = True):
    if isinstance(dataFile,list):
        frame =  cv2.imread(dataFile[frameNumber])
    else:
        dataFile.set(1,frameNumber)
        ret,frame = dataFile.read()
    if finalConf.colorSpace != "BGR" and convertColor:
        frame = cv2.cvtColor(np.asarray(frame),finalConf.colorSpaceConversion[finalConf.colorSpace])
    return frame

def getFrameAndPrevious(dataFile,frameNumber,convertColor = True):
    return getSingleFrame(dataFile,frameNumber,convertColor),getSingleFrame(dataFile,frameNumber-1,convertColor)

def getFrameAndNext(dataFile,frameNumber,convertColor = True):
    return getSingleFrame(dataFile,frameNumber,convertColor),getSingleFrame(dataFile,frameNumber+1,convertColor)
