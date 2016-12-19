import cv2
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import json
import math
import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import configuration as conf


def to_percent(y, position):
    s = str(100 * y)

    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def msen(resultOF, gtOF):
    errorVector = []
    correctPrediction = []

    uResult = []
    vResult = []
    uGT = []
    vGT = []
    imageToReconstruct = []

    validGroundTruth = []

    # flow_u(u, v) = ((float)I(u, v, 1) - 2 ^ 15) / 64.0;
    # flow_v(u, v) = ((float) I(u, v, 2) - 2 ^ 15) / 64.0;
    # valid(u, v) = (bool)I(u, v, 3);
    for pixel in range(0,resultOF[:,:,0].size):
        uResult.append( ((float)(resultOF[:,:,1].flat[pixel]) - math.pow(2, 15) ) / 64.0 )
        vResult.append(((float)(resultOF[:,:,2].flat[pixel])-math.pow(2, 15))/64.0)
        uGT.append(((float)(gtOF[:,:,1].flat[pixel])-math.pow(2, 15))/64.0)
        vGT.append(((float)(gtOF[:,:,2].flat[pixel])-math.pow(2, 15))/64.0)
        validGroundTruth.append( gtOF[:,:,0].flat[pixel] )

    for idx in range(len(uResult)):
        if validGroundTruth[idx] == 0:
            imageToReconstruct.append(0)
            continue
        else:
            squareError = math.sqrt(math.pow((uGT[idx] - uResult[idx]), 2) + math.pow((vGT[idx] - vResult[idx]), 2))

        errorVector.append(squareError)
        imageToReconstruct.append(squareError)

        if (squareError > 3):
            correctPrediction.append(0)
        else:
            correctPrediction.append(1)

    error = (1 - sum(correctPrediction)/(float)(sum(validGroundTruth))) * 100;

    errorArray = np.asarray(errorVector)

    return errorArray, error, imageToReconstruct


def opticalFlowMetrics():
    gtOFFiles = sorted(glob.glob(conf.gtOFFolder + "*"))
    resultsOFFiles = sorted(glob.glob(conf.resultsOFFolder + "*"))

    for idx in range(len(resultsOFFiles)):
        OFimage = cv2.imread(resultsOFFiles[idx], -1)
        OFgt = cv2.imread(gtOFFiles[idx], -1)

        msenValues, error, image = msen(OFimage, OFgt)
        plt.hist(msenValues, bins=25, normed=True)
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel('MSEN value')
        plt.ylabel('Number of Pixels')
        plt.title("%s Histogram. \n Percentage of Erroneous Pixels in Non-occluded areas (PEPN): %d %%" %(conf.imagesOF[idx], error))
        plt.show()

        r, c, d = OFimage.shape;
        visualizeOF = np.reshape(image, (r,c))
        # HSV Color Space
        channelH = (visualizeOF / visualizeOF.max()) * 360
        channelS = np.ones((r, c)) * 100
        channelV = np.ones((r, c)) * 100

        hsvImage = np.stack([channelV, channelH, channelS], axis=-1)
        bgrImage = cv2.cvtColor(hsvImage.astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.imshow('finalImage', cv2.cvtColor(bgrImage.astype(np.uint8), cv2.COLOR_BGR2HSV))

        # channelB = (((visualizeOF / visualizeOF.max()) * 255 ) - np.ones((r, c)) * 255)
        # channelB = np.absolute(channelB)
        # channelG = np.ones((r, c)) * 255
        # channelR = channelB
        #
        # bgrImage = np.stack([channelB, channelG, channelR], axis=-1)
        # cv2.imshow('finalImage', cv2.cvtColor(bgrImage.astype(np.uint8), cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)



if __name__ == "__main__":
    opticalFlowMetrics()