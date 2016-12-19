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


def drawOverlayRegion(img,opticalFlow,topLeft,bottomRight):
    nBins = conf.nBins
    x = topLeft[0]
    y = topLeft[1]

    region = opticalFlow[y:y+conf.OFSquareSize,x:x+conf.OFSquareSize,:]

    regionU = region[:,:,1]
    regionV = region[:,:,2]

    histogram = []

    u = np.less(regionU,32)
    v = np.less(regionV,32)
    histogram.append(sum(sum(np.bitwise_and(u,v))))

    u = np.greater_equal(regionU,128)
    v = np.greater_equal(regionV,128)
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.greater_equal(v,u)))))
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.less(v,u)))))

    v = np.bitwise_and(np.less(regionV,128),np.greater_equal(regionV,32))
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.less(v,u)))))
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.greater_equal(v,u)))))

    u = np.bitwise_and(np.less(regionU,128),np.greater_equal(regionU,32))
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.greater_equal(v,u)))))
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.less(v,u)))))

    v = np.greater_equal(regionV,128)
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.less(v,u)))))
    histogram.append(sum(sum(np.bitwise_and(np.bitwise_and(u,v),np.greater_equal(v,u)))))

    histosum = sum(histogram)
    histogram = [255*(float(el)/histosum) for el in histogram]

    cv2.rectangle(img,topLeft,bottomRight,(255,255,255))
    radius = ((bottomRight[0] - topLeft[0])/2,(bottomRight[1] - topLeft[1])/2)
    center = (topLeft[0] + radius[0], topLeft[1] + radius[1])
    smallRadius = radius[0]/3
    bigRadius = 2*radius[0]/3


    for i in range(0,nBins-1):
        cv2.ellipse(img,tuple(center),(bigRadius,bigRadius),0,startAngle=i*360/(nBins-1), endAngle=(i+1)*360/(nBins-1), color = (0,0,histogram[i+1]),thickness = -1)
    cv2.circle(img,tuple(center),radius[0]/3,(0,0,histogram[0]),-1)
    for i in range(0,nBins-1):
        pt1 = (int(smallRadius*np.cos(np.deg2rad(i*360/(nBins-1)))),int(smallRadius*np.sin(np.deg2rad(i*360/(nBins-1)))))
        pt1 = (pt1[0] + center[0],pt1[1] + center[1])
        pt2 = (int(bigRadius*np.cos(np.deg2rad(i*360/(nBins-1)))),int(bigRadius*np.sin(np.deg2rad(i*360/(nBins-1)))))
        pt2 = (pt2[0] + center[0],pt2[1] + center[1])
        cv2.line(img,pt1,pt2,(255,255,255))
    cv2.circle(img,tuple(center),smallRadius,(255,255,255))
    cv2.circle(img,tuple(center),bigRadius,(255,255,255))

def drawOverlay(img,opticalFlow):
    vRange = range(0,img.shape[0],conf.OFSquareSize)
    hRange = range(0,img.shape[1],conf.OFSquareSize)

    for h in hRange:
        for v in vRange:
            topLeft = (h,v)
            bottomRight = (h+conf.OFSquareSize,v+conf.OFSquareSize)
            drawOverlayRegion(img,opticalFlow,topLeft,bottomRight)


def plotOpticalFlowHistogram(imageFile,opticalFlowFile):

    img = cv2.imread(imageFile)
    opticalFlow = cv2.imread(opticalFlowFile)
    u = opticalFlow[:,:,1]
    v = opticalFlow[:,:,2]
    oF = opticalFlow[:,:,0]
    height = opticalFlow.shape[0]
    width = opticalFlow.shape[1]


    newWidth =  int(pow(2,np.ceil(np.log2(width))))
    newHeight = int(pow(2,np.ceil(np.log2(height))))
    newImg = cv2.resize(img,(newWidth,newHeight))
    newOF = cv2.resize(opticalFlow,(newWidth,newHeight))

    overlay = newImg.copy()
    drawOverlay(overlay,newOF)

    alpha = 0.5
    output = newImg.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    result = cv2.resize(output,(width,height))

    cv2.imshow('test',result)
    cv2.waitKey(0)


if __name__ == "__main__":
    #opticalFlowMetrics()
    plotOpticalFlowHistogram("./datasets/colored_0/000125_10.png","./datasets/flow_noc/000125_10.png")
