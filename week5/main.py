import cv2
import numpy as np
import week5configuration as finalConf
import glob
import sys
import dataReader
import os
import detectionPipeline as pipeline
import trackingObjects as tracking
import stabizateFrames as stFrame


sys.path.append('../')

import sys
sys.path.append('../')
sys.path.append('../tools/')

results_path = "./results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

if __name__ == "__main__":
    # Get mode from configuration file
    iter = 0 #When to compute the speed
    mode = finalConf.mode
    ID = finalConf.ID
    path = finalConf.folders[ID]
    pathT = finalConf.folders["OwnTrain"]
    alpha = finalConf.OptimalAlphaParameter[ID]
    rho = finalConf.OptimalRhoParameter[ID]

    # Read the video/files and count images
    if ID == "NewVideo":
        data = cv2.VideoCapture(path)
        nFrames = int(data.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = data.get(cv2.CAP_PROP_FPS)
    else:
        if path[-1] == '/':
            data = sorted(glob.glob(path + "*.png"))
            dataT = sorted(glob.glob(pathT + "*.jpg"))
        else:
            data = sorted(glob.glob(path + "/*.png"))
            dataT = sorted(glob.glob(pathT + "/*.jpg"))
        nFrames = len(data)
        nFramesT = len(dataT)
        fps = 30.0 # Educated guess

    # First stage: Training
    if ID == "NewVideo":
        trainingRange = 0 # Define range for your new Video
        testingRange = 0
    else:
        trainingRange = range(nFramesT)
        testingRange = range(nFrames)

    mu,sigma, lastStabilizedFrame = pipeline.getMuSigma(dataT,trainingRange)
    # res = np.concatenate((mu,sigma),1)
    # cv2.imwrite('musigma.png',res)

    # Second stage: testing
    startingFrame = testingRange[0]

    for idx in testingRange:
        print "Reading frames " + str(idx) + " and " + str(idx+1)
        if idx == startingFrame:
            originalFrame2,originalFrame1 = dataReader.getFrameAndPrevious(data,idx,False)
            # If we are doing Traffic or Highway we take the last stabilizated sample.
            if ID is not 'NewVideo':
                originalFrame1S = stFrame.stabilizatePairOfImages(lastStabilizedFrame, originalFrame1)
                originalFrame2S = stFrame.stabilizatePairOfImages(originalFrame1, originalFrame2)
                last_speed=originalFrame2S
            else:
                originalFrame1S = originalFrame1
                originalFrame2S = stFrame.stabilizatePairOfImages(originalFrame1, originalFrame2)
                last_speed = originalFrame2S
            # Convert images to INT
            frame1 = cv2.cvtColor(originalFrame1S.astype(np.uint8), finalConf.colorSpaceConversion[finalConf.colorSpace])
            frame2 = cv2.cvtColor(originalFrame2S.astype(np.uint8), finalConf.colorSpaceConversion[finalConf.colorSpace])
            # frame1 = frame1[finalConf.area_size:frame1.shape[0]-finalConf.area_size,finalConf.area_size:frame1.shape[1]-finalConf.area_size]
            # frame2 = frame2[finalConf.area_size:frame2.shape[0] - finalConf.area_size, finalConf.area_size:frame2.shape[1] - finalConf.area_size]
        else:
            frame1 = frame2
            originalFrame1S = originalFrame2S
            originalFrame2 = dataReader.getSingleFrame(data,idx,False)
            originalFrame2S = stFrame.stabilizatePairOfImages(originalFrame1S, originalFrame2)
            frame2 = cv2.cvtColor(originalFrame2S.astype(np.uint8),finalConf.colorSpaceConversion[finalConf.colorSpace])
            # frame2 = frame2[finalConf.area_size:frame2.shape[0] - finalConf.area_size, finalConf.area_size:frame2.shape[1] - finalConf.area_size]
        # print frame1.shape, frame2.shape
        out1, mu, sigma = pipeline.getObjectsFromFrame(frame1,mu,sigma,alpha, rho)
        out2, mu, sigma = pipeline.getObjectsFromFrame(frame2,mu,sigma,alpha, rho)
        # originalFrame1SZoom = originalFrame1S[finalConf.area_size:originalFrame1S.shape[0] - finalConf.area_size, finalConf.area_size:originalFrame1S.shape[1] - finalConf.area_size]
        # originalFrame2SZoom = originalFrame2S[finalConf.area_size:originalFrame2S.shape[0] - finalConf.area_size, finalConf.area_size:originalFrame2S.shape[1] - finalConf.area_size]


        if idx == startingFrame:
            objectList, bbox1, bbox2 = tracking.computeTrackingBetweenFrames(True,[],idx,originalFrame1S,out1,originalFrame2S,out2, iter, last_speed)
            if iter%5==0 or iter==1:
                last_speed=originalFrame2S
            res = np.concatenate((bbox1,np.stack([out1*255, out1*255, out1*255], axis=-1)),1)
            cv2.imwrite("./results/Mask_" + str(idx) + '.png', np.stack([out1*255, out1*255, out1*255], axis=-1))
            cv2.imwrite("./results/Image_" + str(idx) + '.png', bbox1)
            cv2.imwrite("./results/ComboImage_" + str(idx) + '.png', res)

            # cv2.imshow("Image " + str(idx), res)
            # cv2.waitKey(1)
        else:
            objectList, bbox1, bbox2 = tracking.computeTrackingBetweenFrames(False,objectList,idx,originalFrame1S,out1,originalFrame2S,out2, iter, last_speed)
            if iter%5==0 or iter==1:
                last_speed=originalFrame2S

        iter = iter + 1 #New frame processed
        res = np.concatenate((bbox2, np.stack([out2 * 255, out2 * 255, out2 * 255], axis=-1)), 1)
        cv2.imwrite("./results/Mask_" + str(idx+1) + '.png', np.stack([out2 * 255, out2 * 255, out2 * 255], axis=-1))
        cv2.imwrite("./results/Image_" + str(idx+1) + '.png', bbox2)
        cv2.imwrite("./results/ComboImage_" + str(idx+1) + '.png', res)
        # cv2.imshow("Image", res)
        # cv2.waitKey(1)
