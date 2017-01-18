#!/usr/bin/env
import numpy as np
import cv2
import configuration as conf
import glob
import os

operativeSystem = os.name
(CVmajor, CVminor, _) = cv2.__version__.split(".")
#  gaussianModelling function models each pixel.
#  It is able to classify pixel as background or foreground.
#  It provides a color representation in order to make easier the understanding
#  Foreground:
#       TP. Blue pixels
#       FP. White pixels
#       FN. Red pixels
#  Background: all the black pixels

def obtainGaussianModell(ID, IDGT, colorSpace, alfa):

    folder = conf.folders[ID]
    folderGT = conf.folders[IDGT]
    framesFiles   = sorted(glob.glob(folder + '*'))
    framesFilesGT = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)
    frame = cv2.imread(framesFiles[0])
    if colorSpace == 'gray':
        frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])

    mu = np.zeros_like(frame).ravel()
    sigma = np.zeros_like(frame).ravel()

    if CVmajor == '3':
        # openCV 3
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        # openCV 2
        fourcc = cv2.cv.CV_FOURCC(*'MJPG')

    videoOutput = cv2.VideoWriter("videos/" + ID + '-alfa' +str(alfa) + '.avi',fourcc, 20.0, (frame.shape[1],frame.shape[0]))

    trainingPercentage = 0.5

    for idx in range(0,max(0,int(nFrames * trainingPercentage))):
        frame = cv2.imread(framesFiles[idx])
        if colorSpace != 'BGR':
            frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])
        mu = ((idx) * mu + frame.ravel())/float(idx + 1)

    for idx in range(0,max(0,int(nFrames * trainingPercentage))):
        frame = cv2.imread(framesFiles[idx])
        if colorSpace != 'BGR':
            frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])
        sigma = sigma + (frame.ravel() - mu)**2

    sigma = np.sqrt(sigma / max(0,int(nFrames * trainingPercentage)))

    if colorSpace == 'gray':
        mu = mu.reshape(frame.shape[0],frame.shape[1])
        sigma = sigma.reshape(frame.shape[0],frame.shape[1])
    else:
        mu = mu.reshape(frame.shape[0], frame.shape[1], frame.shape[2])
        sigma = sigma.reshape(frame.shape[0], frame.shape[1], frame.shape[2])

    cv2.imwrite("results/mean-training" + str(trainingPercentage) + "-alfa-" + str(alfa) + ".png",mu.astype(np.uint8))
    cv2.imwrite("results/sigma-training" + str(trainingPercentage) + "-alfa-" + str(alfa) + ".png",sigma.astype(np.uint8))

    for idx in range(max(0,int(nFrames * trainingPercentage)),nFrames):
        frame = cv2.imread(framesFiles[idx])
        if colorSpace != 'BGR':
            frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])

        groundTruth = cv2.imread(framesFilesGT[idx])
        groundTruth = cv2.cvtColor(groundTruth, conf.colorSpaceConversion['gray'])

        if colorSpace != 'gray':
            out = np.abs(frame[:,:,0] - mu[:,:,0]) >= alfa * (sigma[:,:,0] + 2)
            for channel in range(1,frame.shape[2]):
                out = np.bitwise_or(np.abs(frame[:,:,channel] - mu[:,:,channel]) >= alfa * (sigma[:,:,channel] + 2),out)
        else:
            out = np.abs(frame[:,:] - mu[:,:]) >= alfa * (sigma[:,:] + 2)

        #  Find erroneous pixels
        groundTruth = np.abs(groundTruth[:,:] > 0)
        outError = np.bitwise_xor(out, groundTruth)

        out = out.astype(np.uint8)
        outError = outError.astype(np.uint8)
        groundTruth = groundTruth.astype(np.uint8)

        instance = np.stack([out, out, outError], axis=-1)


        # cv2.imshow("OutputColor", instance * 255)
        # cv2.imshow("Image", frame)
        # # cv2.imshow("Output", out * 255)
        # # cv2.imshow("GT", groundTruth*255)
        #
        # k = cv2.waitKey(5) & 0xff
        # if k == 27:
        #     break

        videoOutput.write(instance * 255)

        file_name = framesFiles[idx]

        #OS dependant writing
        if operativeSystem == 'posix':
            #posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
            cv2.imwrite('./results/imagesGaussianModelling/' + ID + file_name.split('/')[-1] + '.png', out)

        else:
            #say hello to propietary software
            cv2.imwrite('./results/imagesGaussianModelling/' + ID + file_name.split('\\')[-1] + '.png' , out)

    videoOutput.release()


if __name__ == "__main__":
    dataset    = "Traffic"
    datasetGT  = "TrafficGT"
    colorSpace = 'gray' # 'gray', 'HSV', 'YCrCb', 'BGR'
    obtainGaussianModell(dataset, datasetGT, colorSpace, conf.alfa)
