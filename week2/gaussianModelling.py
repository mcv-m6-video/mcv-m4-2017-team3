#!/usr/bin/env
import numpy as np
import cv2
import configuration as conf
import glob


def obtainGaussianModell(ID):

    folder = conf.folders[ID]
    framesFiles = sorted(glob.glob(folder + '*'))
    nFrames = len(framesFiles)

    frame = cv2.imread(framesFiles[0])

    mu = np.zeros_like(frame).ravel()
    sigma = np.zeros_like(frame).ravel()

    trainingPercentage = 0.5

    for idx in range(0,max(0,int(nFrames * trainingPercentage))):
        frame = cv2.imread(framesFiles[idx])
        mu = ((idx) * mu + frame.ravel())/float(idx + 1)
    for idx in range(0,max(0,int(nFrames * trainingPercentage))):
        frame = cv2.imread(framesFiles[idx])
        sigma = sigma + (frame.ravel() - mu)**2
    sigma = np.sqrt(sigma / max(0,int(nFrames * trainingPercentage)))

    mu = mu.reshape(frame.shape[0],frame.shape[1],frame.shape[2])
    sigma = sigma.reshape(frame.shape[0],frame.shape[1],frame.shape[2])

    cv2.imwrite("results/mean-training" + str(trainingPercentage) + "-alfa-" + str(conf.alfa) + ".png",mu.astype(np.uint8))
    cv2.imwrite("results/sigma-training" + str(trainingPercentage) + "-alfa-" + str(conf.alfa) + ".png",sigma.astype(np.uint8))

    for idx in range(max(0,int(nFrames * trainingPercentage)),nFrames):
        frame = cv2.imread(framesFiles[idx])
        out = np.abs(frame[:,:,0] - mu[:,:,0]) >= conf.alfa * (sigma[:,:,0] + 2)
        for channel in range(1,frame.shape[2]):
            out = np.bitwise_or(np.abs(frame[:,:,channel] - mu[:,:,channel]) >= conf.alfa * (sigma[:,:,channel] + 2),out)

        out = out.astype(np.uint8)
        cv2.imshow("Output", out * 255)
        cv2.imshow("Image",frame)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break


if __name__ == "__main__":
    obtainGaussianModell("Highway")
