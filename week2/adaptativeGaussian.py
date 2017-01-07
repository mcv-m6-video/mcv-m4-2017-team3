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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mu = np.zeros_like(frame).ravel()
    sigma = np.zeros_like(frame).ravel()

    trainingPercentage = 0.5

    #Background estimation
    for idx in range(0,max(0,int(nFrames * trainingPercentage))):
        frame = cv2.imread(framesFiles[idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mu = ((idx) * mu + frame.ravel())/float(idx + 1)
    for idx in range(0,max(0,int(nFrames * trainingPercentage))):
        frame = cv2.imread(framesFiles[idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sigma = sigma + (frame.ravel() - mu)**2

    sigma = np.sqrt(sigma / max(0,int(nFrames * trainingPercentage)))

    mu = mu.reshape(frame.shape[0],frame.shape[1])
    sigma = sigma.reshape(frame.shape[0],frame.shape[1])

    cv2.imwrite("results/mean-training" + str(trainingPercentage) + "-alfa-" + str(conf.alfa) + ".png",mu.astype(np.uint8))
    cv2.imwrite("results/sigma-training" + str(trainingPercentage) + "-alfa-" + str(conf.alfa) + ".png",sigma.astype(np.uint8))

    return mu, sigma
    
def foreground_substraction(ID, IDGT, mu, sigma):

    rho = conf.rho

    folder = conf.folders[ID]
    folderGT = conf.folders[IDGT]
    framesFiles   = sorted(glob.glob(folder + '*'))
    framesFilesGT = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)

    frame = cv2.imread(framesFiles[0])

    # openCV 3
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # openCV 2
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    videoOutput = cv2.VideoWriter(ID + '-alfa' +str(conf.alfa) + '.avi',fourcc, 20.0, (frame.shape[0],frame.shape[1]))

    trainingPercentage = 0.5

    #Foreground substraction
    for idx in range(max(0,int(nFrames * trainingPercentage)),nFrames):
        frame = cv2.imread(framesFiles[idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        groundTruth = cv2.imread(framesFilesGT[idx])
        groundTruth = cv2.cvtColor(groundTruth, conf.colorSpaceConverion['gray'])

        out = np.abs(frame[:,:] - mu[:,:]) >= conf.alfa * (sigma[:,:] + 2)
        out = out.astype(np.uint8)

        muFlat       = mu.ravel()
        sigmaFlat    = sigma.ravel()
        outFlat      = out.ravel()
        frameFlat    = frame.ravel()
        updateValues = np.where(outFlat==0)
        for i in range(0, len(updateValues[0])):
            # Update parameters
            muFlat[updateValues[0][i]]    = rho * frameFlat[updateValues[0][i]] + (1 - rho) * muFlat[updateValues[0][i]]
            sigmaFlat[updateValues[0][i]] = rho * (frameFlat[updateValues[0][i]] - muFlat[updateValues[0][i]]) ** 2 + (1 - rho) * sigmaFlat[updateValues[0][i]]

        mu = muFlat.reshape(mu.shape[0], mu.shape[1])
        sigma = sigmaFlat.reshape(sigma.shape[0], sigma.shape[1])

        #  Find erroneous pixels
        groundTruth = np.abs(groundTruth[:, :] > 0)
        outError = np.bitwise_xor(out, groundTruth)

        out = out.astype(np.uint8)
        outError = outError.astype(np.uint8)
        groundTruth = groundTruth.astype(np.uint8)

        instance = np.stack([out, out, outError], axis=-1)
        cv2.imshow("OutputColor", instance * 255)
        # cv2.imshow("Image", frame)
        # cv2.imshow("Output", out * 255)
        # cv2.imshow("GT", groundTruth*255)

        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break
        videoOutput.write(out * 255)

    videoOutput.release()
if __name__ == "__main__":
    dataset    = "Highway"
    datasetGT  = "HighwayGT"
    mu, sigma = obtainGaussianModell(dataset)
    foreground_substraction(dataset, datasetGT, mu, sigma)
