#!/usr/bin/env
import numpy as np
import cv2
import configuration as conf
import glob


def StaufferAndGrimsonAlgorithm(ID, IDGT):

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

    for idx in range(max(0,int(nFrames * trainingPercentage)),nFrames):
        frame = cv2.imread(framesFiles[idx])
        groundTruth = cv2.imread(framesFilesGT[idx])
        groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)

        count = 0
        history = 10
        nGauss = 3
        bgThresh = 0.6
        noise = 20
        out = cv2.BackgroundSubtractorMOG(history, nGauss, bgThresh, noise)

        #  Find erroneous pixels
        groundTruth = np.abs(groundTruth[:,:] > 0)
        outError = np.bitwise_xor(out, groundTruth)

        out = out.astype(np.uint8)
        outError = outError.astype(np.uint8)
        groundTruth = groundTruth.astype(np.uint8)

        instance = np.stack([out, out, outError], axis=-1)
        cv2.imshow("OutputColor", instance * 255)
        cv2.imshow("Image", frame)
        # cv2.imshow("Output", out * 255)
        # cv2.imshow("GT", groundTruth*255)

        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break
        videoOutput.write(instance * 255)

    videoOutput.release()


if __name__ == "__main__":
    StaufferAndGrimsonAlgorithm("Highway", "HighwayGT")