#!/usr/bin/env
import numpy as np
import cv2
import configuration as conf
import glob
import evaluation as ev

def StaufferAndGrimsonAlgorithm(ID, IDGT, history, nGauss, bgThresh):

    folder = conf.folders[ID]
    folderGT = conf.folders[IDGT]
    framesFiles   = sorted(glob.glob(folder + '*'))
    framesFilesGT = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)

    noise = 20
    model_SG = cv2.BackgroundSubtractorMOG(history, nGauss, bgThresh, noise)

    # Training Stage
    trainingPercentage = 0.5

    for idx in range(0,max(0,int(nFrames * trainingPercentage))):
        frame = cv2.imread(framesFiles[idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        model_SG.apply(frame, learningRate=1.0/history)

    # Create video instance
    # openCV 3
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # openCV 2
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    videoOutput = cv2.VideoWriter("videos/" + ID + 'StaufferAndGrimsonAlgorithm.avi',fourcc, 20.0, (frame.shape[0],frame.shape[1]))

    # Compute adaptive background mixture model
    for idx in range(max(0,int(nFrames * trainingPercentage)),nFrames):
        frame = cv2.imread(framesFiles[idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        groundTruth = cv2.imread(framesFilesGT[idx])
        groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)

        out = model_SG.apply(frame, learningRate=0.2/history)
        #  Find erroneous pixels
        out = np.abs(out[:,:] > 0)
        groundTruth = np.abs(groundTruth[:,:] > 0)
        outError = np.bitwise_xor(out, groundTruth)

        out = out.astype(np.uint8)
        outError = outError.astype(np.uint8)
        groundTruth = groundTruth.astype(np.uint8)

        instance = np.stack([out, out, outError], axis=-1)
        # cv2.imshow("OutputColor", instance * 255)
        # cv2.imshow("Image", frame)
        # cv2.imshow("Output", out * 255)
        # cv2.imshow("GT", groundTruth*255)

        # k = cv2.waitKey(5) & 0xff
        # if k == 27:
        #     break

        file_name = framesFiles[idx]
        cv2.imwrite('./results/StaufferAndGrimson/' + file_name[file_name.rfind('\\') + 1:], out)
        videoOutput.write(instance * 255)

    videoOutput.release()
    

if __name__ == "__main__":
    dataset    = "Highway"
    datasetGT  = "HighwayGT"
    history = conf.history
    nGauss = conf.nGauss
    bgThresh = conf.bgThresh
    StaufferAndGrimsonAlgorithm(dataset, datasetGT, history, nGauss, bgThresh)
    aux, aux, aux, aux, aux, aux, F1 = ev.evaluateFolder("./results/StaufferAndGrimson/")

    print ('--- F1 Stauffer And Grimson Model: ' + str(F1))