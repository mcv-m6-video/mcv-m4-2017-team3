#!/usr/bin/env
import numpy as np
import cv2
import sys
sys.path.append('../')
import configuration as conf
import glob
import os
import evaluation as ev
import holefillingFunction as hf
import shadowRemoval as sr
import morphology as mp
import bwareaopen as bwareaopen


operativeSystem = os.name
(CVmajor, CVminor, _) = cv2.__version__.split(".")

def obtainGaussianModell(ID, colorSpace = 'gray'):

    folder = conf.folders[ID]
    framesFiles = sorted(glob.glob(folder + '*'))
    nFrames = len(framesFiles)

    frame = cv2.imread(framesFiles[0])
    if colorSpace == 'gray':
        frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])

    mu = np.zeros_like(frame).ravel()
    sigma = np.zeros_like(frame).ravel()

    trainingPercentage = 0.5

    #Background estimation
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

    return mu, sigma

def foreground_substraction(ID, IDGT, mu, sigma, alpha, rho, colorSpace, P):

    folder = conf.folders[ID]
    folderGT = conf.folders[IDGT]
    framesFiles   = sorted(glob.glob(folder + '*'))
    framesFilesGT = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)

    frame = cv2.imread(framesFiles[0])

    if CVmajor == '3':
        # openCV 3
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        # openCV 2
        fourcc = cv2.cv.CV_FOURCC(*'MJPG')

    # videoOutput = cv2.VideoWriter("videos/adaptative-" + ID + '-alfa' +str(conf.alfa) + '.avi',fourcc, 20.0, (frame.shape[1],frame.shape[0]))

    trainingPercentage = 0.5

    #Foreground substraction
    for idx in range(max(0,int(nFrames * trainingPercentage)),nFrames):

        frame = cv2.imread(framesFiles[idx])
        if colorSpace != 'BGR':
            frame = cv2.cvtColor(frame, conf.colorSpaceConversion[colorSpace])

        groundTruth = cv2.imread(framesFilesGT[idx])
        groundTruth = cv2.cvtColor(groundTruth, conf.colorSpaceConversion['gray'])

        if colorSpace != 'gray':
            out = np.abs(frame[:,:,0] - mu[:,:,0]) >= alpha * (sigma[:,:,0] + 2)
            for channel in range(1,frame.shape[2]):
                out = np.bitwise_or(np.abs(frame[:,:,channel] - mu[:,:,channel]) >= alpha * (sigma[:,:,channel] + 2),out)
        else:
            out = np.abs(frame[:,:] - mu[:,:]) >= alpha * (sigma[:,:] + 2)

        out = out.astype(np.uint8)

        if colorSpace != 'gray':
            outExtraDimension = np.stack([out, out, out], axis=-1)
            outFlat = outExtraDimension.ravel()
        else:
            outFlat = out.ravel()

        muFlat       = mu.ravel()
        sigmaFlat    = (sigma.ravel())**2
        frameFlat    = frame.ravel()

        muFlat = np.multiply(outFlat,muFlat) + np.multiply((rho * frameFlat + (1 - rho) * muFlat),(1-outFlat))
        sigmaFlat = np.multiply(outFlat,sigmaFlat) + np.multiply((rho * (frameFlat - muFlat)**2 + (1 - rho) * sigmaFlat),(1-outFlat))
        sigmaFlat = np.sqrt(sigmaFlat)


        if colorSpace == 'gray':
            mu = muFlat.reshape(mu.shape[0], mu.shape[1])
            sigma = sigmaFlat.reshape(sigma.shape[0], sigma.shape[1])
        else:
            mu = muFlat.reshape(mu.shape[0], mu.shape[1], frame.shape[2])
            sigma = sigmaFlat.reshape(sigma.shape[0], sigma.shape[1], frame.shape[2])

        #  Find erroneous pixels
        groundTruth = np.abs(groundTruth[:, :] > 0)
        outError = np.bitwise_xor(out, groundTruth)

        if conf.isHoleFilling:
            out = hf.holefilling(out, conf.fourConnectivity)

        if conf.isFilteringPerPixel:
            out = bwareaopen.bwareaopen(out, P)

        if conf.isMorphology:
            out = mp.apply_morphology_noise(out, conf.noise_filter_size)
            out = mp.apply_morphology_vertline(out, conf.vert_filter_size)
            out = mp.apply_morphology_horzline(out, conf.horz_filter_size)
            if ID == "Traffic":
                out = mp.apply_morphology_little(out, conf.vert_filter_size, conf.horz_filter_size)

        if conf.isShadowremoval[ID]:
            out = sr.inmask_shadow_removal(frame, out)

        if conf.isHoleFilling and conf.isShadowremoval[ID]:
            out = hf.holefilling(out, conf.fourConnectivity)


        outError = outError.astype(np.uint8)
        groundTruth = groundTruth.astype(np.uint8)

        instance = np.stack([out, out, outError], axis=-1)
        # cv2.imshow("OutputColor", instance * 255)
        # cv2.imshow("Image", frame)
        # cv2.imshow("Output", out * 255)
        # cv2.imshow("GT", groundTruth*255)
        #
        # k = cv2.waitKey(5) & 0xff
        # if k == 27:
        #     break

        file_name = framesFiles[idx]
        #OS dependant writing
        if operativeSystem == 'posix':
            #posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
            if conf.isMac:
                cv2.imwrite('./results/imagesAdaptativeGaussianModelling/' + ID + file_name.split('/')[-1][0:-4] + '.png', out)
            else:
                cv2.imwrite('./results/imagesAdaptativeGaussianModelling/' + ID + file_name.split('/')[-1] + '.png', out)
        else:
            #say hello to propietary software
            cv2.imwrite('./results/imagesAdaptativeGaussianModelling/' + ID + file_name.split('\\')[-1].split('.')[0] + '.png', out)

        # videoOutput.write(instance * 255)

    # videoOutput.release()


if __name__ == "__main__":
    dataset    = "Fall"
    datasetGT  = "FallGT"

    # Check if 'results' folder exists.
    results_path = "./results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    gM_path = results_path + "/imagesGaussianModelling/"
    if not os.path.exists(gM_path):
        os.makedirs(gM_path)

    aG_path = results_path + "/imagesAdaptativeGaussianModelling/"  # "/imagesAdaptativeGaussian/"
    if not os.path.exists(aG_path):
        os.makedirs(aG_path)

    if conf.isShadowremoval:
        print('--- obtainGaussianModell')
        mu, sigma = obtainGaussianModell(dataset, conf.OptimalColorSpaces["ShadowRemoval"])
        print('--- foreground_substraction')
        foreground_substraction(dataset, datasetGT, mu, sigma,
                        conf.OptimalAlphaParameter[dataset], conf.OptimalRhoParameter[dataset], conf.OptimalColorSpaces["ShadowRemoval"])
        print('--- evaluateFolder')
        aux, aux, aux, aux, aux, aux, F1 = ev.evaluateFolder(aG_path, dataset)
    else:
        print('--- obtainGaussianModell')
        mu, sigma = obtainGaussianModell(dataset, conf.OptimalColorSpaces[dataset])
        print('--- foreground_substraction')
        foreground_substraction(dataset, datasetGT, mu, sigma,
                        conf.OptimalAlphaParameter[dataset], conf.OptimalRhoParameter[dataset], conf.OptimalColorSpaces[dataset])
        print('--- evaluateFolder')
        aux, aux, aux, aux, aux, aux, F1 = ev.evaluateFolder(aG_path, dataset)

    print ('--- DataSet: ' + dataset)
    print ('--- Rho: ' + str(conf.OptimalRhoParameter[dataset]) + ' --- ' + ' Alpha: ' + str(conf.OptimalAlphaParameter[dataset]))
    print ('--- F1 Adaptative Color Gaussian Model: ' + str(F1))
