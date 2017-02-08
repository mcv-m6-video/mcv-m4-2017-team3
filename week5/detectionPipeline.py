import cv2
import sys
import numpy as np
import dataReader
sys.path.append("../")
sys.path.append('../tools')

for i in range(1,5):
    sys.path.append("../week" + str(i))

import configuration as conf
import week5configuration as finalConf

import stabizateFrames

import holefillingFunction as hf
import shadowRemoval as sr
import morphology as mp
import stabizateFrames as stFrame



def getObjectsFromFrame(frame,mu,sigma,alpha, rho):

    colorSpace = finalConf.colorSpace

    # Background Substraction:
    if colorSpace != 'gray':
        out = np.abs(frame[:,:,0] - mu[:,:,0]) >= alpha * (sigma[:,:,0] + 2)
        for channel in range(1,frame.shape[2]):
            out = np.bitwise_or(np.abs(frame[:,:,channel] - mu[:,:,channel]) >= alpha * (sigma[:,:,channel] + 2),out)
    else:
        out = np.abs(frame[:,:] - mu[:,:]) >= alpha * (sigma[:,:] + 2)

    out = out.astype(np.uint8)
    if finalConf.ID == 'Own':
        ROImask = cv2.imread('ROI.png')
        ROImask = ROImask[:,:,0]
        ROImask[np.where(ROImask != 0)] = 1
        # ROI filtering
        out = np.multiply(ROImask,out)
    # Hole filling
    if finalConf.isHoleFilling:
        out = hf.holefilling(out, conf.fourConnectivity)

    # Morpholoy filters
    if finalConf.isMorphology:
        out = mp.apply_morphology_noise(out, conf.noise_filter_size)
        out = mp.apply_morphology_vertline(out, conf.vert_filter_size)
        out = mp.apply_morphology_horzline(out, conf.horz_filter_size)
        out = mp.apply_morphology_little( out, conf.little_filter_size, conf.little_filter_size)
    # Shadow removal
    if finalConf.isShadowremoval:
        out = sr.inmask_shadow_removal(frame, out)
    # Shadow removal tends to remove some car components such as
    if finalConf.isHoleFilling and conf.isShadowremoval:
        out = hf.holefilling(out, conf.fourConnectivity)

    if colorSpace != 'gray':
        outExtraDimension = np.stack([out, out, out], axis=-1)
        outFlat = outExtraDimension.ravel()
    else:
        outFlat = out.ravel()

    muFlat = mu.ravel()
    sigmaFlat = (sigma.ravel()) ** 2
    frameFlat = frame.ravel()

    muFlat = np.multiply(outFlat, muFlat) + np.multiply((rho * frameFlat + (1 - rho) * muFlat), (1 - outFlat))
    sigmaFlat = np.multiply(outFlat, sigmaFlat) + np.multiply((rho * (frameFlat - muFlat) ** 2 + (1 - rho) * sigmaFlat),
                                                              (1 - outFlat))
    sigmaFlat = np.sqrt(sigmaFlat)

    if colorSpace == 'gray':
        mu = muFlat.reshape(mu.shape[0], mu.shape[1])
        sigma = sigmaFlat.reshape(sigma.shape[0], sigma.shape[1])
    else:
        mu = muFlat.reshape(mu.shape[0], mu.shape[1], frame.shape[2])
        sigma = sigmaFlat.reshape(sigma.shape[0], sigma.shape[1], frame.shape[2])

    return out, mu, sigma



def getMuSigma(data,trainingRange):
    colorSpace = finalConf.colorSpace
    currentFrame = dataReader.getSingleFrame(data, 0, False)

    mu = np.zeros_like(currentFrame).ravel()
    sigma = np.zeros_like(currentFrame).ravel()

    #Background estimation
    print 'Computing mu ...'
    for idx in trainingRange:
        # print 'Image ' + str(idx)
        if idx is not 0:
            frame = dataReader.getSingleFrame(data, idx, False)
            stabilizedFrame = stFrame.stabilizatePairOfImages(currentFrame, frame)
            currentFrame = stabilizedFrame
            if colorSpace != 'BGR':
                stabilizedFrame = cv2.cvtColor(stabilizedFrame.astype(np.uint8), finalConf.colorSpaceConversion[colorSpace])
            mu = ((idx) * mu + stabilizedFrame.ravel())/float(idx + 1)
        else:
            if colorSpace != 'BGR':
                stabilizedFrame = cv2.cvtColor(currentFrame.astype(np.uint8), finalConf.colorSpaceConversion[colorSpace])
            else:
                stabilizedFrame = currentFrame
            mu = ((idx) * mu + stabilizedFrame.ravel())/float(idx + 1)

    currentFrame = dataReader.getSingleFrame(data, 0, False)
    print 'Computing sigma ...'
    for idx in trainingRange:
        # print 'Image ' + str(idx)
        if idx is not trainingRange[0]:
            frame = dataReader.getSingleFrame(data, idx, False)
            stabilizedFrame = stFrame.stabilizatePairOfImages(currentFrame, frame)
            currentFrame = stabilizedFrame
            if colorSpace != 'BGR':
                stabilizedFrame = cv2.cvtColor(stabilizedFrame.astype(np.uint8), finalConf.colorSpaceConversion[colorSpace])
            sigma = sigma + (stabilizedFrame.ravel() - mu) ** 2
        else:
            if colorSpace != 'BGR':
                stabilizedFrame = cv2.cvtColor(currentFrame.astype(np.uint8), finalConf.colorSpaceConversion[colorSpace])
            else:
                stabilizedFrame = currentFrame
            sigma = sigma + (stabilizedFrame.ravel() - mu) ** 2

    sigma = np.sqrt(sigma / len(trainingRange))

    if colorSpace == 'gray':
        mu = mu.reshape(frame.shape[0],frame.shape[1])
        sigma = sigma.reshape(frame.shape[0],frame.shape[1])
    else:
        mu = mu.reshape(frame.shape[0], frame.shape[1], frame.shape[2])
        sigma = sigma.reshape(frame.shape[0], frame.shape[1], frame.shape[2])

        # mu = mu[finalConf.area_size:mu.shape[0] - finalConf.area_size, finalConf.area_size:mu.shape[1] - finalConf.area_size]
        # sigma = sigma[finalConf.area_size:sigma.shape[0] - finalConf.area_size,finalConf.area_size:sigma.shape[1] - finalConf.area_size]

    return mu, sigma, stabilizedFrame
