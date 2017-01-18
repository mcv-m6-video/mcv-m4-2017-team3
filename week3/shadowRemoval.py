# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:42:53 2017
@author: Group 3
"""

import cv2
import os
import numpy as np
from scipy import stats
import configuration as conf
import glob


# Original idea from:
# http://www.ee.cuhk.edu.hk/~jshao/papers_jshao/jshao_MovObjectDet_2012.pdf
# 2005-A RE-EVALUATION OF MIXTURE-OF-GAUSSIAN BACKGROUND MODELING.pdf

def bgr_to_rgs(img):
    # This is an implementation of the rgs method described by Elgammal et al.
    # in "Non-parametric model for background subtraction." Computer Vision 751-767.
    # It expects to receive a BGR image.

    # First, extract the BGR components of the given image
    B = np.array(img[:, :, 0], dtype=np.float32)
    G = np.array(img[:, :, 1], dtype=np.float32)
    R = np.array(img[:, :, 2], dtype=np.float32)

    # Compute the luminance as the sum of the previous channels
    sum_channels = np.zeros(img.shape, dtype=np.float32)
    sum_channels = B + G + R

    # Compute the normalized chromacity coordinates
    gp = np.divide(G, sum_channels, dtype=np.float32)
    rp = np.divide(R, sum_channels, dtype=np.float32)

    # Compute the luminance
    I = np.divide(sum_channels, 3, dtype=np.float32)

    # Map from [0,1] values to [0,255]
    rint = rp * 255.0
    gint = gp * 255.0

    # Generate the image in uint8 format
    rgs = np.array([rint, gint, I], dtype=np.uint8)
    RGS = np.array([R, G, I], dtype=np.uint8)

    # Convert the output image shape to (width,height,channels)
    rgs = rgs.transpose(1, 2, 0)
    RGS = RGS.transpose(1, 2, 0)

    # Turn to black the bottom part of the image
    # rgs[(int(rgs.shape[0]/1.5)):,:,:] = 0

    return rgs, RGS


def rgs_thresholding(rgs, th=np.array([0, 27])):
    checkLogicalOne = rgs[:, :, 2] >= th[0]
    checkLogicalTwo = rgs[:, :, 2] < th[1]
    checkLogical = np.bitwise_and(checkLogicalOne, checkLogicalTwo)
    check = checkLogical.astype(np.uint8)

    one = np.multiply(check, rgs[:, :, 0])
    two = np.multiply(check, rgs[:, :, 1])
    three = np.multiply(check, rgs[:, :, 2])

    filtered_image = np.stack([one, two, three], axis=-1)

    return filtered_image


def generate_mask_and_inpaint_shadows(original_img, filtered_img):
    rows = len(filtered_img)
    cols = len(filtered_img[0])
    channels = 3

    # Generate the mask to inpaint given the shadow mask
    mask_to_inpaint = np.array(np.zeros([rows, cols]), dtype=np.uint8)

    for x in range(rows):  # for each row
        for y in range(cols):  # for each column
            for c in range(channels):  # for each color channel
                if filtered_img[x, y, c] > 0:
                    mask_to_inpaint[x, y] = 255
                else:
                    mask_to_inpaint[x, y] = 0

    # Apply a dilation to increase the area to inpaint
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_to_inpaint = cv2.dilate(mask_to_inpaint, kernel, iterations=3)

    # Perform an image inpainting
    inpaint = cv2.inpaint(original_img, mask_to_inpaint, 3, cv2.INPAINT_TELEA)

    return inpaint


# Remove the shadows on an image given the input image
def shadow_removal(frame):
    # Convert from BGR to RGS/rgS color space
    rgs, RGS = bgr_to_rgs(frame)

    # Compute the shadow mask given the rgS image and a threshold
    filtered_img = rgs_thresholding(rgs)
    cv2.imwrite('test/filtered_img.png', filtered_img)

    # Inpaint the regions corresponding to the shadows
    inpainted_img = generate_mask_and_inpaint_shadows(frame, filtered_img)

    return inpainted_img


# Remove the shadows on an image given the input image [0,255] and a mask [0,1]
def inmask_shadow_removal(frame, mask):
    # Apply the mask to the original image
    # If the mask is 3d
    # filter_img = frame[:,:,:] * np.divide(mask, 255.0, dtype=np.float32)
    # If the mask is 2d
    one = np.multiply(mask, frame[:, :, 0])
    two = np.multiply(mask, frame[:, :, 1])
    three = np.multiply(mask, frame[:, :, 2])

    filter_img = np.stack([one, two, three], axis=-1)

    # Convert from BGR to RGS/rgS color space
    filter_img, RGS = bgr_to_rgs(filter_img)

    # Threshold the S channel corresponding to the luminance in order to remove the shadows
    filter_img = rgs_thresholding(filter_img, np.array([0, 25]))

    # Generate the output mask
    # if the mask is 3d
    # output_mask =  (filter_img[:,:,:] == 0) * mask[:,:,:]
    # if the mask is 2d
    output_mask = np.multiply(mask, filter_img[:, :, 2] == 0)

    return output_mask


if __name__ == "__main__":
    dataset = "Highway"
    datasetGT = "HighwayGT"
    ID = dataset
    IDGT = datasetGT
    folder = conf.folders[ID]
    folderGT = conf.folders[IDGT]
    framesFiles = sorted(glob.glob(folder + '*'))
    framesFilesGT = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)

    background = cv2.imread(framesFiles[0])
    original = cv2.imread(framesFiles[1200])
    mask = cv2.imread(framesFilesGT[0])
    mask = mask[:, :, 0]
    mask = mask / 255.0

    # Generate the output mask
    # inpaint = shadow_removal(original)
    output_mask = inmask_shadow_removal(original, mask)
