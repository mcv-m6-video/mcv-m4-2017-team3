import cv2
import configuration as conf
import numpy as np
import glob
import sys
import os
import block_matching as match
import Image
import scipy.misc

import opticalFlowMethods as opticalflow
import evaluateOpticalFlow as evaluateOF

sys.path.append('../')
operativeSystem = os.name
(CVmajor, CVminor, _) = cv2.__version__.split(".")

resultsPath = "./resultsStabilizationTrafficVideo/"
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

if not os.path.exists("./videos/"):
    os.makedirs("./videos/")

################### task 2 ######################
# Task 2.1: Video Stabilization with Block Matching
ID = "Traffic"
folder = conf.folders[ID]
block_size = conf.block_size
area_size = conf.area_size
compensation = conf.compensation

folder = conf.folders[ID]
framesFiles = sorted(glob.glob(folder + '*.jpg'))
nFrames = len(framesFiles)

referenceImageName = framesFiles[0]
referenceImage = cv2.imread(referenceImageName)
referenceImageBW = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

# OS dependant writing
if operativeSystem == 'posix':
    # posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
    if conf.isMac:
        cv2.imwrite(resultsPath + referenceImageName.split('/')[-1][0:-4] + '.png', referenceImage)
    else:
        cv2.imwrite(resultsPath + referenceImageName.split('/')[-1] + '.png', referenceImage)
else:
    # say hello to propietary software
    cv2.imwrite(resultsPath + referenceImageName.split('\\')[-1].split('.')[0] + '.png', referenceImage)

if CVmajor == '3':
    # openCV 3
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
else:
    # openCV 2
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')

size = referenceImage.shape[1], referenceImage.shape[0]

videoOutputOriginal = cv2.VideoWriter("videos/originalVideo.avi", fourcc, 20.0,size, True)
videoOutputPost = cv2.VideoWriter("videos/stabilizedVideo.avi", fourcc, 20.0,size, True)
videoOutputPost.write(referenceImage)
videoOutputOriginal.write(referenceImage)

prev_x = 0
prev_y = 0

for idx in range(1, nFrames):
    file_name = framesFiles[idx]
    if operativeSystem == 'posix':
        # posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
        if conf.isMac:
            print "Computing image " + str(idx) + " " + file_name.split('/')[-1][0:-4]
        else:
            print "Computing image " + str(idx) + " " + file_name.split('/')[-1]
    else:
        print "Computing image " + str(idx) + " " + file_name.split('\\')[-1].split('.')[0]

    currentImage = cv2.imread(framesFiles[idx])
    currentImageBW = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)

    # Apply block matching
    motion_matrix = match.compute_block_matching(referenceImageBW, currentImageBW)
    # motion_matrix = opticalflow.opticalFlowBW(referenceImageBW, currentImageBW)

    # bincount cannot be negative
    motion_matrix[:, :, :] = motion_matrix[:, :, :] + area_size
    #x_motion = np.intp(motion_matrix[:, :, 0].ravel())
    #y_motion = np.intp(motion_matrix[:, :, 1].ravel())
    votation_matrix=np.zeros([2*area_size+1,2*area_size+1])
    for xx in range(motion_matrix.shape[0]):
        for yy in range(motion_matrix.shape[1]):
            votation_matrix[motion_matrix[xx,yy,0],motion_matrix[xx,yy,1]]=votation_matrix[motion_matrix[xx,yy,0],motion_matrix[xx,yy,1]]+1

    maximum = 0
    real_x = 0
    real_y = 0
    for xx in range(votation_matrix.shape[0]):
        for yy in range(votation_matrix.shape[1]):
            if votation_matrix[xx,yy]>maximum:
                maximum = votation_matrix[xx,yy]
                real_x = xx
                real_y = yy

    real_x = real_x - area_size
    real_y = real_y - area_size

    out = match.camera_motion(real_x, real_y, currentImage)
    referenceImageBW = currentImageBW
    prev_x = prev_x + real_x
    prev_y = prev_y + real_y

    # OS dependant writing
    resultPath = ''
    if operativeSystem == 'posix':
        # posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
        if conf.isMac:
            resultPath = resultsPath + file_name.split('/')[-1][0:-4] + '.png'
        else:
            resultPath = resultsPath + file_name.split('/')[-1] + '.png'
    else:
        resultPath = resultsPath + file_name.split('\\')[-1].split('.')[0] + '.png'

    cv2.imwrite(resultPath, out)

    if size[0] != out.shape[1] and size[1] != out.shape[0]:
        img = cv2.resize(out, size)
    else:
        img = out

    videoOutputPost.write(img)
    if size[0] != currentImage.shape[1] and size[1] != currentImage.shape[0]:
        img = cv2.resize(currentImage, size)
    else:
        img = currentImage

    videoOutputOriginal.write(currentImage)

    if not conf.isReferenceImageFixed:
        referenceImage = cv2.imread(resultPath)
        referenceImageBW = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

    # Create prediction image
    # comp_img = create_compensated_image(prev_img, motion_matrix, block_size, x_blocks, y_blocks)

videoOutputPost.release()
videoOutputOriginal.release()
