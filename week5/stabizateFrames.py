import cv2
import week5configuration as conf
import numpy as np
import glob
import sys
import os
#import Image
from PIL import Image
import scipy.misc
import week5configuration as finalConf

sys.path.append('../')
operativeSystem = os.name
(CVmajor, CVminor, _) = cv2.__version__.split(".")



# Auxiliary function
def camera_motion(real_x, real_y, curr_img):
    x_size = curr_img.shape[0]
    y_size = curr_img.shape[1]
    block_size = conf.block_size
    area_size = conf.area_size

    x_blocks = x_size / block_size
    y_blocks = y_size / block_size

    if curr_img.shape.__len__() == 2:
        new_curr_img = np.zeros([x_size + 2 * area_size, y_size + 2 * area_size])
        new_curr_img[area_size:area_size + x_size, area_size:area_size + y_size] = curr_img
        comp_img = np.zeros([x_size, y_size])
    elif curr_img.shape.__len__() == 3:
        new_curr_img = np.zeros([x_size + 2 * area_size, y_size + 2 * area_size, 3])
        auxImg = np.pad(curr_img[:, :, 0], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 0] = auxImg
        auxImg = np.pad(curr_img[:, :, 1], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 1] = auxImg
        auxImg = np.pad(curr_img[:, :, 2], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 2] = auxImg
        comp_img = np.zeros([x_size, y_size, 3])
    else:
        print 'ERROR dimension'
        return curr_img

    if curr_img.shape.__len__() == 2:
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1]] = new_curr_img[0+ real_x :comp_img.shape[0] + real_x , 0 + real_y:comp_img.shape[1]+ real_y]

    elif curr_img.shape.__len__() == 3:
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 0] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 0]
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 1] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 1]
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 2] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 2]

    return comp_img

# Auxiliary function
def compute_error(block1, block2):
    return sum(sum(abs(block1-block2)**2))

# Auxiliary function
def block_search(region_to_explore, block_to_search):
    block_sizeX = block_to_search.shape[0]
    block_sizeY = block_to_search.shape[1]
    area_size = finalConf.area_size
    x_size = region_to_explore.shape[0]
    y_size = region_to_explore.shape[1]

    min_diff = sys.float_info.max
    x_mot = 0
    y_mot = 0
    for row in range(x_size-block_sizeX):
        for column in range(y_size-block_sizeY):
            block2analyse = region_to_explore[row:row+block_sizeX, column:column+block_sizeY]
            diff = compute_error(block2analyse, block_to_search)
            if diff < min_diff:
                min_diff = diff
                x_mot = - row + area_size
                y_mot = column - area_size
    return x_mot, y_mot

# Auxiliary function
def compute_block_matching(prev_img, curr_img):
    block_size = conf.block_size
    area_size = conf.area_size
    compensation = conf.compensation
    # We will apply backward compensation
    if compensation == 'backward':
        img2xplore = curr_img
        searchimg = prev_img
    else:
        img2xplore = prev_img
        searchimg = curr_img

    x_blocks = img2xplore.shape[0] / block_size
    y_blocks = img2xplore.shape[1] / block_size

    # Add padding in the search image
    pad_searchimg = np.zeros([img2xplore.shape[0] + 2 * area_size, img2xplore.shape[1] + 2 * area_size])
    pad_searchimg[area_size:area_size + img2xplore.shape[0], area_size:area_size + img2xplore.shape[1]] = searchimg[:,
                                                                                                          :]

    motion_matrix = np.zeros([x_blocks, y_blocks, 2])

    for row in range(x_blocks):
        for column in range(y_blocks):
            # print "Computing block " + str(column)
            block_to_search = img2xplore[row * block_size:row * block_size + block_size,
                              column * block_size:column * block_size + block_size]
            region_to_explore = pad_searchimg[row * block_size:row * block_size + block_size + 2 * area_size,
                                column * block_size:column * block_size + block_size + 2 * area_size]
            x_mot, y_mot = block_search(region_to_explore, block_to_search)

            motion_matrix[row, column, 0] = x_mot
            motion_matrix[row, column, 1] = y_mot

    return motion_matrix



# def stabilizatePairOfImages( image1, image2) will stabilizate image2 respect
# to image1. Thus, image1 must be previously stabilizated.
def stabilizatePairOfImages( image1, image2):
    # Comment this line if you do not want to stabilizate
    return image2
    # Inizializate variables
    block_size = conf.block_size
    area_size = conf.area_size
    compensation = conf.compensation

    # Conver images to BW for computing OF
    image1BW = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    image2BW = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Apply block matching
    motion_matrix = compute_block_matching(image1BW, image2BW)

    # Build matrix of motion
    motion_matrix[:, :, :] = motion_matrix[:, :, :] + area_size
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

    # Apply stabilization to second image.
    image2Stabilizated = camera_motion(real_x, real_y, image2)
    return image2Stabilizated


if __name__ == "__main__":
    print 'Stabilization image functions'

    # Example traffic dataset, sequence 18.
    startFrame = 18
    # Loading images
    folderGT = conf.folders["Traffic"]
    framesFiles = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)
    img1 = cv2.imread(framesFiles[startFrame])
    img2 = cv2.imread(framesFiles[startFrame+1])

    #  Get result to compare
    img2S = stabilizatePairOfImages(img1, img2)