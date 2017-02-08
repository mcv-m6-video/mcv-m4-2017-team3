import sys
sys.path.append("../")
sys.path.append('../tools/')

import numpy as np
import configuration as conf
import glob
from skimage import morphology
import cv2.cv as cv
import numpy as np
import cv2

def createImageWithLines(img):
    cv2.line(img, (conf.pointLine1[conf.ID][0][0], conf.pointLine1[conf.ID][0][1]), (conf.pointLine1[conf.ID][1][0], conf.pointLine1[conf.ID][1][1]),
             (0,255,0), conf.linesThickness[conf.ID])

    cv2.line(img, (conf.pointLine2[conf.ID][0][0], conf.pointLine2[conf.ID][0][1]), (conf.pointLine2[conf.ID][1][0], conf.pointLine2[conf.ID][1][1]),
             (0,255,0), conf.linesThickness[conf.ID])

    return img


if __name__ == "__main__":
    print('Create images with lines ')