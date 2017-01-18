import numpy as np
import cv2

def bwareaopen(im,P):

    if isinstance(im,str):
        im = cv2.imread(im,0)

    if len(im.shape) == 3:
        if im.shape[2] == 3:
            im = im[:,:,0]

    mask = np.zeros((im.shape[0],im.shape[1],3),dtype = np.uint8)

    out,contours,hierarchy = cv2.findContours(np.array(im), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for idx,c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > P:
            mask = cv2.drawContours(mask,contours,idx,(255,255,255),-1)

    out = np.bitwise_and(mask[:,:,0],im)

    return out
