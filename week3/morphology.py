#!/usr/bin/env
#Script to apply some morphology operations on the foreground substaction results

import numpy as np
import cv2
import glob
import ntpath
import evaluation as ev
#from matplotlib import pyplot as plt

def read_files(folder):
    #List the images to process
    image_files = sorted(glob.glob(folder + '*'))
    return image_files

def apply_morphology_noise(image, noise_filter_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (noise_filter_size, noise_filter_size))
    trans_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return trans_img

def apply_morphology_vertline(image, vert_filter_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_filter_size))
    trans_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return trans_img

def apply_morphology_horzline(image, horz_filter_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horz_filter_size, 1))
    trans_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return trans_img

def apply_morphology_little(image, vert_filter_size, horz_filter_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horz_filter_size, 2))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, vert_filter_size))
    trans_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    trans_img = cv2.morphologyEx(trans_img, cv2.MORPH_OPEN, kernel2)
    return trans_img


if __name__ == "__main__":
    dataset = "highwayDataset"
    dataset_path = "/Users/marccarneherrera/Desktop/M4-week3/dataset/"
    result_path = "/Users/marccarneherrera/Desktop/M4-week3/results/"
    dataset_folder = dataset_path + dataset + "/"
    result_folder = result_path + dataset + "/"

    image_files = read_files(dataset_folder)

    #Morphological parameters
    noise_filter_size = 5;
    vert_filter_size = 20;
    horz_filter_size = 20;

    for image in image_files:
        #Read binary image, the mask for the foreground substraction
        img = cv2.imread(image)
        img = img[:,:,1]
        trans_img = apply_morphology_noise(img, noise_filter_size)
        #trans_little = apply_morphology_little(trans_img)
        trans_img_vert = apply_morphology_vertline(trans_img, vert_filter_size)
        trans_img_horz = apply_morphology_horzline(trans_img_vert, horz_filter_size)
        #trans_little = apply_morphology_little(trans_img_horz)
        #cv2.imshow("Original", img)
        #cv2.imshow("Transformation", trans_img)
        #cv2.imshow("Completion", trans_img_vert)
        #cv2.imshow("Max completion", trans_img_horz)
        #cv2.waitKey(0)

        #Write the image
        file_name = ntpath.basename(image)
        print "Done for image: " + file_name
        cv2.imwrite(result_folder + file_name, trans_img_horz)

    TP, TN, FP, FN, precision, recall, F1 = ev.evaluateFolder("/Users/marccarneherrera/Desktop/M4-week3/results/","highwayDataset")

    print precision
    print recall
    print F1