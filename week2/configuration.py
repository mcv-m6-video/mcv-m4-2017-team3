#!/usr/bin/env
import cv2


colorSpaceConverion = {}
folders = {}
alfa = 1
rho = 0


# YCrCb (99 out of 100 you will be doing it in YCrCb when doing video)
colorSpaceConverion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConverion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConverion['gray']  = cv2.COLOR_BGR2GRAY

folders["Highway"]  = "../../../datasetDeliver_2/highway/input/"
folders["HighwayGT"]  = "../../../datasetDeliver_2/highway/groundtruth/"
