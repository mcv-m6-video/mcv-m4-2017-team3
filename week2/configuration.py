#!/usr/bin/env
import cv2


colorSpaceConverion = {}
folders = {}
alfa = 1
rho = 0
history = 20
nGauss = 3
bgThresh = 0.3

# YCrCb (99 out of 100 you will be doing it in YCrCb when doing video)
colorSpaceConverion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConverion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConverion['gray']  = cv2.COLOR_BGR2GRAY

folders["Highway"]  = "../../../datasetDeliver_2/highway/input/"
folders["HighwayGT"]  = "../../../datasetDeliver_2/highway/groundtruth/"

# Evaluation

# Results logging
outputFolder = "./results/"
outputFile = outputFolder + "results.json"
outputFileDelay = outputFolder + "delayResults.json"

# The highway labels, as they are defined in the gt.
highwayLabelsInv = dict()

highwayLabelsInv['Static'] = 0
highwayLabelsInv['Hard Shadow'] = 50
highwayLabelsInv['Outside ROI'] = 85
highwayLabelsInv['Unknown motion'] = 170
highwayLabelsInv['Motion'] = 255

highwayLabels = dict()

highwayLabels.update({0:'Static'})
highwayLabels.update({50:'Hard Shadow'})
highwayLabels.update({85:'Outside ROI'})
highwayLabels.update({170:'Unknown motion'})
highwayLabels.update({255:'Motion'})

# Define here more mappings.
# Important: The foreground objects MUST have the higher label (1 for a binary case,
# 2 for a three case scenario, etc...), as the precision, recall and accuracy are
# obtained from the last element.

# A possible option for a binary mapping of the labels to bg (0)/fg (1)
highwayBinaryMapping1 = dict()
highwayBinaryMapping1.update({highwayLabelsInv['Static']:0})
highwayBinaryMapping1.update({highwayLabelsInv['Hard Shadow']:0})
highwayBinaryMapping1.update({highwayLabelsInv['Outside ROI']:0})
highwayBinaryMapping1.update({highwayLabelsInv['Unknown motion']:1})
highwayBinaryMapping1.update({highwayLabelsInv['Motion']:1})
highwayBinaryMapping1.update({"Classes":2})

# Mapping used. Change this line to change the mapping
highwayMapping = highwayBinaryMapping1