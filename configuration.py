#!/usr/bin/env
import cv2


colorSpaceConversion = {}
OptimalColorSpaces = {}
OptimalAlphaParameter = {}
OptimalRhoParameter = {}
folders = {}

isMac = True

isHoleFilling = True
isMorphology = True
fourConnectivity = True # If it is set to False, it will be 8 connectivity
isShadowremoval = True

noise_filter_size = 5
vert_filter_size = 20
horz_filter_size = 20

# Best perfomance depending on the dataset that is trained
OptimalColorSpaces["Highway"]   = 'LUV'
OptimalColorSpaces["Fall"]      = 'LUV'
OptimalColorSpaces["Traffic"]   = 'YCrCb'

# ShadowRemoval method only works with BGR components
OptimalColorSpaces["ShadowRemoval"]   = 'BGR'

OptimalAlphaParameter["Highway"]   = 1.8
OptimalAlphaParameter["Fall"]      = 3.6
OptimalAlphaParameter["Traffic"]   = 1.9

OptimalRhoParameter["Highway"]   = 0.04
OptimalRhoParameter["Fall"]      = 0.03
OptimalRhoParameter["Traffic"]   = 0.03



# YCrCb (99 out of 100 you will be doing it in YCrCb when doing video)
colorSpaceConversion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConversion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConversion['HLS']   = cv2.COLOR_BGR2HLS
colorSpaceConversion['gray']  = cv2.COLOR_BGR2GRAY
colorSpaceConversion['LUV']   = cv2.COLOR_BGR2LUV
colorSpaceConversion['LAB']   = cv2.COLOR_BGR2LAB

# Axel's paths
folders["Highway"]  = "../../../datasetDeliver_2/highway/input/"
folders["HighwayGT"]  = "../../../datasetDeliver_2/highway/groundtruth/"

folders["Fall"]  = "../../../datasetDeliver_2/fall/input/"
folders["FallGT"]  = "../../../datasetDeliver_2/fall/groundtruth/"

folders["Traffic"]  = "../../../datasetDeliver_2/traffic/input/"
folders["TrafficGT"]  = "../../../datasetDeliver_2/traffic/groundtruth/"

folders["ownVideo"]  = "./videos/ownVideo3/"

folders["45"]  = "../../../datasetDeliver_2/kitty/45/"
folders["45GT"]  = "../../../datasetDeliver_2/kitty/45/gt/000045_10.png"

folders["157"]  = "../../../datasetDeliver_2/kitty/157/"
folders["157GT"]  = "../../../datasetDeliver_2/kitty/157/gt/000157_10.png"



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
mapping = highwayBinaryMapping1


################### OF  ####################
OFblockSize = 16
OFsearchArea = 16


################# Stab #####################
block_size = 16
area_size = 16
compensation = 'backward'  # or 'forward'
isReferenceImageFixed = False
