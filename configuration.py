#!/usr/bin/env

# Ground truth variables
gtFolder = "./datasets/highway/groundtruth/"
gtPrefix = "gt"
gtExtension = ".png"
# Query Image prefix
queryImgPrefix = "test_A_"

# Results logging
outputFolder = "./results/"
outputFile = outputFolder + queryImgPrefix + "results.json"
outputFileDelay = outputFolder + queryImgPrefix + "delayResults.json"


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


# Optical Flow variables
gtOFFolder = "./datasets/Optical_Flow/gt/"
imagesOFFolder = "./datasets/Optical_Flow/images/"
imagesOF = ["000045_10", "000157_10"]
resultsOFFolder = "./results/resultsOF/"

# Optical flow visualization variables
OFSquareSize = pow(2,3)
nBins = 9
