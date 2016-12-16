#!/usr/bin/env

import cv2
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
import json

import configuration as conf
import highwayEvents as ev

# Evaluates one single frame, according to the mapping defined in the
# configuration file. Returns the confusion matrix, precision, recall and f-1
# score for all the labels. (The background is on the last one)
def evaluateImage(queryFile,gtFile):
    queryImg = cv2.imread(queryFile,0)
    gt = cv2.imread(gtFile,0)

    if (queryImg.size <= 0):
        print "Image not found"
        return 0
    if (gt.size <= 0):
        print "Groundtruth not found"
        return 0


    predictionVector = []
    gtVector = []

    for pixel in range(0,queryImg.size):
        predictionVector.append(queryImg.flat[pixel])
    for pixel in range(0,gt.size):
        gtVector.append(conf.highwayMapping[gt.flat[pixel]])

    confMat = confusion_matrix(gtVector,predictionVector)
    precision, recall, fscore, support = score(gtVector, predictionVector)

    return confMat,precision,recall,fscore


# Evaluates a whole folder, using the groundtruth and image prefixes of configuration file
def evaluateFolder(folderPath):
    queryFiles = sorted(glob.glob(folderPath + conf.queryImgPrefix + "*"))

    results = dict()
    for idx,queryFile in enumerate(queryFiles):
        print str(1+idx) + "/" + str(len(queryFiles))
        gtFile = conf.gtFolder + conf.gtPrefix + queryFile[len(folderPath) + len(conf.queryImgPrefix):-4] + conf.gtExtension
        confusion,precision,recall,f1 = evaluateImage(queryFile,gtFile)
        accuracy = float(confusion.trace())/np.sum(confusion)
        results[queryFile[len(folderPath):]] = {"Confusion Matrix":confusion.tolist(),"Precision":precision.tolist(),"Recall":recall.tolist(),"Accuracy":accuracy,"Fscore":f1.tolist()}

    with open(conf.outputFile,"w") as f:
        json.dump(results,f)

    if (conf.highwayMapping["Classes"] == 2):
        TN = sum([results[el]["Confusion Matrix"][0][0] for el in results.keys()])
        FP = sum([results[el]["Confusion Matrix"][0][1] for el in results.keys()])
        FN = sum([results[el]["Confusion Matrix"][1][0] for el in results.keys()])
        TP = sum([results[el]["Confusion Matrix"][1][1] for el in results.keys()])

        precision = float(TP)/(TP+FP)
        recall = float(TP)/(TP+FN)

    else:
        #this case is weird and won't be used (hopefully)
        TP = 0
        FP = 0
        FN = 0
        FN = 0
        precision = 0
        recall = 0
        # Precision and recall for each label:
        for label in range(0,conf.highwayMapping["Classes"]):
            TP = sum([results[el]["Confusion Matrix"][label][label] for el in results.keys()])
            for i in range(0,conf.highwayMapping["Classes"]):
                if(i == label):
                    continue
                FP += sum([results[el]["Confusion Matrix"][i][label] for el in results.keys()])
                FN += sum([results[el]["Confusion Matrix"][label][i] for el in results.keys()])
            # mean precision and recall as a serie (arithmetic, this may give a headache)
            precision = (precision*(label) + float(TP)/(TP+FP))/(label+1)
            recall = (recall*(label) + float(TP)/(TP+FN))/(label+1)

    beta = 1
    F1 = (1+pow(beta,2))*precision*recall/(pow(beta,2)*precision + recall)

    print TP,TN,FP,FN
    print precision,recall,F1


# Plots the evolution of the video sequence (task 2 basically)
def temporalEvaluation():

    #Reading the file from task 1
    with open(conf.outputFile) as f:
        results = json.load(f)

    if (conf.highwayMapping["Classes"] == 2):
        TP = [results[el]["Confusion Matrix"][1][1] for el in results.keys()]
        P  = [results[el]["Confusion Matrix"][1][1] + results[el]["Confusion Matrix"][1][0] for el in results.keys()]
        F1 = [results[el]["Fscore"][-1] for el in results.keys()]


    plt.figure(1)
    plt.plot(TP)
    plt.plot(P)
    for key in ev.highwayEvents.keys():
        plt.plot((key-1200,key-1200),(0,max(P)),'r-')
        plt.text(key-1205, 0.8*max(P), ev.highwayEvents[key], rotation=90)

    plt.show()

    plt.figure(2)
    plt.plot(F1)
    for key in ev.highwayEvents.keys():
        plt.plot((key-1200,key-1200),(0,1),'r-')
        plt.text(key-1205, 0.8, ev.highwayEvents[key], rotation=90)

    plt.show()





if __name__ == "__main__":
    #evaluateFolder("./results/testAB/highway/")
    temporalEvaluation()
