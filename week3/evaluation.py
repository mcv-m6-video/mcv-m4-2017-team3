#!/usr/bin/env

import cv2
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import collections
import os
import sys
sys.path.append('../')
import configuration as conf

operativeSystem = os.name

# Evaluates one single frame, according to the mapping defined in the
# configuration file. Returns the confusion matrix, precision, recall and f-1
# score for all the labels. (The background is on the last one)
def evaluateImage(queryFile,gtFile):
    queryImg = cv2.imread(queryFile,0)
    gt = cv2.imread(gtFile,0)
    if (queryImg.size <= 0):
        print "Image not found"
        return 0
    if (gt is None):
        print "Groundtruth " + gtFile + " not found"
        return 0

    predictionVector = []
    gtVector = []
    predictionValues = np.unique(queryImg).tolist()
    if predictionValues == [0,255]:
        queryImg = queryImg/255
    for pixel in range(0,queryImg.size):
        predictionVector.append(queryImg.flat[pixel])
    for pixel in range(0,gt.size):
        gtVector.append(conf.mapping[gt.flat[pixel]])

    if len(set(predictionVector)) != len(set(gtVector)):

        predictionVector = [el if el != 2 else 1 for el in predictionVector]

    confMat = confusion_matrix(gtVector,predictionVector)
    if len(confMat) == 1:
        if list(set(gtVector))[0] == 0:
            confMat = np.ndarray((2,2),dtype = np.uint8)
            TP = 0
            TN = predictionVector.count(0)
            FP = predictionVector.count(1)
            FN = 0
            confMat[0][0] = predictionVector.count(0)
            confMat[0][1] = 0
            confMat[1][0] = predictionVector.count(1)
            confMat[1][1] = 0
            precision = np.array([0, float(TP)/float(TP+FP)]) if (TP+FP) != 0 else np.array([0, 0])
            recall = np.array([0, 0])
            fscore = np.array([0, 0])
    else:
        precision, recall, fscore, support = score(gtVector, predictionVector)
    try:
        auc = roc_auc_score(gtVector, predictionVector)
    except:
        print "Warning: only one class present in y_true. ROC AUC score is not defined in that case."
        auc = 0.0

    return confMat,precision,recall,fscore,auc


# Evaluates a whole folder, using the groundtruth and image prefixes of configuration file
def evaluateFolder(folderPath,ID = "Highway"):
    queryFiles = sorted(glob.glob(folderPath + ID + "*"))
    results = dict()
    numItems = len(queryFiles)
    auc = 0.0
    for idx, queryFile in enumerate(queryFiles[:]):
        file_name = queryFile
        #OS dependant writing
        if operativeSystem == 'posix':
            #posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
            base_name = file_name.split("/")[-1][2 + len(ID):-4]
            gtFile = conf.folders[ID+"GT"] + 'gt' + base_name + '.png'
        else:
            #say hello to propietary software
            base_name = file_name.split("\\")[-1][2 + len(ID):-4]
            gtFile = conf.folders[ID+"GT"] + 'gt' + base_name + '.png'
        # print ('===================')
        # print (gtFile)
        confusion,precision,recall,f1,auc = evaluateImage(queryFile,gtFile)

        accuracy = float(confusion.trace())/np.sum(confusion)
        results[queryFile[len(folderPath):]] = {"Confusion Matrix":confusion.tolist(),"Precision":precision.tolist(),"Recall":recall.tolist(),"Accuracy":accuracy,"Fscore":f1.tolist()}


    with open(conf.outputFile,"w") as f:
        json.dump(results,f)

    if (conf.mapping["Classes"] == 2):
        TN = sum([results[el]["Confusion Matrix"][0][0] for el in results.keys()])
        FP = sum([results[el]["Confusion Matrix"][0][1] for el in results.keys()])
        FN = sum([results[el]["Confusion Matrix"][1][0] for el in results.keys()])
        TP = sum([results[el]["Confusion Matrix"][1][1] for el in results.keys()])

        if (TP+FP) > 0.0:
            precision = float(TP)/(TP+FP)
        else:
            precision = 0.0

        if (TP+FN) > 0.0:
            recall = float(TP)/(TP+FN)
        else:
            recall = 0.0

    else:
        #this case is weird and won't be used (hopefully)
        TP = 0
        FP = 0
        FN = 0
        FN = 0
        precision = 0
        recall = 0
        # Precision and recall for each label:
        for label in range(0,conf.mapping["Classes"]):
            TP = sum([results[el]["Confusion Matrix"][label][label] for el in results.keys()])
            for i in range(0,conf.mapping["Classes"]):
                if(i == label):
                    continue
                FP += sum([results[el]["Confusion Matrix"][i][label] for el in results.keys()])
                FN += sum([results[el]["Confusion Matrix"][label][i] for el in results.keys()])
            # mean precision and recall as a serie (arithmetic, this may give a headache)
            if (TP+FP) > 0.0:
                precision = (precision*(label) + float(TP)/(TP+FP))/(label+1)
            else:
                precision = 0.0

            if (TP+FN) > 0.0:
                recall = (recall*(label) + float(TP)/(TP+FN))/(label+1)
            else:
                recall = 0.0

    beta = 1

    if (precision + recall) > 0.0:
        F1 = (1+pow(beta,2))*precision*recall/(pow(beta,2)*precision + recall)
    else:
        F1 = 0.0


    return TP,TN,FP,FN,precision,recall,F1


# Plots the evolution of the video sequence (task 2 basically)
def temporalEvaluation():

    #Reading the file from task 1
    with open("./results/test_A_results.json") as f:
        results = json.load(f)

    orderedResults = collections.OrderedDict(sorted(results.items()))
    if (conf.mapping["Classes"] == 2):
        TP = [orderedResults[el]["Confusion Matrix"][1][1] for el in orderedResults.keys()]
        P  = [orderedResults[el]["Confusion Matrix"][1][1] + orderedResults[el]["Confusion Matrix"][1][0] for el in orderedResults.keys()]
        F1 = [orderedResults[el]["Fscore"][-1] for el in orderedResults.keys()]
    nSamples = len(orderedResults)


    with open("./results/test_B_results.json") as f:
        results = json.load(f)

    orderedResults = collections.OrderedDict(sorted(results.items()))
    if (conf.mapping["Classes"] == 2):
        TPb = [orderedResults[el]["Confusion Matrix"][1][1] for el in orderedResults.keys()]

        F1b = [orderedResults[el]["Fscore"][-1] for el in orderedResults.keys()]
    nSamples = len(orderedResults)

    x = np.linspace(0,nSamples,nSamples)

    plt.figure(1)
    #plt.fill_between(x,0,P,facecolor='green')
    #plt.fill_between(x,0,TP,facecolor='blue')
    plt.plot(x,TP,'b',label = 'Method A')
    plt.plot(x,TPb,'c',label = 'Method B')
    plt.plot(x,P,'g',label = 'Total Foreground \n pixels')
    plt.xlabel("Frames")
    plt.ylabel("Number of pixels")
    plt.axis([0,200,0,max(P)])

    lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

    for key in ev.highwayEvents.keys():
        plt.plot((key-1200,key-1200),(0,max(P)),'r-')
        plt.text(key-1205, float(max(P)/2) , ev.highwayEvents[key], rotation=90)

    plt.savefig("./temporalEvaluationTPvsP.png",bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(2)
    plt.plot(F1,'b',label = 'Method A')
    plt.plot(F1b,'c',label = 'Method B')
    plt.xlabel("Frames")
    plt.ylabel("Number of pixels")

    lgd =  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    for key in ev.highwayEvents.keys():
        plt.plot((key-1200,key-1200),(0,1),'r-')
        plt.text(key-1205, 0.5 , ev.highwayEvents[key], rotation=90)

    plt.savefig("./temporalEvaluationF1.png",bbox_extra_artists=(lgd,), bbox_inches='tight')

def methodsComparison():
    gt = cv2.imread("datasets/highway/groundtruth/gt001240.png")
    imA = cv2.imread("results/highway/test_A_001240.png")
    imB = cv2.imread("results/highway/test_B_001240.png")

    height,width,channel = gt.shape

    for w in range(0,width-1):
        for h in range(0,height-1):
            if(conf.mapping[gt[h,w,0]] == 0):
                if(imA[h,w,0] == 1):
                    imA[h,w,:] = [0,0,255]
                if(imB[h,w,0] == 1):
                    imB[h,w,:] = [0,0,255]
            if(conf.mapping[gt[h,w,0]] == 1):
                if(imA[h,w,0] == 1):
                    imA[h,w,:] = [0,255,0]
                else:
                    imA[h,w,:] = [255,0,0]
                if(imB[h,w,0] == 1):
                    imB[h,w,:] = [0,255,0]
                else:
                    imB[h,w,:] = [255,0,0]

    f, (ax1, ax2, ax3) = plt.subplots(nrows = 1,ncols = 3, sharex=False, sharey=False )

    ax1.imshow(gt)
    ax1.set_title("Ground truth")
    plt.setp(ax1.get_xticklabels() , visible=False)
    plt.setp(ax1.get_yticklabels() , visible=False)
    ax2.imshow(imA)
    ax2.set_title("Method A")
    plt.setp(ax2.get_xticklabels() , visible=False)
    plt.setp(ax2.get_yticklabels() , visible=False)
    ax3.imshow(imB)
    ax3.set_title("Method B")
    plt.setp(ax3.get_xticklabels() , visible=False)
    plt.setp(ax3.get_yticklabels() , visible=False)

    red_patch = mpatches.Patch(color='red', label='FN')
    blue_patch = mpatches.Patch(color='blue', label='FP')
    green_patch = mpatches.Patch(color='green', label='TP')

    lgd = plt.legend(handles = [green_patch,blue_patch,red_patch],bbox_to_anchor=(1.8,0.8))

    plt.savefig("methodscomparison.png",bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
    #evaluateFolder("./results/testAB/highway/")
    #temporalEvaluation()
    methodsComparison()
