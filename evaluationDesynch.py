#!/usr/bin/env

import numpy as np
import cv2
import glob
import configuration as conf
import evaluation
import json
import matplotlib.pyplot as plt
import os.path
import collections

# Evaluates a whole folder, using the groundtruth and image prefixes of configuration file
def evaluateFolderDesynch(folderPath,queryImgPrefix,fileDelay):
    if(os.path.exists(fileDelay)):
        with open(fileDelay) as f:
            results = json.load(f)

    else:
        queryFiles = sorted(glob.glob(folderPath + queryImgPrefix + "*"))
        results = []
        for delay in range(-25,25):

            print "\n##########################\n# Desyncrhonization is " + str(delay) + " #\n##########################\n"
            res = dict()
            for idx,queryFile in enumerate(queryFiles):
                gtFile = conf.gtFolder + conf.gtPrefix + (str(int(queryFile[len(folderPath) + len(queryImgPrefix):-4]) + delay)).zfill(len(queryFile[len(folderPath) + len(queryImgPrefix):-4])) + conf.gtExtension
                print "Comparing file " + queryFile[len(folderPath):] + " vs gt " + gtFile
                confusion,precision,recall,f1 = evaluation.evaluateImage(queryFile,gtFile)
                accuracy = float(confusion.trace())/np.sum(confusion)
                res[queryFile[len(folderPath):]] = {"Confusion Matrix":confusion.tolist(),"Precision":precision.tolist(),"Recall":recall.tolist(),"Accuracy":accuracy,"Fscore":f1.tolist()}
            if (conf.highwayMapping["Classes"] == 2):
                TN = sum([res[el]["Confusion Matrix"][0][0] for el in res.keys()])
                FP = sum([res[el]["Confusion Matrix"][0][1] for el in res.keys()])
                FN = sum([res[el]["Confusion Matrix"][1][0] for el in res.keys()])
                TP = sum([res[el]["Confusion Matrix"][1][1] for el in res.keys()])

                precision = float(TP)/(TP+FP)
                recall = float(TP)/(TP+FN)
                beta = 1
                F1 = (1+pow(beta,2))*precision*recall/(pow(beta,2)*precision + recall)

            delayResults = dict()
            delayResults["Delay"] = delay
            delayResults["Precision"] = precision
            delayResults["F1"] = F1
            delayResults["TP"] = TP
            delayResults["TN"] = TN
            delayResults["FP"] = FP
            delayResults["FN"] = FN
            delayResults["Results"] = res
            results.append(delayResults)

        with open(fileDelay,"w") as f:
            json.dump(results,f)

    return results

def evaluateDesynch():
    resultsA = evaluateFolderDesynch("./results/highway/","test_A_","./results/test_A_delayResults.json")
    resultsB = evaluateFolderDesynch("./results/highway/","test_B_","./results/test_B_delayResults.json")

    f1scoreToPlotA = [resultsA[el]["F1"] for el in range(len(resultsA))]
    f1scoreToPlotB = [resultsB[el]["F1"] for el in range(len(resultsB))]

    x = np.arange(-25,25,1)
    major_ticks = np.arange(-25, 30, 5)

    plt.figure(1)
    plt.plot(x,f1scoreToPlotA,'g',label = "Method A")
    plt.plot(x,f1scoreToPlotB,'b',label = "Method B")
    lgd = plt.legend(bbox_to_anchor=(1.3,0.8))
    plt.xlabel("Desynchronization frames")
    plt.title("Desynchronization of f1")
    plt.ylabel("f1 score")
    plt.xticks(major_ticks)
    plt.axis([-25,25,0,1])
    plt.savefig("delay.png",bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == "__main__":
    evaluateDesynch()
