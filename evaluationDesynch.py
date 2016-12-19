#!/usr/bin/env

import numpy as np
import cv2
import glob
import configuration as conf
import evaluation
import json
import matplotlib.pyplot as plt
import os.path

# Evaluates a whole folder, using the groundtruth and image prefixes of configuration file
def evaluateFolderDesynch(folderPath):
    queryFiles = sorted(glob.glob(folderPath + conf.queryImgPrefix + "*"))

    if(os.path.exists(conf.outputFileDelay)):
        with open(conf.outputFileDelay) as f:
            results = json.load(f)
    else:
        results = []
        for delay in range(-25,25):

            print "\n##########################\n# Desyncrhonization is " + str(delay) + " #\n##########################\n"
            res = dict()
            for idx,queryFile in enumerate(queryFiles):
                gtFile = conf.gtFolder + conf.gtPrefix + (str(int(queryFile[len(folderPath) + len(conf.queryImgPrefix):-4]) + delay)).zfill(len(queryFile[len(folderPath) + len(conf.queryImgPrefix):-4])) + conf.gtExtension
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

        with open(conf.outputFileDelay,"w") as f:
            json.dump(results,f)

    f1scoreToPlot = [results[el]["F1"] for el in range(len(results))]

    plt.figure(1)
    plt.plot(f1scoreToPlot)
    plt.show()




if __name__ == "__main__":
    evaluateFolderDesynch("./results/highway/")
