# -*- coding: utf-8 -*-

import os
import evaluation as ev
import configuration as conf
import numpy as np
import json
import gaussianModelling as gm
import adaptativeGaussian as ag
import matplotlib.pyplot as plt
from numpy import trapz
import os

def evaluateTask1(colorSpace = 'gray'):
    datasets =['Fall'] # ["Highway","Fall","Traffic"]
    datasetsGT = [el + "GT" for el in datasets]
    alfaRange = np.arange(0,5.001,0.5)
    rhoRange = np.arange(0,1.1,0.1)

    if(os.path.exists('results/task1.json')):
        with open('results/task1.json') as f:
            datasetsResults = json.load(f)
            print "Dataset results loaded"
    else:
        datasetsResults = dict()

    for d,dGT in zip(datasets,datasetsGT):
        if d not in datasetsResults.keys():
            datasetsResults[d] = []
        alfaResults = [el['Alfa'] for el in datasetsResults[d]]

        for alfa in alfaRange:
            if alfa in alfaResults:
                print "Alfa " + str(alfa) + " already done for this dataset. Skipping"
                continue
            print "Dataset " + d  +", alfa: "  +str(alfa)
            gm.obtainGaussianModell(d, dGT, colorSpace, alfa)
            TPi,TNi,FPi,FNi,precisioni,recalli,F1i,_ = ev.evaluateFolder("./results/imagesGaussianModelling/",d)
            performance = dict()
            performance['Alfa'] = alfa
            performance['TP'] = TPi
            performance['TN'] = TNi
            performance['FP'] = FPi
            performance['FN'] = FNi
            performance['precision'] = precisioni
            performance['recall'] = recalli
            performance['F1'] = F1i
            print F1i
            datasetsResults[d].append(performance)

            with open('results/task1.json',"w") as f:
                json.dump(datasetsResults,f)

    alfaRange = sorted([el['Alfa'] for el in datasetsResults['Fall']])

    fall = sorted(datasetsResults['Fall'],key = lambda k: k['Alfa'])
    traffic = sorted(datasetsResults['Traffic'],key = lambda k: k['Alfa'])
    highway = sorted(datasetsResults['Highway'],key = lambda k: k['Alfa'])


    for el in fall:
        print str(el['Alfa']) + ': ' + str(el['F1'])

    fig = plt.figure()
    plt.plot(alfaRange,[el['F1'] for el in fall],'r', label = 'Fall')
    plt.plot(alfaRange,[el['F1'] for el in traffic],'g', label = 'Traffic')
    plt.plot(alfaRange,[el['F1'] for el in highway],'b', label = 'Highway')
    plt.xlabel("Alfa")
    plt.ylabel("F1")
    plt.title("F1 vs alfa")
    plt.axis([0,5,0,1])
    lgd = plt.legend(bbox_to_anchor=(1.3,0.8))
    plt.savefig("task1.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    #AUC:

    y = np.array([el['precision'] for el in fall])
    x = np.array([el['recall'] for el in fall])
    AUC = trapz(y,x)
    print " AUC of Fall: " + str(AUC)

    y = np.array([el['precision'] for el in traffic])
    x = np.array([el['recall'] for el in traffic])
    AUC = trapz(y,x)
    print " AUC of Traffic: " + str(AUC)

    y = np.array([el['precision'] for el in highway])
    x = np.array([el['recall'] for el in highway])
    AUC = trapz(y,x)
    print " AUC of Highway: " + str(AUC)


    fig = plt.figure()
    plt.plot([el['recall'] for el in fall],[el['precision'] for el in fall],'r', label = 'Fall')
    plt.plot([el['recall'] for el in traffic],[el['precision'] for el in traffic],'g', label = 'Traffic')
    plt.plot([el['recall'] for el in highway],[el['precision'] for el in highway],'b', label = 'Highway')
    plt.axis([0,1,0,1])
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    lgd = plt.legend(bbox_to_anchor=(1.3,0.8))
    plt.savefig("task1-curves.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def evaluateResults():
    alfas = np.arange(0,15)
    rhos = np.arange(0,1.1,0.1)
    TP = []
    TN = []
    FP = []
    FN = []
    precision = []
    recall = []
    F1 = []
    F1_adaptative = []
    bestF1 = 0
    optAlpha = 0

    # Compute the evaluation
    '''
        for alfa in alfas:

        # One gaussian
        dataset    = "Traffic"
        datasetGT  = "TrafficGT"
        colorSpace = 'gray' # 'gray', 'HSV', 'YCrCb', 'BGR'
        gm.obtainGaussianModell(dataset, datasetGT, colorSpace, alfa)

        TPi,TNi,FPi,FNi,precisioni,recalli,F1i = ev.evaluateFolder("./results/imagesGaussianModelling/","Traffic")

        TP.append(TPi)
        TN.append(TNi)
        FP.append(FPi)
        FN.append(FNi)
        precision.append(precisioni)
        recall.append(recalli)
        F1.append(F1i)

        # Adaptative gaussian
        mu, sigma = ag.obtainGaussianModell("Traffic")
        ag.foreground_substraction("Traffic", "TrafficGT", mu, sigma, alfa, 0) # rho equal to 0, in order to find the optimal alpha
        aux,aux,aux,aux,aux,aux,F1i_adaptative = ev.evaluateFolder("./results/imagesAdaptativeGaussian/")
        F1_adaptative.append(F1i_adaptative)

        if F1i_adaptative > bestF1:
            bestF1 = F1i_adaptative
            optAlpha = alfa

        print ('--- Alfa: ' + str(alfa) + ' --- Rho: 0')
        print ('--- F1 Gaussian Model: ' + str(F1i))
        print ('--- F1 Adaptative Gaussian Model: ' + str(F1i_adaptative))
        '''

    bestF1 = 0
    optAlpha = 1.8
    for rho in rhos:
        # Adaptative gaussian
        mu, sigma = ag.obtainGaussianModell("Highway")
        ag.foreground_substraction("Highway", "HighwayGT", mu, sigma, optAlpha, rho)
        aux, aux, aux, aux, aux, aux, F1 = ev.evaluateFolder("./results/imagesAdaptativeGaussianModelling/")

        if F1 > bestF1:
            bestF1 = F1
            optRho = rho

        print ('--- Rho: ' + str(rho) + ' --- ' + 'optimal Alpha: ' + str(optAlpha))
        print ('--- F1 Adaptative Gaussian Model: ' + str(F1))
alfas = np.arange(0,15)
rhos = np.arange(0,1.1,0.1)
TP = []
TN = []
FP = []
FN = []
precision = []
recall = []
F1 = []
F1_adaptative = []
auc = []
bestF1 = 0
optAlpha = 0

# Check if 'results' folder exists.
results_path = "./results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

gM_path = results_path + "/imagesGaussianModelling/"
if not os.path.exists(gM_path):
    os.makedirs(gM_path)

aG_path = results_path + "/imagesAdaptativeGaussianModelling/" #"/imagesAdaptativeGaussian/"
if not os.path.exists(aG_path):
    os.makedirs(aG_path)


# Compute the evaluation
for alfa in alfas:

    # One gaussian
    dataset    = "Highway"
    datasetGT  = "HighwayGT"
    colorSpace = 'gray' # 'gray', 'HSV', 'YCrCb', 'BGR'
    gm.obtainGaussianModell(dataset, datasetGT, colorSpace, alfa)
    TPi,TNi,FPi,FNi,precisioni,recalli,F1i,auci = ev.evaluateFolder(gM_path, datasetGT)

    TP.append(TPi)
    TN.append(TNi)
    FP.append(FPi)
    FN.append(FNi)
    precision.append(precisioni)
    recall.append(recalli)
    F1.append(F1i)
    auc.append(auci)

    # Adaptative gaussian
    mu, sigma = ag.obtainGaussianModell(dataset)
    ag.foreground_substraction(dataset, datasetGT, mu, sigma, alfa, 0) # rho equal to 0, in order to find the optimal alpha
    aux,aux,aux,aux,aux,aux,F1i_adaptative,aux = ev.evaluateFolder(aG_path, datasetGT)
    F1_adaptative.append(F1i_adaptative)

    if F1i_adaptative > bestF1:
        bestF1 = F1i_adaptative
        optAlpha = alfa

    print ('--- Alfa: ' + str(alfa) + ' --- Rho: 0')
    print ('--- F1 Gaussian Model: ' + str(F1i))
    print ('--- F1 Adaptative Gaussian Model: ' + str(F1i_adaptative))

bestF1 = 0
for rho in rhos:
    # Adaptative gaussian
    mu, sigma = ag.obtainGaussianModell(dataset)
    ag.foreground_substraction(dataset, datasetGT, mu, sigma, optAlpha, rho)
    aux, aux, aux, aux, aux, aux, F1, aux = ev.evaluateFolder(aG_path, datasetGT)

    if F1 > bestF1:
        bestF1 = F1
        optRho = rho

    print ('--- Rho: ' + str(rho) + ' --- ' + 'optimal Alpha: ' + str(optAlpha))
    print ('--- F1 Adaptative Gaussian Model: ' + str(F1))

'''
# Plot the features
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
plt.title('TP FN TN FP for ONE GAUSSIAN')
ax1.plot(TP,color='red')
ax1.plot(FP,color='blue')
ax1.plot(TN,color='green')
ax1.plot(FN,color='black')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Number of pixels')

ax2 = fig.add_subplot(1,2,2)
plt.title('F-measure depending on threshold')
ax2.plot(F1,color='red')
ax2.plot(F1_adaptative,color='blue')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('F1-Measure')
#fig.show()

fig.savefig('Plot1.png')

# Task 1.3 Precision vs Recall curve and AUC

# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# plt.title('Precision vs recall')
# ax1.plot(recall, precision,color='green')

#ax1 = fig.add_subplot(1,2,2)
#plt.title('Area under the curve')
#ax1.plot(recall, precision,color='green')


#We can see very different curves for each dataset.

if __name__ == "__main__":
    evaluateResults()
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1,2,1)
    plt.title('Precision vs recall')
    ax1.plot(recall, precision,color='green')

    ax2 = fig2.add_subplot(1,2,2)
    plt.title('Area under the curve')
    ax2.plot(auc, precision,color='green')

    fig2.savefig('Plot2.png')
'''
