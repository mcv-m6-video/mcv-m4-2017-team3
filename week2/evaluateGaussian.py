# -*- coding: utf-8 -*-

import evaluation as ev
import configuration as conf
import numpy as np
import gaussianModelling as gm 
import adaptativeGaussian as ag
import matplotlib.pyplot as plt

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

aG_path = results_path + "/imagesAdaptativeGaussian/"
if not os.path.exists(aG_path):
    os.makedirs(aG_path)


# Compute the evaluation
for alfa in alfas:
    
    # One gaussian
    dataset    = "Highway"
    datasetGT  = "HighwayGT"
    colorSpace = 'gray' # 'gray', 'HSV', 'YCrCb', 'BGR'
    gm.obtainGaussianModell(dataset, datasetGT, colorSpace, alfa)
    TPi,TNi,FPi,FNi,precisioni,recalli,F1i,auci = ev.evaluateFolder(gM_path)

    TP.append(TPi)
    TN.append(TNi)
    FP.append(FPi)
    FN.append(FNi)
    precision.append(precisioni)
    recall.append(recalli)
    F1.append(F1i)
    auc.append(auci)

    # Adaptative gaussian
    mu, sigma = ag.obtainGaussianModell("Highway")
    ag.foreground_substraction("Highway", "HighwayGT", mu, sigma, alfa, 0) # rho equal to 0, in order to find the optimal alpha
    aux,aux,aux,aux,aux,aux,F1i_adaptative,aux = ev.evaluateFolder(aG_path)
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
    mu, sigma = ag.obtainGaussianModell("Highway")
    ag.foreground_substraction("Highway", "HighwayGT", mu, sigma, optAlpha, rho)
    aux, aux, aux, aux, aux, aux, F1 = ev.evaluateFolder("./results/imagesAdaptativeGaussian/")

    if F1 > bestF1:
        bestF1 = F1
        optRho = rho

    print ('--- Rho: ' + str(rho) + ' --- ' + 'optimal Alpha: ' + str(optAlpha))
    print ('--- F1 Adaptative Gaussian Model: ' + str(F1))

# Plot the features
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# plt.title('TP FN TN FP for ONE GAUSSIAN')
# ax1.plot(TP,color='red')
# ax1.plot(FP,color='blue')
# ax1.plot(TN,color='green')
# ax1.plot(FN,color='black')
# ax1.set_xlabel('Threshold')
# ax1.set_ylabel('Number of pixels')

# ax2 = fig.add_subplot(1,2,2)
# plt.title('F-measure depending on threshold')
# ax2.plot(F1,color='red')
# ax2.plot(F1_adaptative,color='blue')
# ax2.set_xlabel('Threshold')
# ax2.set_ylabel('F1-Measure')
# fig.show()

# Task 1.3 Precision vs Recall curve and AUC
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# plt.title('Precision vs recall')
# ax1.plot(recall, precision,color='green')

#ax1 = fig.add_subplot(1,2,2)
#plt.title('Area under the curve')
#ax1.plot(auc, precision,color='green')
