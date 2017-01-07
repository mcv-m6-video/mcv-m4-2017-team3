# -*- coding: utf-8 -*-

import evaluation as ev
import configuration as conf
import numpy as np
import old_gaussianModelling as gm 
import adaptativeGaussian as ag
import matplotlib.pyplot as plt

alfas = np.arange(0,15)
TP = []
TN = []
FP = []
FN = []
precision = []
recall = []
F1 = []
F1_adaptative = []

# Compute the evaluation
for alfa in alfas:
    print ('--- Alfa: ' + str(alfa))
    
    # One gaussian
    gm.obtainGaussianModell("Highway", alfa)
    TPi,TNi,FPi,FNi,precisioni,recalli,F1i = ev.evaluateFolder("./results/images/")
    
    TP.append(TPi)
    TN.append(TNi)
    FP.append(FPi)
    FN.append(FNi)
    precision.append(precisioni)
    recall.append(recalli)
    F1.append(F1i)
    
    # Adaptative gaussian
    mu, sigma = ag.obtainGaussianModell("Highway", alfa)
    ag.foreground_substraction("Highway", mu, sigma, alfa)
    aux,aux,aux,aux,aux,aux,F1i_adaptative = ev.evaluateFolder("./results/images/")
    F1_adaptative.append(F1i_adaptative)


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
