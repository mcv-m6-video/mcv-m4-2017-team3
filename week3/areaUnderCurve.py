import numpy as np
import cv2
import sys
sys.path.append('../')
import configuration as conf
import glob
import os
import evaluation as ev
import holefillingFunction as hf
import adaptativeColorGaussian as acg
import matplotlib.pyplot as plt
from numpy import trapz
import json

results_path = "./results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

gM_path = results_path + "/imagesGaussianModelling/"
if not os.path.exists(gM_path):
    os.makedirs(gM_path)

aG_path = results_path + "/imagesAdaptativeGaussianModelling/"  # "/imagesAdaptativeGaussian/"
if not os.path.exists(aG_path):
    os.makedirs(aG_path)


datasets = ["Highway","Fall","Traffic"]
datasetsGT = [el + "GT" for el in datasets]

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
    if d == "Fall":
        alfaPre = np.arange(0, 2.6, 0.2)
        alfaRange = np.arange(2.6, 4, 0.1)
        alfaPost = np.arange(4, 12, 2)
        alfaRange = np.concatenate((alfaPre, alfaRange, alfaPost), axis = 0)
    else:
        alfaPre = np.arange(0,1,0.2)
        alfaRange = np.arange(0.9, 3, 0.1)
        alfaPost = np.arange(3,12,2)
        alfaRange = np.concatenate((alfaPre, alfaRange, alfaPost), axis = 0)

    for alfa in alfaRange:
        if alfa in alfaResults:
            print "Alfa " + str(alfa) + " already done for this dataset. Skipping"
            continue
        print "Dataset " + d  +", alfa: "  +str(alfa)

        if conf.isShadowremoval:
            mu, sigma = acg.obtainGaussianModell(d, conf.OptimalColorSpaces["ShadowRemoval"])
            acg.foreground_substraction(d, dGT, mu, sigma, alfa, conf.OptimalRhoParameter[d], conf.OptimalColorSpaces["ShadowRemoval"], 80)
            TPi,TNi,FPi,FNi,precisioni,recalli,F1i = ev.evaluateFolder("./results/imagesAdaptativeGaussianModelling/",d)
        else:
            mu, sigma = acg.obtainGaussianModell(d, conf.OptimalColorSpaces[d])
            acg.foreground_substraction(d, dGT, mu, sigma, alfa, conf.OptimalRhoParameter[d], conf.OptimalColorSpaces[d], 80)
            TPi,TNi,FPi,FNi,precisioni,recalli,F1i = ev.evaluateFolder("./results/imagesAdaptativeGaussianModelling/",d)
        
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

# fig = plt.figure()
# plt.plot(alfaRange,[el['F1'] for el in fall],'r', label = 'Fall')
# plt.plot(alfaRange,[el['F1'] for el in traffic],'g', label = 'Traffic')
# plt.plot(alfaRange,[el['F1'] for el in highway],'b', label = 'Highway')
# plt.xlabel("Alfa")
# plt.ylabel("F1")
# plt.title("F1 vs alfa")
# plt.axis([0,5,0,1])
# lgd = plt.legend(bbox_to_anchor=(1.3,0.8))
# plt.savefig("task1.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()

#AUC:

y = np.array([el['precision'] for el in fall])
x = np.array([el['recall'] for el in fall])
AUC = np.trapz(x,y)
print " AUC of Fall: " + str(AUC)

y = np.array([el['precision'] for el in traffic])
x = np.array([el['recall'] for el in traffic])
AUC = trapz(x,y)
print " AUC of Traffic: " + str(AUC)

y = np.array([el['precision'] for el in highway])
x = np.array([el['recall'] for el in highway])
AUC = trapz(x,y)
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
