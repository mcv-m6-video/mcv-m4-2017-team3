import cv2
import os
import numpy as np
import json
from numpy import trapz
import sys
import glob
#sys.path.append('../week2')
sys.path.append('../tools')
import evaluation as ev
import bwareaopen

operativeSystem = os.name

dataset = '../../../datasets-week3/trafficDataset/'
files = glob.glob(dataset + '*')
ID = "Traffic"

results = dict()
TP = []
TN = []
FP = []
FN = []
Precision = []
Recall = []
F1  = []


for p in range(0,200,10):
    print "Filtering objects smaller than " + str(p)
    for f in files:
        out = bwareaopen.bwareaopen(f,p);
        if operativeSystem == 'posix':
            #posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
            cv2.imwrite('results/areaFiltering/' + ID + f.split('/')[-1],out)

        else:
            #say hello to propietary software
            cv2.imwrite('results/areaFiltering/' + ID + f.split('\\')[-1], out)
    tp,tn,fp,fn,precision,recall,f1 = ev.evaluateFolder('results/areaFiltering/',ID)
    for f in glob.glob('results/areaFiltering/*'):
        os.remove(f)
    print tp,tn,fp,fn,precision,recall,f1
    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    Precision.append(precision)
    Recall.append(recall)
    F1.append(f1)

results['TP'] = TP
results['TN'] = TN
results['FP'] = FP
results['FN'] = FN
results['Precision'] = Precision
results['Recall'] = Recall
results['F1'] = F1



with open('results/resultsAreaFiltering' + ID + '.json','w') as f:
    json.dump(results,f)
