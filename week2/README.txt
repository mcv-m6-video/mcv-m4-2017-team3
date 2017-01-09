Deliver week 2. 

Background modelling. 

This READ.txt file helps us to set up the environment in order to be able to run it properly. 

CREATING FOLDERS

A set of folder must be created. Python code will access to them, if possible.

Into week 2 folder, a results and a video folders must be created.

Into the results folder, three folders must be created in order to record each result of the experiments.
- imagesAdaptativeGaussian
- imagesGaussianModelling
- StaufferAndGrimson

SETTING PATH TO DATASET

The dataset path can be variable depending on your environment. 
The file configuration.py contains a set of variables where you are able to configurate them as you wish. 

In our case, these are:
folders["Highway"]  = "../../../datasetDeliver_2/highway/input/"
folders["HighwayGT"]  = "../../../datasetDeliver_2/highway/groundtruth/"

folders["Fall"]  = "../../../datasetDeliver_2/fall/input/"
folders["FallGT"]  = "../../../datasetDeliver_2/fall/groundtruth/"

folders["Traffic"]  = "../../../datasetDeliver_2/traffic/input/"
folders["TrafficGT"]  = "../../../datasetDeliver_2/traffic/groundtruth/"
