import cv2


ID = "Own" # # Highway | Traffic | Video | NewVideo (In case anyone wants to upload its own video)
# Mode regarding video or folders:
mode = 'folder' # video | folder
# A .mp4 file if mode is video, a folder with all the frames if mode is folder
inputData = 'video.mp4'


#ColorSpace to work
colorSpaceConversion = {}
colorSpaceConversion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConversion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConversion['HLS']   = cv2.COLOR_BGR2HLS
colorSpaceConversion['gray']  = cv2.COLOR_BGR2GRAY
colorSpaceConversion['LUV']   = cv2.COLOR_BGR2LUV
colorSpaceConversion['LAB']   = cv2.COLOR_BGR2LAB

colorSpace = 'HSV'

OptimalAlphaParameter = {}
OptimalAlphaParameter["Highway"]   = 1.8
OptimalAlphaParameter["Traffic"]   = 1.9
OptimalAlphaParameter["Own"]     = 4.8

OptimalRhoParameter = {}
OptimalRhoParameter["Highway"]   = 0.04
OptimalRhoParameter["Traffic"]   = 0.03
OptimalRhoParameter["Own"]      = 0.0



# Background Substraction
trainingPercentage = {}
trainingPercentage["Highway"] = 0.5
trainingPercentage["Traffic"] = 0.5
trainingPercentage["Own"] = 1

isAreaFiltering = True
isHoleFilling = True
isMorphology = True
isShadowremoval = False

folders = {}
# Axel's paths
# folders["Own"]  = "video.mp4"
# folders["Own"]       = "../../frames/"
folders["Own"]       = "../../own/input/"
folders["OwnTrain"]  = "../../own/background/"
folders["Highway"]   = "../../../datasetDeliver_2/highway/input/"
folders["Traffic"]   = "../../../datasetDeliver_2/traffic/input/"


################# Stab #####################
block_size = 16
area_size = 100
compensation = 'backward'  # or 'forward'
isReferenceImageFixed = False


#  KALMAN FILTER
KalmanFilterThreshold = {}
KalmanFilterThreshold["Own"]  = 100
KalmanFilterThreshold["Highway"]  = 50
KalmanFilterThreshold["Traffic"]  = 80
carCounting = 0


# Speed control
pixels2metersX = 2.413/88
pixels2metersY = 1.782/65

slideX = 0.646341
slideY = 0.667

referenceXPoint = 103

framesSeconds = 30
