
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
from multiprocessing import Pool
import multiprocessing
import sys
sys.path.append('../')
import configuration as conf


def readImagesFromVideo(videoFile,frameID1,frameID2):

    cap = cv2.VideoCapture(videoFile)

    cap.set(1,frameID1)
    ret,frame1 = cap.read()

    cap.set(1,frameID2)
    ret,frame2 = cap.read()

    cap.release()
    return frame1,frame2


def LukasKanadeVideo(videoFile):
    frame1,frame2 = readImagesFromVideo(videoFile,280,281)
    height = frame1.shape[0]
    width = frame1.shape[1]

    cap = cv2.VideoCapture('../movingTrain.mp4')

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 5,blockSize = 7 )
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    mask = np.zeros_like(frame1)
    color = np.random.randint(0,255,(100,3))
    p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)
    cap.set(1,280)
    while(1 and cap.get(1) < 400):

        ret,frame2 = cap.read()
        p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame1,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        frame1 = frame2.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

def LukasKanade(frame1,frame2):
    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 5,blockSize = 7 )
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)

    return p0,p1

def FarnebackVideo(videoFile):

    cap = cv2.VideoCapture(videoFile)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[:,:,1] = 255

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoOutput = cv2.VideoWriter('output.avi',fourcc, 10.0, (frame1.shape[1],frame1.shape[0]),True)


    idx = 0
    flow = None
    ret = True
    while(ret):
        ret, frame2 = cap.read()
        if ret == False:
            break
        nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,nextImg, flow, 0.5, 5, 15, 9, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        #hsv[...,2] = 255
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR_FULL)
        mask = np.zeros_like(frame2)
        for h in range(0,frame2.shape[1],5):
            for w in range(0,frame2.shape[0],5):
                cv2.arrowedLine(mask,(h,w),(h+int(flow[w,h,0]),w+int(flow[w,h,1])),(int(bgr[w,h,0]),int(bgr[w,h,1]),int(bgr[w,h,2])),1)
        outputImage = cv2.add(frame2,mask)
        cv2.arrowedLine(outputImage,(int(outputImage.shape[1]/2),int(outputImage.shape[0]/2)),(int(outputImage.shape[1]/2 + max(0,mag.mean()) * np.cos(ang.mean())),int(outputImage.shape[0]/2 + max(0,mag.mean()) * np.sin(ang.mean()))),(0,0,255),1)
        cv2.namedWindow('Next image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('OF', cv2.WINDOW_NORMAL)
        cv2.putText(outputImage,"Frame:" + str(idx),(50,50),1,1,(255,255,255))
        cv2.imshow('Next image',outputImage)
        cv2.putText(bgr,"Mean diff:" + str(np.abs((prvs-nextImg).mean())),(50,100),1,1,(255,255,255))
        cv2.putText(bgr,"Mean movement:" + str(mag.mean()),(50,50),1,1,(255,255,255))
        cv2.putText(bgr,"Mean orientation:" + str(hsv[...,0].mean()),(50,150),1,1,(255,255,255))
        cv2.imshow('OF',bgr)
        videoOutput.write(outputImage)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        idx = idx + 1
        prvs = nextImg
    cap.release()
    videoOutput.release()
    cv2.destroyAllWindows()

def FarnebackOF(frame1,frame2):
    #All these parameters should be at a configuration file.
    flow = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 5, 15, 9, 7, \
            1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow


# Block Matching stuff:

# The fundamental function of Block Matching
# Given a list of :
# x[0] --> the block to be searched (PxP np.array)
# x[1] --> the area where we have to search it (2P+Nx2P+N np.array)
# x[2] --> the rows and cols of x[1] in the image
# x[3] --> the size of the image
# returns [dx,dy], the flow of that block in pixels. Note that x[1] may not
# coincide with the ranges of x[2], i.e.: if the search area indexes are
# [[-2:6],[-2:6]] this should be a 8x8 array. However, as the image does not
# have negative pixel indexes, the searchArea will only be 6x6. The knowledge
# of these search area indexes and image shape are used to compensate the found
# indexes in the different cases (image borders, corners and interior regions)

def findCorrespondentBlock(x):

    originBlock = x[0]
    searchArea = x[1]
    searchIndexes = x[2]
    yRange = searchIndexes[0]
    xRange = searchIndexes[1]

    imHeight = x[3][0]
    imWidth = x[3][1]


    assert originBlock.shape[0] == originBlock.shape[1]
    # To ease understanding
    N = conf.OFblockSize
    P = conf.OFsearchArea

    # The famous adjustment. For an interior region, the adjustment will be
    # [P,P], and for regions near the x = 0 and y = 0 axis, the adjustment
    # is computed
    adjY = -(P + min(0,yRange[0]))
    adjX = -(P + min(0,xRange[0]))

    # The range of search to avoid "index out of range" stuff. Note that the
    # range is 2P +1 unless we are in a border or corner.
    topY = 1 + 2 * P + min(0,yRange[0]) - max(0,yRange[1] - imHeight)
    topX = 1 + 2 * P + min(0,xRange[0]) - max(0,xRange[1] - imWidth)
    score = 10000.0

    #Block matching happens here:
    for x in range(0,topX):
        for y in range(0,topY):
            temp = compareRegions(originBlock,searchArea[y:y+originBlock.shape[0],x:x+originBlock.shape[1]])
            if temp < score:
                score = temp
                dx = x
                dy = y

    # Coordinates correction with the previous adjustment.
    return [dx + adjX ,dy + adjY]


# For the moment only the good old square difference between regions is computed.
# If we want to add more, this is the spot. (Use a configuration parameter
# please.)
def compareRegions(block1,block2):
    return np.sqrt(sum(sum((block1-block2)**2)))

# Obtain the indexes of the blocks to compare. This function returns the indexes
# corresponding to the division into blocks of PxP and regions of 2P+1x2P+1.
# Note that negative indexes can be (and are) returned.

# DO NOT CHANGE THIS BECAUSE THE INDEXES ARE A HORROR TO DEBUG
def obtainIndexesImage(shape,blockSize= conf.OFblockSize,searchArea= conf.OFsearchArea):
    blockIndexes = []
    searchAreaIndexes = []

    for y in range(shape[0]/blockSize):
        for x in range(shape[1]/blockSize):
            yBlockRange = [blockSize*y,blockSize*(y+1)]
            xBlockRange = [blockSize*x,blockSize*(x+1)]
            blockIndexes.append([yBlockRange,xBlockRange])

            xSearchRange = [blockSize*x-searchArea,blockSize*(x+1) + searchArea]
            ySearchRange = [blockSize*y-searchArea,blockSize*(y+1) + searchArea]
            searchAreaIndexes.append([ySearchRange,xSearchRange])
    return blockIndexes,searchAreaIndexes

# Computes the optical flow between images VERY fast using the multiprocessing
# capabilities of Python. IT divides the frame1 into blocks and frame2 into
# searchAreas that are processed in parallel. There is commented code that
# corresponds to the same code, but computed sequentally, just in case it does
# not work on your Windows machines.
def opticalFlowBW(frame1,frame2):
    width = frame1.shape[1]
    height = frame1.shape[0]

    newWidth = width/conf.OFblockSize
    newHeight = height/conf.OFblockSize

    p = Pool(multiprocessing.cpu_count())
    blockIndexes,searchAreaIndexes = obtainIndexesImage(frame2.shape)
    xSearchIndexes = [el[0] for el in searchAreaIndexes]
    ySearchIndexes = [el[1] for el in searchAreaIndexes]

    x = [[frame1[b[0][0]:b[0][1],b[1][0]:b[1][1]], \
          frame2[max(0,sX[0]):min(frame2.shape[0],sX[1]),max(0,sY[0]):min(frame2.shape[1],sY[1])], \
          c, \
          frame2.shape] \
          for b,sY,sX,c in zip(blockIndexes,ySearchIndexes,xSearchIndexes,searchAreaIndexes)]
    OF = p.map(findCorrespondentBlock,x)

    # Non-multiprocesing solution:
    '''
    OF = []
    for el in x:
        OF.append(findCorrespondentBlock(x))
    '''
    OFx = np.reshape(np.asarray([el[0] for el in OF]),(newHeight,newWidth))
    OFy = np.reshape(np.asarray([el[1] for el in OF]),(newHeight,newWidth))

    return np.dstack((OFx,OFy))


if __name__ == "__main__":
    print 'Optical Flow method'
