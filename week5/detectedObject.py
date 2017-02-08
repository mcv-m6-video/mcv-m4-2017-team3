import numpy as np
import sys
sys.path.append('../')
import configuration as conf
import KalmanFilterClass as kf
import cv2
import math

class detection:

    def checkWidthHeight(self,tl,br):
        try:
            w = br[0] - tl[0]
            assert w >= 0, "Object with id " + str(self.detectionID) + " has impossible height: " + str(w)
        except AssertionError, e:
            raise Exception(e.args)
        try:
            h = br[1] - tl[1]
            assert h >= 0,"Object with id " + str(self.detectionID) + " has impossible height: " + str(h)
        except AssertionError, e:
            raise Exception(e.args)
        self.width = w
        self.height = h


    def __init__(self,detectionID,startFrame,topLeft,bottomRight,indexes):
        self.detectionID = detectionID
        self.startFrame = startFrame
        self.currentFrame = startFrame
        self.frames = [self.currentFrame]
        self.checkWidthHeight(topLeft,bottomRight)
        self.onScreen = True
        self.topLeft = topLeft
        self.bottomRight = bottomRight
        self.centroid = sum(indexes[0])/len(indexes[0]),sum(indexes[1])/len(indexes[1])
        self.centroids = [sum(indexes[0])/len(indexes[0]),sum(indexes[1])/len(indexes[1])]
        self.kalmanFilter = kf.KalmanFilterClass(id, startFrame, self.centroid)

    def setVisibleOnScreen(self, boolValue):
        self.onScreen = boolValue

    def getVisibleOnScreen(self):
        return self.onScreen

    def getCurrentFrame(self):
        return self.currentFrame

    def update(self,currentFrame,topLeft,bottomRight,indexes):
        self.currentFrame = currentFrame
        self.frames.append(currentFrame)
        self.centroid = (sum(indexes[0])/len(indexes[0]),sum(indexes[1])/len(indexes[1]))
        self.centroids.append(self.centroid)

        self.checkWidthHeight(topLeft,bottomRight)
        # The white pixels position
        self.topLeft = topLeft
        self.bottomRight = bottomRight

    def comet(self,image,color,cometTail):
        comet = np.zeros_like(image)

        # for iTail in range(0,cometTail):
        #     if self.currentFrame + iTail in self.frames:
        #         cv2.line(comet, centroids[-1-iTail],centroids[-2-iTail], color, 2)
        #     else:
        #         break

        return comet

    def isInLine(self,line):
        distanceToLine = np.abs(self.indexes[0] * line[0] + self.indexes[1] * line[1] + line[2])
        if min(distanceToLine) < 2:
            return True
        else:
            return False

    def isCentroidInLine(self,line):
        if np.abs(self.centroid[0] * line[0] + self.centroid[1] * line[1] + line[2]) < 2:
            return True
        else:
            return False

    def computeDistance(self, point1, point2):
        distance = pow((point1[0]-point2[0])**2 + (point1[1]-point2[1])**4,0.5)
        return distance

    def getVector(self, a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.
        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx / dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx / dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx / dy))
            else:
                angle = 180.0
        return distance, angle

    def isVectorValid(a):
        distance, angle = a
        threshold_distance = max(10.0, -0.008 * angle ** 2 + 0.4 * angle + 25.0)
        return (distance <= threshold_distance)
