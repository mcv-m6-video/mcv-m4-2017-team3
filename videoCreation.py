import cv2
import os

def videoCreation(videoName, path):
    if not os.path.exists(path):
        os.makedirs(path)

    vidcap = cv2.VideoCapture(path + videoName)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file
      if cv2.waitKey(10) == 27:                     # exit if Escape is hit
          break
      count += 1

if __name__ == "__main__":
    print 'Video Creation'
    path = './videos/ownVideo/'
    videoName = 'test.wmv'
    videoCreation(videoName, path)