#!/usr/bin/env python3.7

import numpy as np
import sys
import cv2 as cv

frameCount = 0

import cv2

gX = 0
gY = 0
regionSet = False

def mouseCallback(event, x, y, flags, param):
    global regionSet, gX, gY
    if event == cv.EVENT_LBUTTONDBLCLK:
        gX = x
        gY = y
        regionSet = True
        print("Mouse event %d: %d,%d" % (event, x, y))


def main(argv):
    global regionSet, gX, gY

    template = cv.imread("template-intersection.png",cv.IMREAD_REDUCED_COLOR_8)
    cap = cv.VideoCapture(argv[0])
    fgbg = cv.createBackgroundSubtractorKNN()

    cv.namedWindow("original")
    cv.setMouseCallback("original", mouseCallback)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if frame is opened
        if frame is None:
            break

        if regionSet:
            cv.rectangle(frame, (gX-50,gY-50), (gX+50,gY+50), cv.COLORMAP_PINK, 2,  4, 0)

        cv.imshow("original", frame)

        while not regionSet:
            if cv.waitKey(20) & 0xFF == ord('m'):
                height, width, channel = template.shape
                print("Template h: %d, w: %d" %(height, width))
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                templ_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                result = cv.matchTemplate(frame, template, cv.TM_CCOEFF)
                #cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
                minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
                print("%s; %s - %s; %s" % ( minVal, maxVal, minLoc, maxLoc))
                cv.rectangle(frame, maxLoc, (maxLoc[0]-width,maxLoc[1]-height), cv.COLORMAP_PINK, 2,  4, 0)
                cv.imshow("original", frame)

        fgmask = fgbg.apply(frame)

        cv.imshow("region sub", fgmask[gY-50:gY+50, gX-50:gX+50])
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
