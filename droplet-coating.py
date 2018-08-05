#!/usr/bin/env python3.7

import numpy as np
import sys
import cv2 as cv

frameCount = 0

import cv2

gX = 0
gY = 0
regionSet = False
rect_color = (0, 0, 0)
templ_height = 0
templ_width = 0


def mouseCallback(event, x, y, flags, param):
    global regionSet, gX, gY
    if event == cv.EVENT_LBUTTONDBLCLK:
        gX = x
        gY = y
        regionSet = True
        print("Mouse event %d: %d,%d" % (event, x, y))


def main(argv):
    global regionSet, gX, gY, rect_color, templ_height, templ_width

    template = cv.imread("template-intersection.png",cv.IMWRITE_PNG_STRATEGY_FILTERED)
    cap = cv.VideoCapture(argv[0])
    fgbg = cv.createBackgroundSubtractorKNN()

    cv.namedWindow("original")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if frame is opened
        if frame is None:
            break

        if not regionSet:
            templ_height, templ_width, channel = template.shape
            print("template h: %d, w: %d" % (templ_height, templ_width))
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            templ_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
            result = cv.matchTemplate(frame_gray, templ_gray, cv.TM_CCOEFF)
            minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
            print("Region found: [%s, %s]" % (minLoc, maxLoc))

        cv.circle(frame, maxLoc, 10, cv.COLORMAP_PINK, 1, 4, 0)
        cv.rectangle(frame, maxLoc, (maxLoc[0]+templ_width,maxLoc[1]+templ_height), rect_color, 2,  4, 0)
        cv.imshow("original", frame)

        if regionSet:
            region = frame[maxLoc[1]:maxLoc[1]+templ_height, maxLoc[0]:maxLoc[0]+templ_width]
            sub_region = fgbg.apply(region)
            cv.imshow("bubble region", sub_region)

        key = cv.waitKey(500)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('r'):
            regionSet = True
            rect_color = (255, 0, 0)


    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
