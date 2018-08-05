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

    cap = cv.VideoCapture(argv[0])
    fgbg = cv.createBackgroundSubtractorKNN()

    cv.namedWindow("original")
    cv.setMouseCallback("original", mouseCallback, param=None)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if frame is opened
        if frame is None:
            break

        if regionSet:
            cv.rectangle(frame, (gX-50,gY-50), (gX+50,gY+50), cv.COLORMAP_PINK, 1,  4, 0)

        cv.imshow("original", frame)

        while not regionSet:
            cv.waitKey(20)


        # Transform source image to gray if it is not already
        if len(frame.shape) != 2:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame
        # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
        gray = cv.bitwise_not(gray)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                    cv.THRESH_BINARY, 15, -2)

        # Create the images that will use to extract the horizontal and vertical lines
        horizontal = np.copy(bw)
        vertical = np.copy(bw)
            # Specify size on horizontal axis
        cols = horizontal.shape[1]

            # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = int(rows / 30)
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)
        rgb = cv.cvtColor(vertical, cv.COLOR_GRAY2BGR)

        rgb[np.where((rgb==[255,255,255]).all(axis=2))] = [0,255,255]
        fgmask = fgbg.apply(frame)
        cgmask = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)
        # Show extracted vertical line
        comb = cgmask + rgb
        cv.imshow("both", comb)
        cv.imshow("cut both", comb[gY-50:gY+50, gX-50:gX+50])
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
