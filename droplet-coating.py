import numpy as np
import sys
import cv2 as cv

frameCount = 0

import cv2
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

        
def main(argv):
    
    cap = cv.VideoCapture("examples/video_1527101251.mp4")
    fgbg = cv.createBackgroundSubtractorKNN()
       
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if frame is opened
        if frame is None:
            print ('Error opening video:')
            return -1

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
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
