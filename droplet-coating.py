import numpy as np
import cv2

cap = cv2.VideoCapture("examples/video_1527101251.mp4")
fgbg = cv2.createBackgroundSubtractorKNN()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()