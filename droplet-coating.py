#!/usr/bin/env python3.7

import cv2
import sys


def main():
    video_file = cv2.VideoCapture(sys.argv[1])

    while True:
        ret, frame = video_file.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BayerRG2GRAY)

        cv2.imshow('Video Greyscale', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_file.release()
    cv2.destroyAllWindows()