# make it so that every time the space key is pressed, a new image is saved
# and the calibration is performed
import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# open the camera
capture = cv.VideoCapture(0)

# start a loop and show camera feed
amt = 0
while True:
    ret, frame = capture.read()
    cv.imshow('frame', frame)
    # if space is pressed, save the image
    if cv.waitKey(1) & 0xFF == ord(' '):
        cv.imwrite(f'calibration images/camera_calibration_{amt:0>3}.jpg', frame)
        amt += 1
    # if q is pressed, quit the program
    if cv.waitKey(1) & 0xFF == ord('q'):
        break