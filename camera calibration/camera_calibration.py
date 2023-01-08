import numpy as np
import cv2 as cv
import glob

export_file = 'CameraCalibration'
grid_width = 14
grid_height = 9

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_width * grid_height, 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('*.jpg')
print(images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (14, 9), None)
    # If found, add object points, image points (after refining them)
    print(ret, corners)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (14, 9), corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez(f'{export_file}', cameraMatrix=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()
