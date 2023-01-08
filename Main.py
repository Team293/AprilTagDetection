import cv2
from dt_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt

# Edit these variables for config.
camera_params = 'CameraParamsBlender.npz'
webcam = False
video_source = 'Testing_apriltag.mp4'
output_file = 'apriltag_output.mp4'
show_graph = False
undistort_frame = False

# Load camera parameters
with np.load(camera_params) as file:
    cameraMatrix, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]

aprilCameraMatrix = [cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2]]

capture = cv2.VideoCapture(video_source)

video_fps = capture.get(cv2.CAP_PROP_FPS),
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_file, apiPreference=0, fourcc=fourcc, fps=video_fps[0],
                         frameSize=(int(width), int(height)))

if show_graph:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("3D scatterplot", pad=25, size=15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

# options = DetectorOptions(families="tag36h11")
detector = Detector(families='tag36h11')

# Check if camera opened successfully
if not capture.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame
    ret, frame = capture.read()
    if ret:

        inputImage = frame

        h, w = inputImage.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

        if undistort_frame:
            inputImage = cv2.undistort(inputImage, cameraMatrix, dist, None, newCameraMatrix)
        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        print("[INFO] detecting AprilTags...")
        results = detector.detect(image, estimate_tag_pose=True, camera_params=aprilCameraMatrix, tag_size=0.1651)
        # print(results)
        print(f"[INFO] {len(results)} total AprilTags detected")
        print(f"[INFO] Looping over {len(results)} apriltags and getting data")
        # loop over the AprilTag detection results
        if len(results) == 0:
            if not show_graph:
                cv2.imshow("Image", inputImage)
            writer.write(inputImage)

        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(inputImage, ptA, ptB, (0, 255, 0), 2)
            cv2.line(inputImage, ptB, ptC, (0, 255, 0), 2)
            cv2.line(inputImage, ptC, ptD, (0, 255, 0), 2)
            cv2.line(inputImage, ptD, ptA, (0, 255, 0), 2)

            cv2.circle(inputImage, ptA, 4, (0, 0, 255), -1)
            cv2.circle(inputImage, ptB, 4, (0, 0, 255), -1)
            cv2.circle(inputImage, ptC, 4, (0, 0, 255), -1)
            cv2.circle(inputImage, ptD, 4, (0, 0, 255), -1)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(inputImage, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(inputImage, tagFamily, (ptD[0], ptD[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x_centered = cX - width / 2
            y_centered = -1 * (cY - height / 2)

            cv2.putText(inputImage, f"Center X coord: {x_centered}", (ptB[0] + 10, ptB[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

            cv2.putText(inputImage, f"Center Y coord: {y_centered}", (ptB[0] + 10, ptB[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

            cv2.putText(inputImage, f"Tag ID: {r.tag_id}", (ptC[0] - 70, ptC[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

            cv2.circle(inputImage, (int((width / 2)), int((height / 2))), 5, (0, 0, 255), 2)

            # pose = detector.detection_pose(detection=r, camera_params=aprilCameraMatrix, tag_size=8)
            poseRotation = r.pose_R
            poseTranslation = r.pose_t
            print(f"[DATA] Detection rotation matrix:\n{poseRotation}")
            print(f"[DATA] Detection translation matrix:\n{poseTranslation}")
            # print(f"[DATA] Apriltag position:\n{}")
            if show_graph:
                ax.scatter(poseTranslation[0][0], poseTranslation[1][0], poseTranslation[2][0])
                plt.pause(0.01)

        # show the output image after AprilTag detection
        print("[INFO] displaying image after overlay")
        if not show_graph:
            cv2.imshow("Image", inputImage)
        writer.write(inputImage)

        # Press Q on keyboard to  exit
        if not show_graph:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
writer.release()
capture.release()
if show_graph:
    plt.show()