import cv2
import apriltag
import numpy as np

with np.load('CameraParams.npz') as file:
    cameraMatrix, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]

cap = cv2.VideoCapture('0001-0080.mp4')

video_fps = cap.get(cv2.CAP_PROP_FPS),
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter('Testing_apriltag.mp4', apiPreference=0, fourcc=fourcc, fps=video_fps[0],
                         frameSize=(int(width), int(height)))

# test
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:

        inputImage = frame
        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        print("[INFO] detecting AprilTags...")
        results = detector.detect(image)
        print(f"[INFO] {len(results)} total AprilTags detected")

        print(f"[INFO] Looping over {len(results)} apriltags and getting data")
        # loop over the AprilTag detection results
        if len(results) == 0:
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
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(inputImage, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(inputImage, tagFamily, (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x_centered = cX - width / 2
            y_centered = -1 * (cY - height / 2)

            cv2.putText(inputImage, f"Center X coord: {x_centered}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(inputImage, f"Center Y coord: {y_centered}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(inputImage, f"Tag ID: {r.tag_id}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.line(inputImage, (int((width / 2) + 30), int((height / 2))), (int((width / 2) - 30), int((height / 2))),
                     (0, 255, 0), 2)
            cv2.line(inputImage, (int((width / 2)), int((height / 2) + 30)), (int((width / 2)), int((height / 2) - 30)),
                     (0, 255, 0), 2)

        # show the output image after AprilTag detection
        print("[INFO] displaying image after overlay")
        cv2.imshow("Image", inputImage)
        writer.write(inputImage)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
writer.release()
cap.release()
