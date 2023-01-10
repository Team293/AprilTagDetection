import cv2
from dt_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Edit these variables for config.
camera_params = 'camera calibration/CameraCalibration.npz'
webcam = True

video_source = 'Testing_apriltag.mp4'
framerate = 30

output_overlay = True
output_file = 'vision output/test_output.mp4'
undistort_frame = True

show_graph = False
debug_mode = False
show_framerate = True
error_threshold = 150
tag_size = 6

# Load camera parameters
with np.load(camera_params) as file:
    cameraMatrix, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]

aprilCameraMatrix = [cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2]]

if webcam:
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(video_source)

video_fps = capture.get(cv2.CAP_PROP_FPS),
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_file, apiPreference=0, fourcc=fourcc, fps=framerate,
                         frameSize=(int(frame_width), int(frame_height)))

if show_graph:
    fig = plt.figure()
    axes = plt.axes(projection='3d')
    axes.set_title("3D scatterplot", pad=25, size=15)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")

# detector options
detector = Detector(
    families='tag16h5',
    nthreads=2,
    quad_decimate=1.0,
    quad_sigma=3.0,
    decode_sharpening=1.0,
    refine_edges=3,
)

# Check if camera opened successfully
if not capture.isOpened():
    print("Error opening video stream or file")


# create a function that draws a dot in 3d space from the camera
def point_3d(x, y, z):
    # converts the 3d point to a point in the camera
    point_x = x * aprilCameraMatrix[0] / z + aprilCameraMatrix[2]
    point_y = y * aprilCameraMatrix[1] / z + aprilCameraMatrix[3]
    return int(point_x), int(point_y)


def draw_3d_point(frame, point, color):
    # draws a point in 3d space
    point_x, point_y = point_3d(point[0], point[1], point[2])
    cv2.circle(frame, (point_x, point_y), 5, color, -1)


def draw_3d_line(frame, pt_a, pt_b, color):
    a = point_3d(pt_a[0], pt_a[1], pt_a[2])
    b = point_3d(pt_b[0], pt_b[1], pt_b[2])
    cv2.line(frame, a, b, color, 2)


def add_from_direction(tagPos, tagRotation, valueToAdd):
    # add a value based on the direction of a tag
    # print length of tag rotations
    # apply the 3x3 rotation matrix to the vector
    new_x = tagRotation[0][0] * valueToAdd[0] + tagRotation[0][1] * valueToAdd[1] + tagRotation[0][2] * valueToAdd[2]
    new_y = tagRotation[1][0] * valueToAdd[0] + tagRotation[1][1] * valueToAdd[1] + tagRotation[1][2] * valueToAdd[2]
    new_z = tagRotation[2][0] * valueToAdd[0] + tagRotation[2][1] * valueToAdd[1] + tagRotation[2][2] * valueToAdd[2]
    new_x += tagPos[0]
    new_y += tagPos[1]
    new_z += tagPos[2]
    return new_x, new_y, new_z


# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame
    ret, frame = capture.read()
    if ret:
        start_time = time()

        inputImage = frame

        if undistort_frame:
            height, width = inputImage.shape[:2]
            newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
            inputImage = cv2.undistort(inputImage, cameraMatrix, dist, None, newCameraMatrix)

        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        if debug_mode:
            print("[INFO] detecting AprilTags...")
        results = detector.detect(image, estimate_tag_pose=True, camera_params=aprilCameraMatrix, tag_size=tag_size)

        # print(results)
        if debug_mode:
            print(f"[INFO] {len(results)} total AprilTags detected")
            print(f"[INFO] Looping over {len(results)} apriltags and getting data")

        # loop over the AprilTag detection results
        if len(results) == 0:
            if not show_graph:
                cv2.imshow("Image", inputImage)
            writer.write(inputImage)

        for r in results:
            if debug_mode:
                print(r)

            if r.tag_id > 8:
                continue

            if r.decision_margin < error_threshold:
                continue

            # should be the size of the tag in the camera
            cube_size = tag_size
            cube_color = (0, 0, 0)
            vertex_color = (120, 100, 0)

            tag_position = (r.pose_t[0][0], r.pose_t[1][0], r.pose_t[2][0])
            tag_rotation = r.pose_R
            tag_id = r.tag_id

            # get the center of the tag
            center_x, center_y = point_3d(tag_position[0], tag_position[1], tag_position[2])

            # find the corners of the tag, based on the direction of the tag
            pointA = add_from_direction(tag_position, tag_rotation, (-cube_size / 2, cube_size / 2, 0))
            pointB = add_from_direction(tag_position, tag_rotation, (cube_size / 2, cube_size / 2, 0))
            pointC = add_from_direction(tag_position, tag_rotation, (cube_size / 2, -cube_size / 2, 0))
            pointD = add_from_direction(tag_position, tag_rotation, (-cube_size / 2, -cube_size / 2, 0))

            # create the top of the cube
            pointE = add_from_direction(tag_position, tag_rotation, (-cube_size / 2, cube_size / 2, -cube_size))
            pointF = add_from_direction(tag_position, tag_rotation, (cube_size / 2, cube_size / 2, -cube_size))
            pointG = add_from_direction(tag_position, tag_rotation, (cube_size / 2, -cube_size / 2, -cube_size))
            pointH = add_from_direction(tag_position, tag_rotation, (-cube_size / 2, -cube_size / 2, -cube_size))

            # draw the line of the cube
            draw_3d_line(inputImage, pointA, pointB, cube_color)
            draw_3d_line(inputImage, pointB, pointC, cube_color)
            draw_3d_line(inputImage, pointC, pointD, cube_color)
            draw_3d_line(inputImage, pointD, pointA, cube_color)

            draw_3d_line(inputImage, pointE, pointF, cube_color)
            draw_3d_line(inputImage, pointF, pointG, cube_color)
            draw_3d_line(inputImage, pointG, pointH, cube_color)
            draw_3d_line(inputImage, pointH, pointE, cube_color)

            draw_3d_line(inputImage, pointA, pointE, cube_color)
            draw_3d_line(inputImage, pointB, pointF, cube_color)
            draw_3d_line(inputImage, pointC, pointG, cube_color)
            draw_3d_line(inputImage, pointD, pointH, cube_color)

            # invert the tag position to get camera position from tag
            camera_pos = (-tag_position[0], -tag_position[1], -tag_position[2])
            print(f"Camera position: {camera_pos}")

            # display the tag ID of the cube
            cv2.putText(inputImage, f"#{r.tag_id}", (center_x - 15, center_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if debug_mode:
                print(f"[DATA] Detection rotation matrix:\n{tag_rotation}")
                print(f"[DATA] Detection translation matrix:\n{tag_position}")

            if show_graph:
                axes.scatter(camera_pos[0], camera_pos[1], camera_pos[2])
                plt.pause(0.1)

        if debug_mode:
            # show the output image after AprilTag detection
            print("[INFO] displaying image after overlay")

        if show_framerate:
            end_time = time()
            cv2.putText(inputImage, f"FPS: {1 / (end_time - start_time)}", (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

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
# save the output video to disk
writer.release()
capture.release()
if show_graph:
    plt.show()
