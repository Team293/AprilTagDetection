import cv2
from dt_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Edit these variables for config.
CAMERA_PARAMS = 'camera calibration/CameraCalibration.npz'
WEBCAM = True

VIDEO_SOURCE = 'Testing_apriltag.mp4'
FRAMERATE = 30

OUTPUT_OVERLAY = True
OUTPUT_FILE = 'vision output/test_output.mp4'
CAMERA_WARP_FIX = True

SHOW_GRAPH = False
DEBUG_MODE = False
SHOW_FRAMERATE = True
ERROR_THRESHOLD = 150
TAG_SIZE = 6

# x, y, z, pitch yaw, roll
WORLD_TAG_LOCATIONS = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, -18, 0, 0, 0, 0],
]


# takes input pitch yaw and roll in degrees and converts it to a rotation matrix
def euler_matrix(pitch, yaw, roll):
    # convert from degrees to radians
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # create the rotation matrix
    rotation_matrix = np.array([
        [
            np.cos(yaw) * np.cos(pitch),
            np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
            np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
        ],
        [
            np.sin(yaw) * np.cos(pitch),
            np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
            np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
        ],
        [
            -np.sin(pitch),
            np.cos(pitch) * np.sin(roll),
            np.cos(pitch) * np.cos(roll)
        ]
    ])
    return rotation_matrix


# Load camera parameters
with np.load(CAMERA_PARAMS) as file:
    cameraMatrix, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]

aprilCameraMatrix = [cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2]]

if WEBCAM:
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(VIDEO_SOURCE)

video_fps = capture.get(cv2.CAP_PROP_FPS),
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_FILE, apiPreference=0, fourcc=fourcc, fps=FRAMERATE,
                         frameSize=(int(frame_width), int(frame_height)))

if SHOW_GRAPH:
    fig = plt.figure()
    axes = plt.axes(projection='3d')
    axes.set_title("3D scatterplot", pad=25, size=15)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")

# detector options
detector = Detector(
    families='tag16h5',
    nthreads=5,
    quad_decimate=2.0,
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


def draw_3d_point(frame, point, color):
    # draws a point in 3d space
    point_x, point_y = point_3d(point[0], point[1], point[2])
    cv2.circle(frame, (point_x, point_y), 5, color, -1)


def draw_3d_line(frame, pt_a, pt_b, color):
    a = point_3d(pt_a[0], pt_a[1], pt_a[2])
    b = point_3d(pt_b[0], pt_b[1], pt_b[2])
    cv2.line(frame, a, b, color, 2)


def point_from_camera(frame, point, color, camera_pos, camera_rot):
    # find the point in 3d space
    point = add_from_direction(camera_pos, camera_rot, point)
    # draw the point
    return point


def draw_point_from_camera(frame, point, color, camera_pos, camera_rot):
    # find the point in 3d space
    point = point_from_camera(frame, point, color, camera_pos, camera_rot)
    # draw the point
    draw_3d_point(frame, point, color)


def draw_line_from_camera(frame, pt_a, pt_b, camera_pos, camera_rot, color):
    # find the point in 3d space
    # invert rotation matrix
    camera_rot = np.linalg.inv(camera_rot)
    pt_a = point_from_camera(frame, pt_a, color, camera_pos, camera_rot)
    pt_b = point_from_camera(frame, pt_b, color, camera_pos, camera_rot)
    # draw the point
    print(pt_a, pt_b)
    draw_3d_line(frame, pt_a, pt_b, color)


# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame
    ret, frame = capture.read()
    if ret:
        start_time = time()

        inputImage = frame

        if CAMERA_WARP_FIX:
            height, width = inputImage.shape[:2]
            newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
            inputImage = cv2.undistort(inputImage, cameraMatrix, dist, None, newCameraMatrix)

        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        if DEBUG_MODE:
            print("[INFO] detecting AprilTags...")
        results = detector.detect(image, estimate_tag_pose=True, camera_params=aprilCameraMatrix, tag_size=TAG_SIZE)

        # print(results)
        if DEBUG_MODE:
            print(f"[INFO] {len(results)} total AprilTags detected")
            print(f"[INFO] Looping over {len(results)} apriltags and getting data")

        # loop over the AprilTag detection results
        if len(results) == 0:
            if not SHOW_GRAPH:
                cv2.imshow("Image", inputImage)
            writer.write(inputImage)

        camera_positions = []
        camera_rotations = []

        for r in results:
            if DEBUG_MODE:
                print(r)

            if r.tag_id > 8:
                continue

            if r.decision_margin < ERROR_THRESHOLD:
                continue

            # should be the size of the tag in the camera
            cube_size = TAG_SIZE
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

            if tag_id <= 8:
                world_tag = WORLD_TAG_LOCATIONS[tag_id - 1]
                # create a rotation matrix from roll pitch yaw
                world_tag_rotation_matrix = euler_matrix(world_tag[3], world_tag[4], world_tag[5])
                # get the camera position in the world
                camera_pos = add_from_direction((world_tag[0], world_tag[1], world_tag[2]), world_tag_rotation_matrix, camera_pos)
                # inverse the tag rotation
                camera_rot = np.linalg.inv(tag_rotation)
                # print(f"Camera position: {camera_pos}")
                camera_rotations.append(camera_rot)
                camera_positions.append(camera_pos)

            # display the tag ID of the cube
            cv2.putText(inputImage, f"#{r.tag_id}", (center_x - 15, center_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if DEBUG_MODE:
                print(f"[DATA] Detection rotation matrix:\n{tag_rotation}")
                print(f"[DATA] Detection translation matrix:\n{tag_position}")

            # if SHOW_GRAPH:
            #     axes.scatter(camera_pos[0], camera_pos[1], camera_pos[2])
            #     plt.pause(0.1)

        # get the average of all the camera positions
        if len(camera_positions) > 0:
            camera_pos = np.average(camera_positions, axis=0)
            camera_rot = np.average(camera_rotations, axis=0)
            print(f"Average camera position: {camera_pos}")
            if SHOW_GRAPH:
                axes.scatter(-camera_pos[0], camera_pos[1], -camera_pos[2])
                plt.pause(0.1)

            # draw a dot at the origin
            # draw_3d_point(inputImage, (-camera_pos[0], camera_pos[1], -camera_pos[2]), (0, 0, 255))
            # draw lines for the x, y, z axes
            origin = (0, 0, 0)
            x_axis = (15, 0, 0)
            y_axis = (0, 15, 0)
            z_axis = (0, 0, 15)

            # draw the x-axis
            draw_line_from_camera(inputImage, origin, x_axis, camera_pos, camera_rot, (0, 0, 255))
            # draw the y-axis
            draw_line_from_camera(inputImage, origin, y_axis, camera_pos, camera_rot, (0, 255, 0))
            # draw the z-axis
            draw_line_from_camera(inputImage, origin, z_axis, camera_pos, camera_rot, (255, 0, 0))

        if DEBUG_MODE:
            # show the output image after AprilTag detection
            print("[INFO] displaying image after overlay")

        if SHOW_FRAMERATE:
            end_time = time()
            cv2.putText(inputImage, f"FPS: {1 / (end_time - start_time)}", (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

        if not SHOW_GRAPH:
            cv2.imshow("Image", inputImage)
        writer.write(inputImage)

        # Press Q on keyboard to  exit
        if not SHOW_GRAPH:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Break the loop
    else:
        break


# When everything done, release the video capture object
# save the output video to disk
writer.release()
capture.release()
if SHOW_GRAPH:
    plt.show()
