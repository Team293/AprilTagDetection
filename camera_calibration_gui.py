import os
import tkinter.filedialog as fd
from tkinter import LEFT, RIGHT, END

import customtkinter
import cv2
import numpy as np
from PIL import Image, ImageOps

app = customtkinter.CTk()
app.geometry("1000x700")
app.title("Camera Calibrator")
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")
files = 0
imageScale = customtkinter.StringVar()

global img, gray, currentPicture, pictureDisplay, outputtedPictures, outputMessage
currentPicture = 0
outputtedPictures = []


def get_files():
    global files, outputtedPictures, outputMessage
    try:
        outputMessage.destroy()
    except:
        print("DEBUG: output message not created")
    outputtedPictures = []
    files = app.tk.splitlist(
        fd.askopenfilenames(
            parent=rightFrame, title="Choose config files (around 4-10)"
        )
    )
    processPicturesButton.configure(state="enabled")
    imageSwich.configure(state="enabled")
    change_image()


def get_data():
    global currentPicture, files
    if currentPicture == len(files) - 1:
        currentPicture = 0
    else:
        currentPicture += 1
    image = Image.open(files[currentPicture])
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    # Create a Label Widget to display the text or Image
    if imageScale.get() == "" or imageScale.get() == "Image Scale (def: 3)":
        imageScaleOut = 3
    else:
        imageScaleOut = imageScale.get()
    return imageScaleOut, width, height, image


def change_image():
    global files, outputtedPictures
    imageScaleOut, width, height, image = get_data()

    if len(outputtedPictures) == 0:
        image = customtkinter.CTkImage(
            dark_image=image,
            size=(int(width / int(imageScaleOut)), int(height / int(imageScaleOut))),
        )
        pictureDisplay.configure(image=image)
    else:
        image = customtkinter.CTkImage(
            dark_image=outputtedPictures[currentPicture],
            size=(int(width / int(imageScaleOut)), int(height / int(imageScaleOut))),
        )
        pictureDisplay.configure(image=image)

    imageInfo.configure(text=f"Image: {os.path.basename(files[currentPicture])}")


def process_images():
    # Defining the dimensions of checkerboard
    global img, gray, files, outputtedPictures, outputMessage
    try:
        outputMessage.destroy()
    except:
        print("DEBUG: output message not created")
    outputtedPictures = []

    CHECKERBOARD = (7, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

    for fname in files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            outputtedPictures.append(Image.fromarray(img))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Camera matrix :")
    print(mtx)
    print("\ndist :")
    print(dist)
    print("\nrvecs :")
    print(rvecs)
    print("\ntvecs :")
    print(tvecs)

    np.savez("CameraCalibration", cameraMatrix=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("\nCalibration compleat, outputted to file: CameraParams.npz")

    outputMessage = customtkinter.CTkLabel(
        master=leftFrame,
        text="Calibration completed\noutputted to file:\nCameraParams.npz",
    )
    outputMessage.pack(pady=10, padx=10, anchor="center")

    imageScaleOut, width, height, image = get_data()
    image = customtkinter.CTkImage(
        dark_image=outputtedPictures[currentPicture],
        size=(int(width / int(imageScaleOut)), int(height / int(imageScaleOut))),
    )
    pictureDisplay.configure(image=image)


def clear_entry(event, entry):
    entry.delete(0, END)


leftFrame = customtkinter.CTkFrame(master=app)
leftFrame.pack(pady=20, padx=(20, 10), fill="x", expand=False, side=LEFT, anchor="nw")

rightFrame = customtkinter.CTkFrame(master=app, width=750)
rightFrame.pack(pady=20, padx=(10, 20), fill="y", expand=True, side=RIGHT)
rightFrame.pack_propagate(False)

label = customtkinter.CTkLabel(master=leftFrame, text="Camera Calibration System")
label.pack(pady=10, padx=10)

getFilesButton = customtkinter.CTkButton(
    master=leftFrame, corner_radius=10, command=get_files, text="Get picture files"
)
getFilesButton.pack(pady=10, padx=10, anchor="center")

# Use CTkButton instead of tkinter Button
processPicturesButton = customtkinter.CTkButton(
    master=leftFrame,
    corner_radius=10,
    command=process_images,
    text="Process pictures",
    state="disabled",
)
processPicturesButton.pack(pady=10, padx=10, anchor="center")

imageScaleInput = customtkinter.CTkEntry(master=leftFrame, textvariable=imageScale)
imageScaleInput.pack(pady=10, padx=10, anchor="center")
imageScaleInput.insert("0", "Image Scale (def: 3)")
imageScaleInput.bind("<FocusIn>", lambda event: clear_entry(event, imageScaleInput))

imageSwich = customtkinter.CTkButton(
    master=leftFrame, text="Switch image", state="disabled", command=change_image
)
imageSwich.pack(pady=10, padx=10, anchor="center")

imageInfo = customtkinter.CTkLabel(master=rightFrame, text="Image: ")
imageInfo.pack(anchor="nw", side=LEFT, padx=(5, 0))

image = Image.open("no_image.png")
pictureDisplay = customtkinter.CTkLabel(
    master=rightFrame,
    text="",
    image=customtkinter.CTkImage(
        dark_image=image,
        size=(400, 400),
    ),
)
pictureDisplay.pack(pady=10, padx=10, anchor="center")

app.mainloop()

# NOTE, the sintax for getting the output from the .npz file is the following:
# with np.load('CameraParams.npz') as file:
#     mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]
