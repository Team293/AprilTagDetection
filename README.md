# AprilTagDetection
Detects apriltags, gets an apriltags position in a video or picture, will have pose detection
at some point. It also makes an overlay of the input with the bounding box of the apriltag.

# Installation

If you are on a Debian-based system, you might need to run the following in order to install tkinter and/or to install Python Image Library (PIL):

```
sudo apt-get install python-tk
```
```
sudo apt-get install python3-pil.imagetk
```

When installing on a Raspberry Pi 3B using Raspberry Pi OS, some difficult challenges occured during installation, related to not having necessary packages and proccessor architecture (the rpi3b uses ARM7/8). The following commands were run instead:

```
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev
```
A better guide can be found [here](https://raspberrypi-guide.github.io/programming/install-opencv).

Next, install the necessary requirements with:

```
pip install -r requirements.txt
```

# Camera Calibration

To calibrate your camera, navigate to the `camera calibration` folder and run the `generate_calibration_images.py` file. Use the SPACE key to create 15 images for creating a camera profile. After 15 images are created (or the Q key is pressed) the GUI will close. Use the calibration board provided in `assets` for calibration images.

After the images are created, run the `calibrate_camera.py` file in the `camera calibration` folder. While this is happening, a GUI should appear with lines and dots overlayed on top of input images. If you do not see this, there may be a problem with your calibration images and you should try generating them using the `generate_calibration_images.py` file. If you see the overlays on your images, wait for the interface to close. Afterwards, a `CameraCalibration.npz` file should have been created in the `camera calibration` folder.

If you would like to use a gui or have your calabration images in a seprate folder you can use `calibrate_camera_gui.py`.
# Use

After calibrating your camera, you should be able to run `track_apriltags.py` in the main folder. This will display a camera feed and an overlay of the april tag tracking data.

Soon, the data from april tags should be able to be send to other devices if necessary. This will include position and rotation of tags as well as ID.
