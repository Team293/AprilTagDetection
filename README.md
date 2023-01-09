# AprilTagDetection
Detects apriltags, gets an apriltags position in a video or picture, will have pose detection
at some point. It also makes an overlay of the input with the bounding box of the apriltag.

# Installation
First, make  sure you have Python installed on your device. Then, install the necessary requirements with:

```
pip install -r requirements.txt
```

Install tkinter with:

```
pip install tkinter
```

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
A better guid can be found [here](https://raspberrypi-guide.github.io/programming/install-opencv).
