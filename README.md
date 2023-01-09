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

When installing on a Raspberry Pi 3B using Raspberry Pi OS, some difficult challenges occured during installation, related to not having necessary packages. To solve this, the following commands were run:

```
pip install --upgrade pip setuptools wheel
```
```
pip install python-opencv
```
