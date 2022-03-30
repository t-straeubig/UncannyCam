# UncannyCam

### Requirements

You need to install following python packages:

- OpenCV
  - `pip install opencv-python`
- pyvirtualcam
  - `pip install pyvirtualcam`
- PyQT5
  - `pip install PyQt5`
  - or `conda install pyqt`
- Mediapipe
  - `pip install mediapipe`
- keyboard
  - `pip install keyboard`
  - or `conda install -c conda-forge keyboard`

We tested this with Python 3.7 and 3.8.

To use the virtual camera you also need to install OBS. You can download it [here](https://obsproject.com/de) for free.

### Usage

Start the program with `python main.py`. This will open the GUI with an integrated camera window. To use the virtual camera start the program with `python main.py --virtual_cam`.
