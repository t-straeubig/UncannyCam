from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSlider,
)
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from UncannyCam import UncannyCam
import numpy as np
import cv2


class VideoThread(QThread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._run_flag = True

    def run(self):
        while self._run_flag:
            self.show_video()
        self.camera.close_camera()

    def show_video(self):
        self.camera.get_frame()
        self.camera.cam.send(cv2.cvtColor(self.camera.img.image, cv2.COLOR_BGR2RGB))
        self.camera.cam.sleep_until_next_frame()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class VideoDisplayThread(VideoThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def show_video(self):
        self.change_pixmap_signal.emit(self.camera.get_frame())


class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera = UncannyCam()
        self.camera.testMode = False
        self.central_widget = QWidget()
        self.disply_width = 640
        self.display_height = 480

        self.add_video()
        self.setupButtons()
        self.setupSliders()
        self.createLayout()
        self.connectButtons()
        self.connectSliders()

        self.video_thread()

    def video_thread(self):
        self.thread = VideoThread(self.camera)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def add_video():
        pass

    def setupButtons(self):
        self.button_filter = QPushButton("Smoothing Filter", self.central_widget)
        self.button_eye = QPushButton("Eye Freezer", self.central_widget)
        self.button_symmetry = QPushButton("Face Symmetry", self.central_widget)
        self.button_swap = QPushButton("Face Swap", self.central_widget)
        self.button_cheek = QPushButton("Red Cheeks", self.central_widget)

    def setupSliders(self):
        self.slider_cheeks_label = QLabel(self)
        self.slider_cheeks_label.setText("Cheeks Hue")
        self.slider_cheeks = QSlider(Qt.Horizontal)
        self.slider_cheeks.setRange(0, 180)

    def connectButtons(self):
        self.button_filter.clicked.connect(
            lambda: self.camera.toggleFilter(self.camera.faceFilter)
        )
        self.button_eye.clicked.connect(
            lambda: self.camera.toggleFilter(self.camera.eyeFreezer)
        )
        self.button_symmetry.clicked.connect(
            lambda: self.camera.toggleFilter(self.camera.faceSymmetry)
        )
        self.button_swap.clicked.connect(
            lambda: self.camera.toggleFilter(self.camera.faceSwap)
        )
        self.button_cheek.clicked.connect(
            lambda: self.camera.toggleFilter(self.camera.cheeksFilter)
        )

    def connectSliders(self):
        self.slider_cheeks.valueChanged.connect(
            self.camera.cheeksFilter.update_hue_difference
        )

    def createLayout(self):
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_filter)
        self.layout.addWidget(self.button_eye)
        self.layout.addWidget(self.button_symmetry)
        self.layout.addWidget(self.button_swap)
        self.layout.addWidget(self.button_cheek)
        self.layout.addWidget(self.slider_cheeks_label)
        self.layout.addWidget(self.slider_cheeks)
        self.setCentralWidget(self.central_widget)


class CameraWindow(StartWindow):
    def add_video(self):
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

    def createLayout(self):
        super().createLayout()
        self.layout.addWidget(self.image_label)

    def video_thread(self):
        self.thread = VideoDisplayThread(self.camera)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, Qt.KeepAspectRatio
        )
        return QPixmap.fromImage(p)


app = QApplication([])
start_window = CameraWindow()
start_window.show()
app.exit(app.exec_())
