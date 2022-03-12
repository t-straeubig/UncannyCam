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
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QSplitter
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
        self.display_width = 640
        self.display_height = 480

        self.add_video()
        self.setupButtons()
        self.setupSliders()
        self.createLayout()

        self.video_thread()

    def video_thread(self):
        self.thread = VideoThread(self.camera)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def add_video(self):
        pass

    def setupButtons(self):
        self.buttons = []
        self.setupDefaultButton("Smoothing Filter", self.camera.bilateralFilter)
        self.setupDefaultButton("Morphology Filter", self.camera.morphologyFilter)
        self.setupDefaultButton("Eye Freezer", self.camera.eyeFreezer)
        self.setupDefaultButton("Face Symmetry", self.camera.faceSymmetry)
        self.setupDefaultButton("Face Swap", self.camera.faceSwap)
        self.setupDefaultButton("Red Cheeks", self.camera.cheeksFilter)

    def setupDefaultButton(self, text, filter):
        button = QPushButton(text, self.central_widget)
        button.clicked.connect(lambda: self.camera.toggleFilter(filter))
        self.buttons.append(button)

    def setupSliders(self):
        self.sliders = []
        self.setupDefaultSlider(
            "Smoothing Filter",
            self.camera.bilateralFilter,
            min_range=1,
            max_range=60,
            default_value=self.camera.bilateralFilter.slider_value,
        )
        self.setupDefaultSlider(
            "Morphology Filter",
            self.camera.morphologyFilter,
            min_range=1,
            max_range=5,
            default_value=self.camera.morphologyFilter.slider_value,
        )
        self.setupDefaultSlider(
            "Eye Freezer",
            self.camera.eyeFreezer,
            min_range=1,
            max_range=12,
            default_value=self.camera.eyeFreezer.slider_value,
        )
        self.setupDefaultSlider(
            "Face Symmetry",
            self.camera.faceSymmetry,
            min_range=0,
            max_range=10,
            default_value=self.camera.faceSymmetry.slider_value,
        )
        self.setupDefaultSlider(
            "Face Swap",
            self.camera.faceSwap,
            min_range=0,
            max_range=10,
            default_value=self.camera.faceSwap.slider_value,
        )
        self.setupDefaultSlider(
            "Cheeks Hue",
            self.camera.cheeksFilter,
            max_range=180,
            default_value=self.camera.cheeksFilter.slider_value,
        )

    def setupDefaultSlider(
        self, text, filter, min_range=0, max_range=1, default_value=0
    ):
        slider_label = QLabel(self)
        slider_label.setText(text)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_range, max_range)
        slider.setValue(default_value)
        slider.valueChanged.connect(filter.set_slider_value)
        self.sliders.append((slider_label, slider))

    def createLayout(self):
        self.layout = QVBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal, self.central_widget)
        self.verticalLayoutButtons = QVBoxLayout()
        self.verticalLayoutSliders = QVBoxLayout()
        self.layout.addWidget(self.splitter)
        self.leftWidget = QWidget()
        self.rightWidget = QWidget()
        self.leftWidget.setLayout(self.verticalLayoutButtons)
        self.rightWidget.setLayout(self.verticalLayoutSliders)
        self.splitter.addWidget(self.leftWidget)
        self.splitter.addWidget(self.rightWidget)
        self.splitter.setStretchFactor(1,1)
        self.splitter.setStretchFactor(2,1)
        for button in self.buttons:
            self.verticalLayoutButtons.addWidget(button)
        for label, slider in self.sliders:
            self.verticalLayoutSliders.addWidget(label)
            self.verticalLayoutSliders.addWidget(slider)
        self.setCentralWidget(self.central_widget)


class CameraWindow(StartWindow):
    def add_video(self):
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

    def createLayout(self):
        super().createLayout()
        self.splitter.addWidget(self.image_label)
        self.splitter.setStretchFactor(3,1)

    def video_thread(self):
        self.thread = VideoDisplayThread(self.camera)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        myScaledPixmap = qt_img.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(myScaledPixmap)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.display_width, self.display_height, Qt.KeepAspectRatio
        )
        return QPixmap.fromImage(p)


app = QApplication([])
start_window = CameraWindow()
start_window.show()
app.exit(app.exec_())
