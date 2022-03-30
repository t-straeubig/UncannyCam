import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSlider,
    QLabel,
    QSplitter,
)
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from UncannyCam import UncannyCam


class VideoThread(QThread):
    def __init__(self, camera: UncannyCam):
        super().__init__()
        self.camera = camera
        self._run_flag = True

    def run(self):
        while self._run_flag:
            self.show_video()
        self.camera.close_camera()

    def show_video(self):
        self.camera.get_frame()
        self.camera.cam.send(cv2.cvtColor(self.camera.img.raw, cv2.COLOR_BGR2RGB))
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
        self.camera.test_mode = False
        self.central_widget = QWidget()

        self.add_video()
        self.setup_buttons()
        self.setup_sliders()
        self.create_layout()

        self.create_video_thread()

    def create_video_thread(self):
        self.thread = VideoThread(self.camera)
        self.thread.start()

    def close_event(self, event):
        self.thread.stop()
        event.accept()

    def add_video(self):
        pass

    def setup_buttons(self):
        self.buttons = []
        self.setup_default_button("Smoothing Filter", self.camera.bilateral_filter)
        self.setup_default_button("Morphology Filter", self.camera.morphology_filter)
        self.setup_default_button("Eye Freezer", self.camera.eye_freezer)
        self.setup_default_button("Lazy Eye", self.camera.lazy_eye)
        self.setup_default_button("Face Symmetry", self.camera.face_symmetry)
        self.setup_default_button("Face Swap", self.camera.face_swap)
        self.setup_default_button("Skin Color Hue Shift", self.camera.hue_shift)
        self.setup_default_button("Red Cheeks", self.camera.cheeks_filter)
        self.setup_default_button("Basic Noise", self.camera.basic_noise_filter)
        self.setup_default_button("Perlin Noise", self.camera.perlin_noise_filter)
        self.setup_swap_button()

    def setup_default_button(self, text, effect):
        button = QPushButton(text, self.central_widget)
        button.clicked.connect(lambda: self.press_button(button, effect))
        self.buttons.append(button)

    def press_button(self, button, effect):
        self.camera.toggle_effect(effect)
        if effect in self.camera.effects:
            button.setStyleSheet("QPushButton { background-color: green }")
        else:
            button.setStyleSheet("QPushButton { background-color: light gray }")

    def setup_swap_button(self):
        self.swap_button = QPushButton("Capture Swap Image", self.central_widget)
        self.swap_button.clicked.connect(self.camera.face_swap.change_swap_image)

    def setup_sliders(self):
        self.sliders = []
        self.setup_default_slider(
            "Smoothing Filter",
            self.camera.bilateral_filter,
            min_range=1,
            max_range=60,
            default_value=self.camera.bilateral_filter.intensity,
        )
        self.setup_default_slider(
            "Morphology Filter",
            self.camera.morphology_filter,
            min_range=1,
            max_range=5,
            default_value=self.camera.morphology_filter.intensity,
        )
        self.setup_default_slider(
            "Eye Freezer",
            self.camera.eye_freezer,
            min_range=0,
            max_range=2,
            default_value=self.camera.eye_freezer.intensity,
        )
        self.setup_default_slider(
            "Lazy Eye",
            self.camera.lazy_eye,
            max_range=12,
            default_value=self.camera.lazy_eye.intensity,
        )
        self.setup_default_slider(
            "Face Symmetry",
            self.camera.face_symmetry,
            min_range=0,
            max_range=10,
            default_value=self.camera.face_symmetry.intensity,
        )
        self.setup_default_slider(
            "Face Swap",
            self.camera.face_swap,
            max_range=10,
            default_value=self.camera.face_swap.intensity,
        )
        self.setup_default_slider(
            "Skin Hue",
            self.camera.hue_shift,
            max_range=180,
            default_value=self.camera.hue_shift.intensity,
        )
        self.setup_default_slider(
            "Cheeks Hue",
            self.camera.cheeks_filter,
            max_range=180,
            default_value=self.camera.cheeks_filter.intensity,
        )
        self.setup_default_slider(
            "Basic Noise",
            self.camera.basic_noise_filter,
            max_range=20,
            default_value=self.camera.basic_noise_filter.intensity,
        )
        self.setup_default_slider(
            "Perlin Noise",
            self.camera.perlin_noise_filter,
            max_range=20,
            default_value=self.camera.perlin_noise_filter.intensity,
        )

    def setup_default_slider(
        self, text, effect, min_range=0, max_range=1, default_value=0
    ):
        slider_label = QLabel(self)
        slider_label.setText(text)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_range, max_range)
        slider.setValue(default_value)
        slider.valueChanged.connect(effect.set_intensity)
        self.sliders.append((slider_label, slider))

    def create_layout(self):
        self.layout = QVBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal, self.central_widget)
        self.vertical_layout_buttons = QVBoxLayout()
        self.vertical_layout_sliders = QVBoxLayout()
        self.layout.addWidget(self.splitter)
        self.left_widget = QWidget()
        self.right_widget = QWidget()
        self.left_widget.setLayout(self.vertical_layout_buttons)
        self.right_widget.setLayout(self.vertical_layout_sliders)
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        for button in self.buttons:
            self.vertical_layout_buttons.addWidget(button)
        for label, slider in self.sliders:
            self.vertical_layout_sliders.addWidget(label)
            self.vertical_layout_sliders.addWidget(slider)
        self.layout.addWidget(self.swap_button)
        self.setCentralWidget(self.central_widget)


class CameraWindow(StartWindow):
    def add_video(self):
        self.image_label = QLabel(self)
        self.display_width = 640
        self.display_height = 480
        self.image_label.resize(self.display_width, self.display_height)

    def create_layout(self):
        super().create_layout()
        self.splitter.addWidget(self.image_label)

    def create_video_thread(self):
        self.thread = VideoDisplayThread(self.camera)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        scaled_pixmap = qt_img.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

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
