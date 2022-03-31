import argparse
from gui import CameraWindow, StartWindow
from PyQt5.QtWidgets import QApplication


def main(virtual_cam):
    app = QApplication([])
    if virtual_cam:
        window = StartWindow()
    else:
        window = CameraWindow()
    window.show()
    app.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument(
        "--virtual_cam", help="Use as virtual camera", action="store_true"
    )
    virtual_cam = parser.parse_args().virtual_cam
    main(virtual_cam)
