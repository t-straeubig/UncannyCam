from typing import List
import cv2
import pyvirtualcam
from effects import (
    DebuggingFilter,
    Effect,
    EyeFreezer,
    FaceFilter,
    FaceSwap,
    HueShift,
    CheeksFilter,
    FaceSymmetry,
    LazyEye,
    NoiseFilter,
)

from imagetools import Image


class UncannyCam:
    def __init__(self) -> None:
        self.img = None
        self.testMode = True
        self.cap = cv2.VideoCapture(0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.effects: List[Effect] = []
        self.bilateralFilter = FaceFilter(self)
        self.morphologyFilter = FaceFilter(self, bilateralFilter=False)
        self.eyeFreezer = EyeFreezer(self)
        self.lazyEye = LazyEye(self)
        self.faceSwap = FaceSwap(self)
        self.hueShift = HueShift(self)
        self.cheeksFilter = CheeksFilter(self)
        self.faceSymmetry = FaceSymmetry(self)
        self.basicNoiseFilter = NoiseFilter(self)
        self.perlinNoiseFilter = NoiseFilter(self, 1)

        self.cam = pyvirtualcam.Camera(width=width, height=height, fps=60)
        print(f"Using virtual camera: {self.cam.device}")

    def toggleFilter(self, filter):
        if filter in self.effects:
            self.effects.remove(filter)
            filter.reset()
        else:
            self.effects.append(filter)

    def get_frame(self):
        success, self.img_raw = self.cap.read()

        if not success:
            print("No Image could be captured")

        if not self.img:
            self.img = Image(self.img_raw, selfieseg=True)
        else:
            self.img.change_image(self.img_raw, reprocess=True)

        for effect in self.effects:
            self.img = effect.apply()

        return self.img.image

    def close_camera(self):
        self.cap.release()

    def mainloop(self) -> None:
        while self.cap.isOpened():
            self.get_frame()

            if self.testMode:
                # self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Image", self.img.image)
                key = cv2.waitKey(20)
                if key == 27:  # exit on ESC
                    break

            else:
                self.cam.send(cv2.cvtColor(self.img.image, cv2.COLOR_BGR2RGB))
                self.cam.sleep_until_next_frame()

        print("main loop terminated")


if __name__ == "__main__":
    uncannyCam = UncannyCam()
    uncannyCam.mainloop()
