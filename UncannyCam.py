from typing import List
import cv2
import pyvirtualcam
import keyboard
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

    REDUCTION_KEY = "l"

    def __init__(self) -> None:
        self.img = None
        self.testMode = True
        self.cap = cv2.VideoCapture(0)
        self.intensity = 10

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.effects: List[Effect] = []
        self.bilateralFilter = FaceFilter()
        self.morphologyFilter = FaceFilter(bilateralFilter=False)
        self.eyeFreezer = EyeFreezer()
        self.lazyEye = LazyEye()
        self.faceSwap = FaceSwap()
        self.hueShift = HueShift()
        self.cheeksFilter = CheeksFilter()
        self.faceSymmetry = FaceSymmetry()
        self.basicNoiseFilter = NoiseFilter()
        self.perlinNoiseFilter = NoiseFilter(1)

        # The effects that get reduced when pressing the REDUCTION_KEY
        self.reducable_effects = [self.lazyEye]

        self.cam = pyvirtualcam.Camera(width=width, height=height, fps=60)
        print(f"Using virtual camera: {self.cam.device}")

    def toggleEffect(self, effect):
        if effect in self.effects:
            self.effects.remove(effect)
            effect.reset()
        else:
            self.effects.append(effect)

    def decrease_temporary_intensity(self):
        if self.intensity > 0:
            self.intensity -= 1

    def increase_temporary_intensity(self):
        if self.intensity < 10:
            self.intensity += 1

    def get_temporary_intensity(self):
        return self.intensity / 10

    def apply_effect(self, effect):
        effect.apply(self.img)

    def apply_effect_reduced(self, effect):
        prev_intensity = effect.intensity
        effect.set_intensity(int(effect.intensity * self.get_temporary_intensity()))
        effect.apply(self.img)
        effect.set_intensity(prev_intensity)

    def get_frame(self):
        success, self.img_raw = self.cap.read()

        if not success:
            print("No Image could be captured")

        if not self.img:
            self.img = Image(self.img_raw, detect_selfieseg=True)
        else:
            self.img.change_image(self.img_raw, reprocess=True)

        if keyboard.is_pressed(UncannyCam.REDUCTION_KEY):
            self.decrease_temporary_intensity()
        else:
            self.increase_temporary_intensity()

        for effect in self.effects:
            if effect in self.reducable_effects:
                self.apply_effect_reduced(effect)
            else:
                self.apply_effect(effect)

        return self.img._raw

    def close_camera(self):
        self.cap.release()

    def mainloop(self) -> None:
        while self.cap.isOpened():
            self.get_frame()

            if self.testMode:
                # self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Image", self.img._raw)
                key = cv2.waitKey(20)
                if key == 27:  # exit on ESC
                    break

            else:
                self.cam.send(cv2.cvtColor(self.img._raw, cv2.COLOR_BGR2RGB))
                self.cam.sleep_until_next_frame()

        print("main loop terminated")


if __name__ == "__main__":
    uncannyCam = UncannyCam()
    uncannyCam.mainloop()
