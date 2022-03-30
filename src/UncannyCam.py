from typing import List
import cv2
import pyvirtualcam
import keyboard
from effects import (
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


def cuda_support() -> bool:
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return True
        else:
            return False
    except:
        return False


class UncannyCam:

    REDUCTION_KEY = "l"

    def __init__(self, with_virtual_cam=False) -> None:
        self.img = None
        self.img_raw = None
        self.test_mode = True
        self.cap = cv2.VideoCapture(0)
        self.intensity = 10

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cuda_support():
            print("OpenCV doesn't have Cuda-Support. All effects run on CPU.")

        self.effects: List[Effect] = []
        self.bilateral_filter = FaceFilter()
        self.morphology_filter = FaceFilter(bilateral_filter=False)
        self.eye_freezer = EyeFreezer()
        self.lazy_eye = LazyEye()
        self.face_swap = FaceSwap()
        self.hue_shift = HueShift()
        self.cheeks_filter = CheeksFilter(with_cuda=cuda_support())
        self.face_symmetry = FaceSymmetry()
        self.basic_noise_filter = NoiseFilter()
        self.perlin_noise_filter = NoiseFilter(1)

        # The effects that get reduced when pressing the REDUCTION_KEY
        self.reducable_effects = [self.lazy_eye]

        if with_virtual_cam:
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=60)
            print(f"Using virtual camera: {self.cam.device}")

    def toggle_effect(self, effect):
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

        return self.img.raw

    def close_camera(self):
        self.cap.release()

    def mainloop(self) -> None:
        while self.cap.isOpened():
            self.get_frame()

            if self.test_mode:
                # self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Image", self.img.raw)
                key = cv2.waitKey(20)
                if key == 27:  # exit on ESC
                    break

            else:
                self.cam.send(cv2.cvtColor(self.img.raw, cv2.COLOR_BGR2RGB))
                self.cam.sleep_until_next_frame()

        print("main loop terminated")


if __name__ == "__main__":
    uncannyCam = UncannyCam(with_virtual_cam=True)
    uncannyCam.mainloop()
