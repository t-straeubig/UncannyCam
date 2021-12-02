from abc import ABC
import time
import numpy as np
import utils
from mediapipe.python.solutions import \
    drawing_utils as mpDraw, \
    face_mesh as mpFaceMesh, \
    selfie_segmentation as mpSelfieSeg


class Effect(ABC):

    def __init__(self, uncannyCam) -> None:
        super().__init__()
        self.uncannyCam = uncannyCam

    def apply(self) -> np.ndarray:
        return self.uncannyCam.img


class EyeFreezer(Effect):

    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.last_time = 0
        self.old = None

    def apply(self) -> np.ndarray:
        self.img = self.uncannyCam.img  # for readability
        if (time.time() - self.last_time) > 0.1:
            print("reset Eyefreezer")
            self.last_time = time.time()
            self.old = self.uncannyCam.img
        eye_polygon = utils.getPolygon(self.img.shape[0], self.img.shape[1], self.uncannyCam.faceMesh_results.multi_face_landmarks, mpFaceMesh.FACEMESH_LEFT_EYE)
        mask = utils.getMask(self.img.shape, eye_polygon)
        return utils.displace(self.img, self.old, mask)