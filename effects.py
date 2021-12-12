from abc import ABC
import time
import numpy as np
from numpy.lib.twodim_base import tri
import triangles
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
        self.images = []
        self.landmarks = []
        self.indices = None

    def apply(self) -> np.ndarray:
        self.img = self.uncannyCam.img  # for readability
        landmarks = self.uncannyCam.faceMesh_results.multi_face_landmarks
        if self.indices is None:
            self.indices = triangles.getTriangleIndices(self.img, landmarks, mpFaceMesh.FACEMESH_LEFT_EYE)
        
        if not landmarks:
            return self.img
        self.images.append(self.img)
        self.landmarks.append(landmarks)

        if len(self.images) < 5:
            return self.img
        
        return triangles.displace(self.img, self.images.pop(0), landmarks, self.landmarks.pop(0), self.indices)

class FaceFilter(Effect):

    def __init__(self, uncannyCam, mode) -> None:
        super().__init__(uncannyCam)
        self.mode = mode

    def filterFace(self):
        landmarks = self.uncannyCam.faceMesh_results.multi_face_landmarks
        facemeshOval = mpFaceMesh.FACEMESH_FACE_OVAL
        return utils.filterPolygon(self.uncannyCam.img, landmarks, facemeshOval)

    def filterPerson(self):
        mask = self.uncannyCam.selfieSeg_results.segmentation_mask
        return utils.segmentationFilter(self.uncannyCam.img, mask)

    def apply(self) -> np.ndarray:
        return {
            0 : self.filterFace,
            1 : self.filterPerson
        }[self.mode]()
        

