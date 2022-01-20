from abc import ABC
import time
import numpy as np
from numpy.lib.twodim_base import tri
import triangles
import utils
import cv2
from mediapipe.python.solutions import \
    drawing_utils as mpDraw, \
    face_mesh as mpFaceMesh, \
    selfie_segmentation as mpSelfieSeg
import triangulation_media_pipe as tmp
from imagetools import Image

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
        self.eye_triangles = None
        self.eye_points = utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE)

    def apply(self) -> np.ndarray:
        img = self.uncannyCam.img
        if not img.landmarks:
            return img
        
        if self.eye_triangles is None:
            self.eye_triangles = triangles.getTriangleIndices(img.image, img.landmarks, mpFaceMesh.FACEMESH_LEFT_EYE)
        
        self.images.append(img.copy())

        if len(self.images) < 7:
            return img
        swap_img: Image = self.images.pop(0)
        img.image = triangles.insertTriangles(img, swap_img, self.eye_triangles, self.eye_points)
        return img


class FaceSwap(Effect):

    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.swapImg = Image(cv2.imread('image.png'))
        self.triangles = tmp.TRIANGULATION_NESTED
        self.points = utils.distinct_indices(tmp.TRIANGULATION_NESTED)
        self.leaveOutPoints = utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE)

    def apply(self) -> np.ndarray:
        return self.swap()

    def swap(self):
        img = self.uncannyCam.img
        if not img.landmarks:
            return img
        img.image = triangles.insertTriangles(img, self.swapImg, self.triangles, self.points, self.leaveOutPoints, withSeamlessClone=True)
        return img
        

class FaceFilter(Effect):

    def __init__(self, uncannyCam, mode=1) -> None:
        super().__init__(uncannyCam)
        self.mode = mode

    def filterFace(self):
        img = self.uncannyCam.img
        img.image = utils.filterPolygon(img, mpFaceMesh.FACEMESH_FACE_OVAL)
        return img

    def filterPerson(self):
        img = self.uncannyCam.img
        img.image = utils.segmentationFilter(self.uncannyCam.img)
        return img

    def apply(self) -> np.ndarray:
        return {
            0 : self.filterFace,
            1 : self.filterPerson
        }[self.mode]()
        

