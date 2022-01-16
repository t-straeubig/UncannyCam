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

        if len(self.images) < 7:
            return self.img
        
        return triangles.insertTriangles(self.img, self.images.pop(0), landmarks, self.landmarks.pop(0), self.indices)


class FaceSwap(Effect):

    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.swapImg = Image(cv2.imread('image.png'))

    def apply(self) -> np.ndarray:
        return self.swap()

    def swap(self):
        img = Image(self.uncannyCam.img)
        tesselation = tmp.TRIANGULATION

        newFace = np.zeros_like(img.image)
        
        for i in range(0, int(len(tesselation) / 3)):
            triangle_indices = [tesselation[i * 3],
                                tesselation[i * 3 + 1],
                                tesselation[i * 3 + 2]]
            triangleSwap = np.array(self.swapImg.get_denormalized_landmarks(*triangle_indices), np.int32)
            triangle = np.array(img.get_denormalized_landmarks(*triangle_indices), np.int32)
            newFace = triangles.displace(newFace, self.swapImg.image, triangle, triangleSwap)
            
        leaveOutPoints = utils.getPointCoordinates(img.image.shape[0], img.image.shape[1], img.landmarks, mpFaceMesh.FACEMESH_LEFT_EYE)
        return triangles.insertNewFace(img.image, newFace, img.landmarks_denormalized, leaveOutPoints, withSeamlessClone=True)
        

class FaceFilter(Effect):

    def __init__(self, uncannyCam, mode=1) -> None:
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
        

