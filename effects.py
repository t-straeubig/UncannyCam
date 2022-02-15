from abc import ABC
import time
import numpy as np
import triangles
import utils
import cv2
import keyboard
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
            self.eye_triangles = triangles.getTriangleIndices(img, mpFaceMesh.FACEMESH_LEFT_EYE)
        
        self.images.append(img.copy())

        if len(self.images) < 7:
            return img
        swap_img: Image = self.images.pop(0)
        img.image = triangles.insertTriangles(img, swap_img, self.eye_triangles, self.eye_points)
        return img


class FaceSwap(Effect):

    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.swapImg = None
        self.triangles = tmp.TRIANGULATION_NESTED
        self.points = utils.distinct_indices(tmp.TRIANGULATION_NESTED)
        self.leaveOutPoints = utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE)

    def apply(self) -> np.ndarray:
        if keyboard.is_pressed('s'):
            self.swapImg = Image(self.uncannyCam.img_raw)
            if not self.swapImg.landmarks:
                self.swapImg = None
        return self.swap()

    def swap(self):
        img = self.uncannyCam.img
        if not img.landmarks or not self.swapImg:
            return img
        img.image = triangles.insertTriangles(img, self.swapImg, self.triangles, self.points, self.leaveOutPoints, withSeamlessClone=True)
        return img


class FaceSymmetry(Effect):

    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.triangles = tmp.TRIANGULATION_NESTED
        self.points = utils.distinct_indices(tmp.TRIANGULATION_NESTED)
        self.flipped = None

    def apply(self) -> np.ndarray:
        img = self.uncannyCam.img
        if not img.landmarks:
            return img
        if not self.flipped:
            self.flipped = Image(img.flipped())
        else:
            self.flipped.change_image(img.flipped(), reprocess=True)
        if not self.flipped.landmarks:
            return img
        img.image = triangles.insertTriangles(img, self.flipped, self.triangles, self.points, withSeamlessClone=True)
        return img


class FaceFilter(Effect):

    def __init__(self, uncannyCam, mode=3) -> None:
        super().__init__(uncannyCam)
        self.mode = mode

    def filter_face(self):
        polygon = utils.find_polygon(mpFaceMesh.FACEMESH_FACE_OVAL)
        self.uncannyCam.img.filter_polygon(polygon)
        return self.uncannyCam.img

    def filter_person(self):
        self.uncannyCam.img.filter_segmentation()
        return self.uncannyCam.img

    def filter_triangle(self):
        indices = utils.distinct_indices(mpFaceMesh.FACEMESH_TESSELATION)
        triangle = [indices[50], indices[260], indices[150]]
        self.uncannyCam.img.filter_polygon(triangle)
        # if self.uncannyCam.img.landmarks:
        #     self.uncannyCam.img.drawPolygons([self.uncannyCam.img.get_denormalized_landmarks(triangle)])
        return self.uncannyCam.img

    def filter_image(self):
        self.uncannyCam.img.image = self.uncannyCam.img.cudaFilter()
        return self.uncannyCam.img

    def apply(self) -> np.ndarray:
        return {
            0 : self.filter_face,
            1 : self.filter_person,
            2 : self.filter_triangle,
            3 : self.filter_image
        }[self.mode]()
        

class HueShift(Effect):

    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.lower = np.array([0, 48, 80], dtype = "uint8")
        self.upper = np.array([20, 255, 255], dtype = "uint8")

    def skinMask(self, raw_hsv):
        """https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/"""
        mask = cv2.inRange(raw_hsv, self.lower, self.upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations = 2)
        mask = cv2.dilate(mask, kernel, iterations = 2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        return mask

    def apply(self) -> np.ndarray:
        raw = self.uncannyCam.img.image
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        shifted = np.copy(raw)
        shifted[:,:,0] = shifted[:,:,0] + 60
        cv2.imshow("mask", self.skinMask(raw))
        new_raw = np.where(self.skinMask(raw), shifted, raw)
        new_raw = cv2.cvtColor(new_raw, cv2.COLOR_HSV2BGR)
        self.uncannyCam.img.image = new_raw
        return self.uncannyCam.img
