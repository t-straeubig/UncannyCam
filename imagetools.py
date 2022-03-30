import utils
from mediapipe.python.solutions import (
    face_mesh as mpFaceMesh,
    selfie_segmentation as mpSelfieSeg,
)
import cv2
import numpy as np


class Image:
    def __init__(self, raw, detect_landmarks=True, detect_selfieseg=False):
        self.faceMesh = (
            mpFaceMesh.FaceMesh(refine_landmarks=True) if detect_landmarks else None
        )
        self.selfieSeg = (
            mpSelfieSeg.SelfieSegmentation(model_selection=0)
            if detect_selfieseg
            else None
        )
        self.landmarks = None
        self.change_image(raw, reprocess=True)

    def change_image(self, raw, reprocess=False):
        self._raw = raw
        if self.faceMesh and reprocess:
            self.faceMesh_results = self.faceMesh.process(raw)
            self.landmarks = self.faceMesh_results.multi_face_landmarks
            if self.landmarks:
                self.landmarks_denormalized = self._get_denormalized_landmarks()
        if self.selfieSeg and reprocess:
            self.selfieSeg_results = self.selfieSeg.process(raw)

    def _get_denormalized_landmarks(self):
        newFaceLms = []
        for faceLms in self.landmarks:
            newLandmark = []
            for landmark in faceLms.landmark:
                newLandmark.append(self._denormalize(landmark))
            newFaceLms.append(newLandmark)
        return newFaceLms

    def _denormalize(self, landmark):
        height, width, _ = self._raw.shape
        x = int(width * landmark.x)
        y = int(height * landmark.y)
        return x, y

    def copy(self):
        copy = Image(self._raw, False, False)
        copy.faceMesh_results = self.faceMesh_results
        copy.landmarks = self.landmarks
        if self.landmarks:
            copy.landmarks_denormalized = self.landmarks_denormalized
        copy.selfieSeg_results = self.selfieSeg_results
        return copy

    def flipped(self) -> np.ndarray:
        return cv2.flip(self._raw, 1)

    def get_denormalized_landmark(self, index, faceId=0):
        return self.landmarks_denormalized[faceId][index]

    def get_denormalized_landmarks(self, indices, faceId=0):
        """Turns a list of landmark-indices e.g. [0, 5, 3] into a list of denormalized coordinates [[x0, y0], [x5, y5], [x3, y3]].
        Useful for lines, triangles and other polygons."""
        return [self.get_denormalized_landmark(index, faceId) for index in indices]

    def get_denormalized_landmarks_nested(self, nestedIndices, faceId=0):
        """Turns a list of list landmark-indices recursively into denormalized coordinates. Useful for lists of polygons"""
        return [
            self.get_denormalized_landmarks(indices, faceId)
            for indices in nestedIndices
        ]

    def filter_polygon(self, polygon, withCuda=True):
        """Applies a blur filter to the image inside the (indexed) polygon"""
        polygon_denormalized = self.get_denormalized_landmarks(polygon)
        if withCuda:
            blurred = utils.cudaBilateralFilter(self._raw)
        else:
            blurred = cv2.bilateralFilter(self._raw, 20, 50, 50)
        mask = utils.getMask(self._raw.shape, polygon_denormalized)
        self._raw = np.where(mask == np.array([255, 255, 255]), blurred, self._raw)

    def filterTriangle(self, triangleIndices):
        denormalizedTriangle = self.get_denormalized_landmarks(triangleIndices)
        blurred = utils.cudaBilateralFilter(self._raw)
        mask = utils.getMask(self._raw.shape, denormalizedTriangle)
        self._raw = np.where(mask == np.array([255, 255, 255]), blurred, self._raw)

    def segmentationFilter(self, withCuda=True):
        """Applies a bilateral filter to the region returned by the segmentation filter"""
        background = np.zeros(self._raw.shape, dtype=np.uint8)
        background[:] = (0, 0, 0)
        condition = (
            np.stack((self.selfieSeg_results.segmentation_mask,) * 3, axis=-1) > 0.1
        )
        if withCuda:
            blurred = utils.cudaBilateralFilter(self._raw)
        else:
            blurred = cv2.bilateralFilter(self._raw, 5, 50, 50)
        self._raw = np.where(condition, blurred, self._raw)

    def drawLandmarks(self):
        """Draws points at the landmarks"""
        if self.landmarks:
            for faceLms in self.landmarks_denormalized:
                for landmark in faceLms:
                    cv2.circle(self._raw, landmark, 0, (255, 0, 0), 2)

    def drawLines(self, lines):
        """Draws the (denormalized) lines"""
        # for i, j in lines:
        #     cv2.line(self.image, i, j, (0,255,00), 1)
        self.drawPolygons(lines)

    def drawPolygons(self, polygons):
        """Draws the (denormalized) polygons into the image"""
        for polygon in polygons:
            for i in range(1, len(polygon)):
                cv2.line(self._raw, polygon[i - 1], polygon[i], (0, 0, 255), 1)

    def drawPoints(self, lines):
        """Draws the denormalized points"""
        for i, j in lines:
            cv2.circle(self._raw, i, 0, (255, 0, 0), 2)
            cv2.circle(self._raw, j, 0, (255, 0, 0), 2)
