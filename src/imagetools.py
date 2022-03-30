import cv2
import numpy as np
from mediapipe.python.solutions import (
    face_mesh as mpFaceMesh,
    selfie_segmentation as mpSelfieSeg,
)
import utils


class Image:
    def __init__(self, raw, detect_landmarks=True, detect_selfieseg=False):
        self.face_mesh = (
            mpFaceMesh.FaceMesh(refine_landmarks=True) if detect_landmarks else None
        )
        self.selfie_seg = (
            mpSelfieSeg.SelfieSegmentation(model_selection=0)
            if detect_selfieseg
            else None
        )
        self.landmarks = None
        self.change_image(raw, reprocess=True)

    def change_image(self, raw, reprocess=False):
        self.raw = raw
        if self.face_mesh and reprocess:
            self.face_mesh_results = self.face_mesh.process(raw)
            self.landmarks = self.face_mesh_results.multi_face_landmarks
            if self.landmarks:
                self.landmarks_denormalized = self._get_denormalized_landmarks()
        if self.selfie_seg and reprocess:
            self.selfie_seg_results = self.selfie_seg.process(raw)

    def _get_denormalized_landmarks(self):
        new_face_landmarks = []
        for face_landmarks in self.landmarks:
            new_landmark = []
            for landmark in face_landmarks.landmark:
                new_landmark.append(self._denormalize(landmark))
            new_face_landmarks.append(new_landmark)
        return new_face_landmarks

    def _denormalize(self, landmark):
        height, width, _ = self.raw.shape
        x = int(width * landmark.x)
        y = int(height * landmark.y)
        return x, y

    def copy(self):
        copy = Image(self.raw, False, False)
        copy.face_mesh_results = self.face_mesh_results
        copy.landmarks = self.landmarks
        if self.landmarks:
            copy.landmarks_denormalized = self.landmarks_denormalized
        copy.selfie_seg_results = self.selfie_seg_results
        return copy

    def flipped(self) -> np.ndarray:
        return cv2.flip(self.raw, 1)

    def get_denormalized_landmark(self, index, face_id=0):
        return self.landmarks_denormalized[face_id][index]

    def get_denormalized_landmarks(self, indices, face_id=0):
        """Turns a list of landmark-indices e.g. [0, 5, 3] into a list of denormalized coordinates [[x0, y0], [x5, y5], [x3, y3]].
        Useful for lines, triangles and other polygons."""
        return [self.get_denormalized_landmark(index, face_id) for index in indices]

    def get_denormalized_landmarks_nested(self, nested_indices, faceId=0):
        """Turns a list of list landmark-indices recursively into denormalized coordinates. Useful for lists of polygons"""
        return [
            self.get_denormalized_landmarks(indices, faceId)
            for indices in nested_indices
        ]

    def filter_polygon(self, polygon, with_cuda=True):
        """Applies a blur filter to the image inside the (indexed) polygon"""
        polygon_denormalized = self.get_denormalized_landmarks(polygon)
        if with_cuda:
            blurred = utils.cuda_bilateral_filter(self.raw)
        else:
            blurred = cv2.bilateralFilter(self.raw, 20, 50, 50)
        mask = utils.get_mask(self.raw.shape, polygon_denormalized)
        self.raw = np.where(mask == np.array([255, 255, 255]), blurred, self.raw)

    def filter_triangle(self, triangle_indices):
        denormalized_triangle = self.get_denormalized_landmarks(triangle_indices)
        blurred = utils.cuda_bilateral_filter(self.raw)
        mask = utils.get_mask(self.raw.shape, denormalized_triangle)
        self.raw = np.where(mask == np.array([255, 255, 255]), blurred, self.raw)

    def segmentation_filter(self, with_cuda=True):
        """Applies a bilateral filter to the region returned by the segmentation filter"""
        background = np.zeros(self.raw.shape, dtype=np.uint8)
        background[:] = (0, 0, 0)
        condition = (
            np.stack((self.selfie_seg_results.segmentation_mask,) * 3, axis=-1) > 0.1
        )
        if with_cuda:
            blurred = utils.cuda_bilateral_filter(self.raw)
        else:
            blurred = cv2.bilateralFilter(self.raw, 5, 50, 50)
        self.raw = np.where(condition, blurred, self.raw)

    def draw_landmarks(self):
        """Draws points at the landmarks"""
        if self.landmarks:
            for face_landmarks in self.landmarks_denormalized:
                for landmark in face_landmarks:
                    cv2.circle(self.raw, landmark, 0, (255, 0, 0), 2)

    def draw_lines(self, lines):
        """Draws the (denormalized) lines"""
        self.draw_polygons(lines)

    def draw_polygons(self, polygons):
        """Draws the (denormalized) polygons into the image"""
        for polygon in polygons:
            for i in range(1, len(polygon)):
                cv2.line(self.raw, polygon[i - 1], polygon[i], (0, 0, 255), 1)

    def draw_points(self, lines):
        """Draws the denormalized points"""
        for i, j in lines:
            cv2.circle(self.raw, i, 0, (255, 0, 0), 2)
            cv2.circle(self.raw, j, 0, (255, 0, 0), 2)
