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
        """Changes the raw image to the new image. Using this method is preferred over creating new Images,
        because the mediapipe FaceMesh and SelfieSegmentation instances maintain context about the last images they processed.
        If the reprocess flag is set to False the face mesh results don't get updated."""
        self.raw = raw
        if self.face_mesh and reprocess:
            self.face_mesh_results = self.face_mesh.process(raw)
            self.landmarks = self.face_mesh_results.multi_face_landmarks
            if self.landmarks:
                self.landmarks_denormalized = self._get_denormalized_landmarks()
        if self.selfie_seg and reprocess:
            self.selfie_seg_results = self.selfie_seg.process(raw)

    def _get_denormalized_landmarks(self):
        """Returns the landmarks of the picture in pixel space."""
        new_face_landmarks = []
        for face_landmarks in self.landmarks:
            new_landmark = []
            for landmark in face_landmarks.landmark:
                new_landmark.append(self._denormalize(landmark))
            new_face_landmarks.append(new_landmark)
        return new_face_landmarks

    def _denormalize(self, landmark):
        """Returns the given landmark in the pixel space."""
        height, width, _ = self.raw.shape
        x = int(width * landmark.x)
        y = int(height * landmark.y)
        return x, y

    def copy(self):
        """Returns a static copy of the Image. It doesn't instantiate new FaceMesh and SelfieSegmentation objects reducing but keeps their results from this image."""
        copy = Image(self.raw, False, False)
        copy.face_mesh_results = self.face_mesh_results
        copy.landmarks = self.landmarks
        if self.landmarks:
            copy.landmarks_denormalized = self.landmarks_denormalized
        copy.selfie_seg_results = self.selfie_seg_results
        return copy

    def flipped(self) -> np.ndarray:
        """Returns a flipped copy of the image in np-array form."""
        return cv2.flip(self.raw, 1)

    def get_denormalized_landmark(self, index, face_id=0):
        """Turns a landmark-index into denormalized coordinates."""
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

    def blurred(self, method: str, intensity, with_cuda=True) -> np.ndarray:
        """Returns a blurred copy of the image. Method must be either "bilateral" or "morphology"."""
        assert method in ["bilateral", "morphology"]
        if method == "bilateral":
            if with_cuda:
                return utils.cuda_bilateral_filter(self.raw, intensity)
            else:
                return cv2.bilateralFilter(self.raw, intensity, 50, 50)
        if method == "morphology":
            if with_cuda:
                return utils.cuda_morphology_filter(self.raw, intensity)
            else:
                kernel = np.ones((intensity,intensity),np.uint8)
                return cv2.morphologyEx(self.raw, cv2.MORPH_ERODE, kernel, self.raw)

    def get_mask(self, polygon):
        """Get a mask for the (indexed) polygon in the shape of the image."""
        polygon_denormalized = self.get_denormalized_landmarks(polygon)
        return utils.get_mask(self.raw.shape, polygon_denormalized)

    def draw_landmarks(self):
        """Draws points at the landmarks"""
        if self.landmarks:
            for face_landmarks in self.landmarks_denormalized:
                for landmark in face_landmarks:
                    cv2.circle(self.raw, landmark, 0, (255, 0, 0), 2)

    def draw_lines(self, lines):
        """Draws the (denormalized) lines into the image"""
        self.draw_polygons(lines)

    def draw_polygons(self, polygons):
        """Draws the (denormalized) polygons into the image"""
        for polygon in polygons:
            for i in range(1, len(polygon)):
                cv2.line(self.raw, polygon[i - 1], polygon[i], (0, 0, 255), 1)
