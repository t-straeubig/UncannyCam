import utils
from mediapipe.python.solutions import \
    face_mesh as mpFaceMesh, \
    selfie_segmentation as mpSelfieSeg
import cv2
import time

faceMesh = mpFaceMesh.FaceMesh(refine_landmarks=True)
selfieSeg = mpSelfieSeg.SelfieSegmentation(model_selection=0)


class Image():
    def __init__(self, image, landmarks=True, selfieseg=False):
        self.image = image
        if landmarks:
            self.faceMesh_results = faceMesh.process(image)
            self.landmarks = self.faceMesh_results.multi_face_landmarks
            if self.landmarks:
                self.landmarks_denormalized = self._get_denormalized_landmarks()
        if selfieseg:
            self.selfieSeg_results = selfieSeg.process(image)

    def _get_denormalized_landmarks(self):
        newFaceLms = []
        for faceLms in self.landmarks:
            newLandmark = []
            for landmark in faceLms.landmark:
                newLandmark.append(self._denormalize(landmark))
            newFaceLms.append(newLandmark)
        return newFaceLms

    def _denormalize(self, landmark):
        height, width, _ = self.image.shape
        x = int(width * landmark.x)
        y = int(height * landmark.y)
        return x, y

    def copy(self):
        copy = Image(self.image, False, False)
        copy.faceMesh_results = self.faceMesh_results
        copy.landmarks = self.landmarks
        if self.landmarks:
            copy.landmarks_denormalized = self.landmarks_denormalized
        copy.selfieSeg_results = self.selfieSeg_results
        return copy


    def get_denormalized_landmarks(self, faceId=0, *indices):
        """Turns a list of landmark-indices e.g. [0, 5, 3] into a list of denormalized coordinates [[x0, y0], [x5, y5], [x3, y3]].
        Useful for lines, triangles and other polygons."""
        return list(map(lambda index: self.landmarks_denormalized[faceId][index], indices))
    
    def get_denormalized_landmarks_nested(self, faceId=0, *nestedIndices):
        """Turns a list of list landmark-indices recursively into denormalized coordinates. Useful for lists of polygons"""
        return list(map(lambda indices: self.get_denormalized_landmarks(faceId, *indices), nestedIndices))