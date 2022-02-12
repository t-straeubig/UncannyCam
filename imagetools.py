import utils
from mediapipe.python.solutions import \
    face_mesh as mpFaceMesh, \
    selfie_segmentation as mpSelfieSeg
import cv2
import time
import numpy as np

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


    def get_denormalized_landmark(self, index, faceId=0):
        return self.landmarks_denormalized[faceId][index]

    def get_denormalized_landmarks(self, indices, faceId=0):
        """Turns a list of landmark-indices e.g. [0, 5, 3] into a list of denormalized coordinates [[x0, y0], [x5, y5], [x3, y3]].
        Useful for lines, triangles and other polygons."""
        return list(map(lambda index: self.get_denormalized_landmark(index, faceId), indices))
    
    def get_denormalized_landmarks_nested(self, nestedIndices, faceId=0):
        """Turns a list of list landmark-indices recursively into denormalized coordinates. Useful for lists of polygons"""
        return list(map(lambda indices: self.get_denormalized_landmarks(indices, faceId), nestedIndices))

    def find_polygon_denormalized(self, indices):
        """Finds a (denormalized) polygon in the (indexed) lines"""
        polygonIndices = utils.find_polygon(indices)
        return self.get_denormalized_landmarks(polygonIndices)

    def filterPolygon(self, outline, withCuda=True):
        """Applies a blur filter to the image inside the (indexed) lines"""
        polygon = self.find_polygon_denormalized(outline)
        if withCuda:
            blurred = self.cudaFilter()
        else:
            blurred = cv2.bilateralFilter(self.image, 20, 50, 50)
        mask = utils.getMask(self.image.shape, polygon)
        self.image = np.where(mask==np.array([255,255,255]), blurred, self.image)

    def segmentationFilter(self, withCuda=True):
        """Applies a bilateral filter to the region returned by the segmentation filter"""
        background = np.zeros(self.image.shape, dtype=np.uint8)
        background[:] = (0,0,0)
        condition = np.stack((self.selfieSeg_results.segmentation_mask,) * 3, axis=-1) > 0.1
        if withCuda:
            blurred = self.cudaFilter()
        else:
            blurred = cv2.bilateralFilter(self.image, 5, 50, 50)
        self.image = np.where(condition, blurred, self.image)


    def drawLandmarks(self):
        """Draws points at the landmarks"""
        for faceLms in self.landmarks_denormalized:
            for landmark in faceLms:
                cv2.circle(self.image, landmark, 0, (255,0,0), 2)

    def drawLines(self, lines):
        """Draws the (denormalized) lines"""
        # for i, j in lines:
        #     cv2.line(self.image, i, j, (0,255,00), 1)
        self.drawPolygons(lines)

    def drawPolygons(self, polygons):
        """Draws the (denormalized) polygons into the image"""
        for polygon in polygons:
            for i in range(1, len(polygon)):
                cv2.line(self.image, polygon[i-1], polygon[i], (0, 0, 255), 1)

    def drawPoints(self, lines):
        """Draws the denormalized points"""
        for i, j in lines:
            cv2.circle(self.image, i, 0, (255,0,0), 2)
            cv2.circle(self.image, j, 0, (255,0,0), 2)

    def imageAsRGBA(self):
        b_channel, g_channel, r_channel = cv2.split(self.image)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)
        return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    def cudaFilter(self):
        # Use GPU Mat to speed up filtering
        cudaImg = cv2.cuda_GpuMat(cv2.CV_8UC4)
        cudaImg.upload(self.imageAsRGBA())
        filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_ERODE, cv2.CV_8UC4, np.eye(3))
        filter.apply(cudaImg, cudaImg)
        cudaImg = cv2.cuda.bilateralFilter(cudaImg, 10, 30, 30)

        result = cudaImg.download()
        return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        

    

