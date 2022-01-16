import utils
from mediapipe.python.solutions import face_mesh as mpFaceMesh

class Image():
    def __init__(self, image):
        self.image = image
        self.landmarks = mpFaceMesh.FaceMesh(refine_landmarks=True).process(image).multi_face_landmarks
        self.landmarks_denormalized = utils.getLandmarks(image, self.landmarks)

    def get_denormalized_landmarks(self, *indices):
        return list(map(lambda index: self.landmarks_denormalized[index], indices))