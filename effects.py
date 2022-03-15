from abc import ABC
from cv2 import bilateralFilter
import numpy as np
import triangles
import utils
import cv2
import keyboard
from mediapipe.python.solutions import (
    drawing_utils as mpDraw,
    face_mesh as mpFaceMesh,
    selfie_segmentation as mpSelfieSeg,
)
import triangulation_media_pipe as tmp
from imagetools import Image
from noise import generate_perlin_noise_2d


class Effect(ABC):
    def __init__(self, uncannyCam) -> None:
        super().__init__()
        self.uncannyCam = uncannyCam
        self.slider_value = 0

    def apply(self) -> np.ndarray:
        return self.uncannyCam.img

    def set_slider_value(self, value):
        self.slider_value = value

    def alpha_blend_value(self):
        return self.slider_value / 10

    def alpha_blend(self, new_img: Image, old_image: Image):
        new_img.image = cv2.addWeighted(
            old_image.image,
            1 - self.alpha_blend_value(),
            new_img.image,
            self.alpha_blend_value(),
            0,
        )
        return new_img


class EyeFreezer(Effect):
    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.images = []
        self.landmarks = []
        self.eye_triangles = None
        self.eye_points = utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE)
        self.slider_value = 5

    def apply(self) -> np.ndarray:
        img = self.uncannyCam.img
        if not img.landmarks:
            return img

        if self.eye_triangles is None:
            self.eye_triangles = triangles.getTriangleIndices(
                img, mpFaceMesh.FACEMESH_LEFT_EYE
            )

        self.images.append(img.copy())

        if len(self.images) <= self.slider_value or self.slider_value == 1:
            return img
        swap_img: Image = self.images.pop(0)
        img.image = triangles.insertTriangles(
            img, swap_img, self.eye_triangles, self.eye_points
        )
        return img

    def set_slider_value(self, value):
        super().set_slider_value(value)
        self.images = []


class FaceSwap(Effect):
    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.swapImg = None
        self.slider_value = 10
        self.triangles = tmp.TRIANGULATION_NESTED
        self.points = utils.distinct_indices(tmp.TRIANGULATION_NESTED)
        self.leaveOutPoints = utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE)

    def apply(self) -> np.ndarray:
        old_image = self.uncannyCam.img.copy()
        if keyboard.is_pressed("s"):
            self.change_swap_image()
        return self.alpha_blend(self.swap(), old_image)

    def swap(self):
        img = self.uncannyCam.img
        if not img.landmarks or not self.swapImg:
            return img
        img.image = triangles.insertTriangles(
            img,
            self.swapImg,
            self.triangles,
            self.points,
            self.leaveOutPoints,
            withSeamlessClone=True,
        )
        return img

    def change_swap_image(self):
        self.swapImg = Image(self.uncannyCam.img.image)
        if not self.swapImg.landmarks:
            self.swapImg = None


class FaceSymmetry(Effect):
    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.triangles = tmp.TRIANGULATION_NESTED
        self.points = utils.distinct_indices(tmp.TRIANGULATION_NESTED)
        self.flipped = None
        self.slider_value = 10

    def apply(self) -> np.ndarray:
        old_image = self.uncannyCam.img.copy()
        img = self.uncannyCam.img
        if not img.landmarks:
            return img
        if not self.flipped:
            self.flipped = Image(img.flipped())
        else:
            self.flipped.change_image(img.flipped(), reprocess=True)
        if not self.flipped.landmarks:
            return img
        img.image = triangles.insertTriangles(
            img, self.flipped, self.triangles, self.points, withSeamlessClone=True
        )
        return self.alpha_blend(img, old_image)


class FaceFilter(Effect):
    def __init__(self, uncannyCam, mode=3, bilateralFilter=True) -> None:
        super().__init__(uncannyCam)
        self.mode = mode
        if bilateralFilter:
            self.slider_value = 30
        else:
            self.slider_value = 3
        self.bilateralFilter = bilateralFilter

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
        return self.uncannyCam.img

    def filter_image(self):
        if self.bilateralFilter:
            self.uncannyCam.img.image = utils.cudaBilateralFilter(
                self.uncannyCam.img.image, self.slider_value
            )
        else:
            self.uncannyCam.img.image = utils.cudaMorphologyFilter(
                self.uncannyCam.img.image, self.slider_value
            )
        return self.uncannyCam.img

    def apply(self) -> np.ndarray:
        return {
            0: self.filter_face,
            1: self.filter_person,
            2: self.filter_triangle,
            3: self.filter_image,
        }[self.mode]()


class NoiseFilter(Effect):
    def __init__(self, uncannyCam, mode=0) -> None:
        super().__init__(uncannyCam)
        self.slider_value = 0
        self.mode = mode

    def perlin_noise(self):
        old_image = self.uncannyCam.img.copy()
        img = self.uncannyCam.img
        perlin_noise = generate_perlin_noise_2d((img.image.shape[0], img.image.shape[1]), (1,2))
        perlin_noise = np.uint8((perlin_noise*0.5 + 0.5)*255)
        hsv = cv2.cvtColor(img.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.maximum(v, perlin_noise)
        hsv_new = cv2.merge([h, s, v])
        img.image = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
     
        return self.alpha_blend(img, old_image)

    def basic_noise(self):
        old_image = self.uncannyCam.img.copy()
        img = self.uncannyCam.img
        img.image = utils.noiseFilter(img.image)
        return self.alpha_blend(img, old_image) 

    def apply(self) -> np.ndarray:
        return {
            0: self.basic_noise,
            1: self.perlin_noise,
        }[self.mode]()

    def alpha_blend_value(self):
        return self.slider_value / 100


class CheeksFilter(Effect):
    def __init__(self, uncannyCam, withCuda=True) -> None:
        super().__init__(uncannyCam)
        # left and right cheeck
        self.polygon_indices = [
            [111, 100, 207, 147, 123],
            [346, 347, 329, 423, 376, 427],
        ]
        self.withCuda = withCuda
        self.slider_value = 10

    def apply(self) -> np.ndarray:
        img = self.uncannyCam.img
        shifted = self.hueShift(img.image)
        if img.landmarks:
            mask = utils.getBlurredMask(
                img.image.shape, self.denormalized_polygons(), self.withCuda
            )

            img.image = np.uint8(shifted * mask + img.image * (1 - mask))
        return img

    def hue_difference(self):
        return 180 - self.slider_value

    def hueShift(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_new = np.mod(h + self.hue_difference(), 180).astype(np.uint8)
        hsv_new = cv2.merge([h_new, s, v])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    def denormalized_polygons(self):
        polygons = []
        for index, _ in enumerate(self.polygon_indices):
            polygons.append(
                self.uncannyCam.img.get_denormalized_landmarks(
                    self.polygon_indices[index]
                )
            )
        return polygons


class DebuggingFilter(Effect):
    def __init__(self, uncannyCam, mode=1) -> None:
        super().__init__(uncannyCam)
        self.index = 0
        self.landmarks_indices = None

    def apply(self) -> np.ndarray:
        img = self.uncannyCam.img
        img.drawLandmarks()
        landmarks = img.landmarks_denormalized[0]
        if not self.landmarks_indices:
            self.landmarks_indices = list(enumerate(landmarks))
            self.landmarks_indices.sort(key=lambda x: x[1][1])
            self.landmarks_indices = [item[0] for item in self.landmarks_indices]
        if keyboard.is_pressed(" ") and self.index < len(landmarks):
            self.index += 1
        cv2.circle(
            img.image, landmarks[self.landmarks_indices[self.index]], 0, (0, 0, 255), 2
        )
        if keyboard.is_pressed("i"):
            print(f"Index: {self.landmarks_indices[self.index]}")
        return self.uncannyCam.img
