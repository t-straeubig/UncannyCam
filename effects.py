from abc import ABC
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


class EyeEffect(Effect):
    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.images = []
        self.landmarks = []
        self.eye_triangles = []
        self.eye_points = [
            utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE),
            utils.distinct_indices(mpFaceMesh.FACEMESH_RIGHT_EYE),
        ]
        self.slider_value = 1
        self.swap_image = None

    def apply(self) -> np.ndarray:
        img = self.uncannyCam.img
        if not img.landmarks:
            return img

        if not self.eye_triangles:
            self.eye_triangles.append(
                triangles.getTriangleIndices(img, mpFaceMesh.FACEMESH_LEFT_EYE)
            )
            self.eye_triangles.append(
                triangles.getTriangleIndices(img, mpFaceMesh.FACEMESH_RIGHT_EYE)
            )

        self.images.append(img.copy())
        if self.is_deactivated():
            return img

        return self.swap(img)

    def swap(self, img):
        swap_img: Image = self.get_swap_image()
        img.image = triangles.insertTriangles(
            img, swap_img, self.eye_triangles[0], self.eye_points[0]
        )
        return img

    def is_deactivated(self):
        return self.slider_value == 0

    def get_swap_image(self):
        return self.uncannyCam.img


class EyeFreezer(EyeEffect):
    def swap(self, img):
        img = super().swap(img)
        if self.slider_value == 2:
            swap_img: Image = self.get_swap_image()
            img.image = triangles.insertTriangles(
                img, swap_img, self.eye_triangles[1], self.eye_points[1]
            )
        return img

    def get_swap_image(self):
        if len(self.images) > 1:
            self.images.pop()
        return self.images[0]


class LazyEye(EyeEffect):
    def __init__(self, uncannyCam) -> None:
        super().__init__(uncannyCam)
        self.slider_value = 5

    def is_deactivated(self):
        return len(self.images) <= self.slider_value or self.slider_value == 1

    def get_swap_image(self):
        return self.images.pop(0)

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
        perlin_noise = generate_perlin_noise_2d(
            (img.image.shape[0], img.image.shape[1]), (1, 2)
        )
        perlin_noise = np.uint8((perlin_noise * 0.5 + 0.5) * 255)
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


class HueShift(Effect):
    def apply(self) -> np.ndarray:
        image = self.uncannyCam.img
        raw = self.uncannyCam.img.image

        # converting image into LAB and calculate the average color of part of the face
        raw_lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)
        rect = np.float32(image.get_denormalized_landmarks([10, 151, 337, 338]))
        x, y, w, h = cv2.boundingRect(rect)
        avg_color = raw_lab[y : y + h, x : x + w].mean(axis=0).mean(axis=0)

        # calculate distances to the average skin color
        l_diff, a_diff, b_diff = cv2.split(raw_lab - avg_color)
        l_diff = np.square(l_diff / 255)
        a_diff = np.square(a_diff / 255)
        b_diff = np.square(b_diff / 255)

        # converting the color distance to a mask for the effect
        factor = 1 - np.clip(20 * (l_diff + a_diff + b_diff), 0, 1)
        factor = np.repeat(factor[:, :, np.newaxis], 3, axis=2)

        # define the effect to be applied
        shifted = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        shifted[:, :, 0] = np.uint8(np.mod(np.int32(shifted[:, :, 0]) + 180 + 60, 180))
        shifted = cv2.cvtColor(shifted, cv2.COLOR_HSV2BGR)

        # apply the effect
        new_raw = np.uint8(factor * shifted + (1 - factor) * raw)
        image.image = new_raw
        return image


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
        return self.uncannyCam.img.get_denormalized_landmarks_nested(
            self.polygon_indices
        )


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
