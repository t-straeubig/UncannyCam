from abc import ABC, abstractmethod
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
    def __init__(self) -> None:
        self.intensity = 0
        self.reset_flag: bool = False

    @abstractmethod
    def apply(self, image: Image) -> None:
        """Applies the effect to the Image, changing its contents"""
        pass

    def set_intensity(self, value) -> None:
        self.intensity = value

    def alpha_blend_value(self) -> None:
        return self.intensity / 10

    def alpha_blend(self, new_img: Image, old_image: Image) -> None:
        """blends old_image into new_image with self.alpha_blend_value()"""
        new_img.change_image(
            cv2.addWeighted(
                old_image._raw,
                1 - self.alpha_blend_value(),
                new_img._raw,
                self.alpha_blend_value(),
                0,
            )
        )

    def reset(self):
        self.reset_flag = True


class EyeEffect(Effect, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.landmarks = []
        self.eye_triangles = []
        self.eye_points = [
            utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE),
            utils.distinct_indices(mpFaceMesh.FACEMESH_RIGHT_EYE),
        ]
        self.intensity = 1

    def apply(self, image: Image) -> None:

        if self.reset_flag:
            self.eye_triangles = []

        if not image.landmarks:
            return image

        if not self.eye_triangles:
            self.eye_triangles.append(
                triangles.getTriangleIndices(image, mpFaceMesh.FACEMESH_LEFT_EYE)
            )
            self.eye_triangles.append(
                triangles.getTriangleIndices(image, mpFaceMesh.FACEMESH_RIGHT_EYE)
            )

        if self.is_deactivated():
            return

        self.before_swap(image)
        self.swap(image)
        self.reset_flag = False

    def before_swap(self, image: Image) -> None:
        pass

    def swap(self, image: Image) -> None:
        swap_img = self.get_swap_image()
        image.change_image(
            triangles.insertTriangles(
                image, swap_img, self.eye_triangles[0], self.eye_points[0]
            )
        )

    def is_deactivated(self) -> bool:
        return False

    @abstractmethod
    def get_swap_image(self) -> Image:
        pass


class EyeFreezer(EyeEffect):
    def __init__(self) -> None:
        super().__init__()
        self.swap_image = None

    def before_swap(self, image: Image) -> None:
        if self.reset_flag:
            self.swap_image = None

        if not self.swap_image:
            self.swap_image = image.copy()

    def swap(self, image: Image) -> None:
        super().swap(image)
        if self.intensity == 2:
            swap_img = self.get_swap_image()
            image.change_image(
                triangles.insertTriangles(
                    image, swap_img, self.eye_triangles[1], self.eye_points[1]
                )
            )

    def get_swap_image(self) -> Image:
        return self.swap_image

    def is_deactivated(self) -> bool:
        return self.intensity == 0


class LazyEye(EyeEffect):
    def __init__(self) -> None:
        super().__init__()
        self.images = []
        self.intensity = 5

    def before_swap(self, image: Image) -> None:
        if self.reset_flag:
            self.images = []
        if len(self.images) < self.intensity:
            # Add an additional image to the queue to make it longer
            self.images.append(image.copy())
        self.images.append(image.copy())

    def is_deactivated(self) -> bool:
        return self.intensity == 0 and len(self.images) == 0

    def get_swap_image(self) -> Image:
        image = self.images.pop(0)
        if len(self.images) > self.intensity:
            # Remove an additional image from the queue to make it shorter
            image = self.images.pop(0)
        return image


class FaceSwap(Effect):
    def __init__(self) -> None:
        super().__init__()
        self.last_image: Image = None
        self.swapImg: Image = None
        self.intensity = 10
        self.triangles = tmp.TRIANGULATION_NESTED
        self.points = utils.distinct_indices(tmp.TRIANGULATION_NESTED)
        self.leaveOutPoints = [
            utils.distinct_indices(mpFaceMesh.FACEMESH_LEFT_EYE),
            utils.distinct_indices(mpFaceMesh.FACEMESH_RIGHT_EYE),
        ]

    def apply(self, image: Image) -> None:
        old_image = image.copy()
        self.last_image = image.copy()
        if keyboard.is_pressed("s"):
            self.change_swap_image()
        if not image.landmarks or not self.swapImg:
            return
        self.swap_into(image)
        self.alpha_blend(image, old_image)

    def swap_into(self, image: Image) -> None:
        """inserts the contents of self.swapImg into the image"""
        image.change_image(
            triangles.insertTriangles(
                image,
                self.swapImg,
                self.triangles,
                self.points,
                self.leaveOutPoints,
                withSeamlessClone=True,
            )
        )

    def change_swap_image(self) -> None:
        if not self.last_image.landmarks:
            return
        self.swapImg = self.last_image


class FaceSymmetry(Effect):
    def __init__(self) -> None:
        super().__init__()
        self.triangles = tmp.TRIANGULATION_NESTED
        self.points = utils.distinct_indices(tmp.TRIANGULATION_NESTED)
        self.flipped = None
        self.intensity = 10

    def apply(self, image: Image) -> None:
        if not image.landmarks:
            return
        self.update_flipped_image(image)
        if not self.flipped.landmarks:
            return
        old_image = image.copy()
        image.change_image(
            triangles.insertTriangles(
                image, self.flipped, self.triangles, self.points, withSeamlessClone=True
            )
        )
        self.alpha_blend(image, old_image)

    def update_flipped_image(self, image: Image):
        if not self.flipped:
            self.flipped = Image(image.flipped())
        else:
            self.flipped.change_image(image.flipped(), reprocess=True)


class FaceFilter(Effect):
    def __init__(self, mode=3, bilateralFilter=True) -> None:
        super().__init__()
        self.mode = mode
        if bilateralFilter:
            self.intensity = 30
        else:
            self.intensity = 3
        self.bilateralFilter = bilateralFilter

    def filter_face(self, image: Image) -> None:
        polygon = utils.find_polygon(mpFaceMesh.FACEMESH_FACE_OVAL)
        image.filter_polygon(polygon)

    def filter_person(self, image: Image) -> None:
        image.filter_segmentation()

    def filter_triangle(self, image: Image) -> None:
        indices = utils.distinct_indices(mpFaceMesh.FACEMESH_TESSELATION)
        triangle = [indices[50], indices[260], indices[150]]
        image.filter_polygon(triangle)

    def filter_image(self, image: Image) -> None:
        if self.bilateralFilter:
            image.change_image(utils.cudaBilateralFilter(image._raw, self.intensity))
        else:
            image.change_image(utils.cudaMorphologyFilter(image._raw, self.intensity))

    def apply(self, image: Image) -> None:
        {
            0: self.filter_face,
            1: self.filter_person,
            2: self.filter_triangle,
            3: self.filter_image,
        }[self.mode](image)


class NoiseFilter(Effect):
    def __init__(self, mode=0, precomputed=True) -> None:
        super().__init__()
        self.intensity = 0
        self.mode = mode
        self.precomputed = precomputed
        self.last_noise_index = 0
        self.repeat_counter = 0

    def perlin_noise(self, image: Image) -> None:
        old_image = image.copy()
        if self.precomputed:
            perlin_noise = self.load_noise(image._raw.shape[0], image._raw.shape[1])
        else:
            perlin_noise = generate_perlin_noise_2d(
                (image._raw.shape[0], image._raw.shape[1]), (1, 2)
            )
        hsv = cv2.cvtColor(image._raw, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.maximum(v, perlin_noise)
        hsv_new = cv2.merge([h, s, v])
        image.change_image(cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR))
        self.alpha_blend(image, old_image)

    def load_noise(self, height, width) -> np.ndarray:
        if self.repeat_counter < 3:
            self.repeat_counter += 1
        else:
            self.last_noise_index = (self.last_noise_index + 1) % 10
            self.repeat_counter = 0
        img = cv2.imread(f"noise/noise{self.last_noise_index}.png", 0)
        img = cv2.resize(img, (width, height))
        return img

    def basic_noise(self, image: Image) -> None:
        old_image = image.copy()
        image.change_image(utils.noiseFilter(image._raw))
        self.alpha_blend(image, old_image)

    def apply(self, image: Image) -> None:
        return {0: self.basic_noise, 1: self.perlin_noise,}[
            self.mode
        ](image)

    def alpha_blend_value(self) -> float:
        return self.intensity / 100


class HueShift(Effect):
    def __init__(self) -> None:
        super().__init__()
        self.intensity = 60

    def apply(self, image: Image) -> None:
        raw = image._raw

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
        shifted[:, :, 0] = np.uint8(
            np.mod(np.int32(shifted[:, :, 0]) + self.intensity, 180)
        )
        shifted = cv2.cvtColor(shifted, cv2.COLOR_HSV2BGR)

        # apply the effect
        new_raw = np.uint8(factor * shifted + (1 - factor) * raw)
        image.change_image(new_raw)


class CheeksFilter(Effect):
    def __init__(self, withCuda=True) -> None:
        super().__init__()
        # left and right cheek
        self.polygon_indices = [
            [111, 100, 207, 147, 123],
            [346, 347, 329, 423, 376, 427],
        ]
        self.withCuda = withCuda
        self.intensity = 10

    def apply(self, image: Image) -> None:
        shifted = self.hueShift(image._raw)
        if not image.landmarks:
            return

        mask = utils.getBlurredMask(
            image._raw.shape, self.denormalized_polygons(image), self.withCuda
        )

        image.change_image(np.uint8(shifted * mask + image._raw * (1 - mask)))

    def hue_difference(self) -> int:
        return 180 - self.intensity

    def hueShift(self, image_bgr: np.ndarray) -> np.ndarray:
        shifted = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        shifted[:, :, 0] = np.uint8(
            np.mod(np.int32(shifted[:, :, 0]) + self.hue_difference(), 180)
        )
        return cv2.cvtColor(shifted, cv2.COLOR_HSV2BGR)

    def denormalized_polygons(self, image: Image):
        return image.get_denormalized_landmarks_nested(self.polygon_indices)


class DebuggingFilter(Effect):
    def __init__(self, mode=1) -> None:
        super().__init__()
        self.index = 0
        self.landmarks_indices = None

    def apply(self, image: Image) -> None:
        image.drawLandmarks()
        landmarks = image.landmarks_denormalized[0]
        if not self.landmarks_indices:
            self.landmarks_indices = list(enumerate(landmarks))
            self.landmarks_indices.sort(key=lambda x: x[1][1])
            self.landmarks_indices = [item[0] for item in self.landmarks_indices]
        if keyboard.is_pressed(" ") and self.index < len(landmarks):
            self.index += 1
        cv2.circle(
            image._raw, landmarks[self.landmarks_indices[self.index]], 0, (0, 0, 255), 2
        )
        if keyboard.is_pressed("i"):
            print(f"Index: {self.landmarks_indices[self.index]}")
