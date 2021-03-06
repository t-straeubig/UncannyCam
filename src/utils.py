import cv2
import numpy as np


def find_polygon(lines):
    """Converts a list of lines to a list of points forming a polygon"""
    lines = list(lines)
    polygon = []
    polygon.append(lines[0][0])
    polygon.append(lines[0][1])
    while len(polygon) <= len(lines) * 3:
        for line in lines:
            first, second = line
            if first == polygon[-1] and second != polygon[-2]:
                polygon.append(second)
            elif second == polygon[-1] and first != polygon[-2]:
                polygon.append(first)

    return polygon


def get_mask(shape, polygon) -> np.ndarray:
    """Returns an np-array with specified shape where points inside the polygon are white"""
    poly = np.array(polygon)
    mask = np.zeros(shape)
    cv2.fillPoly(mask, np.int32([poly]), color=(255, 255, 255))
    return mask


def get_blurred_mask(shape, polygons, with_cuda) -> np.ndarray:
    """Return blurred mask for alpha-blending with values between 0 and 1"""
    mask = np.zeros(shape)
    for polygon in polygons:
        poly = np.array(polygon)
        cv2.fillPoly(mask, np.int32([poly]), color=(255, 255, 255))

    if with_cuda:
        mask = cuda_gaussian_filter(np.uint8(mask))
    else:
        mask = cv2.GaussianBlur(mask, (65, 65), 0)

    return mask / 255


def image_BGR_RGBA(image):
    b_channel, g_channel, r_channel = cv2.split(image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def cuda_gaussian_filter(image):
    # Use GPU Mat to speed up filtering
    cuda_img = cv2.cuda_GpuMat(cv2.CV_8UC4)
    cuda_img.upload(image_BGR_RGBA(image))
    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC4, -1, (31, 31), 16)
    gaussian_filter.apply(cuda_img, cuda_img)

    result = cuda_img.download()
    return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)


def cuda_bilateral_filter(image, kernel_size):
    # Use GPU Mat to speed up filtering
    cuda_img = cv2.cuda_GpuMat(cv2.CV_8UC4)
    cuda_img.upload(image_BGR_RGBA(image))
    cuda_img = cv2.cuda.bilateralFilter(cuda_img, 10, kernel_size, kernel_size)
    result = cuda_img.download()
    return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)


def cuda_morphology_filter(image, size):
    # Use GPU Mat to speed up filtering
    cuda_img = cv2.cuda_GpuMat(cv2.CV_8UC4)
    cuda_img.upload(image_BGR_RGBA(image))
    filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC4, np.eye(size))
    filter.apply(cuda_img, cuda_img)
    result = cuda_img.download()
    return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)


def noise_filter(image):
    row, col, _ = image.shape
    noise = np.random.normal(0.5, 0.5, (row, col, 1))
    noise = np.uint8(noise * 255)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def distinct_indices(indices):
    distinct = set()
    for poly in indices:
        for point in poly:
            distinct.add(point)
    return list(distinct)
