from email.mime import image
import cv2
import numpy as np
import mediapipe as mp


def find_polygon(lines):
    """Converts a list of lines to a list of points forming a polygon"""
    lines = list(lines)
    polygon = []
    polygon.append(lines[0][0])
    polygon.append(lines[0][1])
    while len(polygon) <= len(lines)*3:
        for line in lines:
            first, second = line
            if first == polygon[-1] and second != polygon[-2]:
                polygon.append(second)
            elif second == polygon[-1] and first != polygon[-2]:
                polygon.append(first)
    
    return polygon

def find_polygon_denormalized(img, indices):
    polygonIndices = find_polygon(indices)
    return img.get_denormalized_landmarks(0, *polygonIndices)

def getMask(shape, polygon):
    """Returns an np-array with specified shape where points inside the polygon are white"""
    poly = np.array(polygon)
    mask = np.zeros(shape)
    cv2.fillPoly(mask, np.int32([poly]), color=(255, 255, 255))
    return mask

def drawPolygon(img, polygon):
    """Draws the polygon into the image"""
    for i in range(1, len(polygon)):
        cv2.line(img, polygon[i-1], polygon[i], (0, 0, 255), 1)


def filterPolygon(img, outline):
    """Applies a blur filter to the image inside the polygon"""
    height, width, _ = img.image.shape
    polygon = find_polygon_denormalized(img, outline)
    blurred = cv2.bilateralFilter(img.image, 20, 50, 50)
    mask = getMask(img.image.shape, polygon)
    return np.where(mask==np.array([255,255,255]), blurred, img.image)

def segmentationFilter(image):
        background = np.zeros(image.image.shape, dtype=np.uint8)
        background[:] = (0,0,0)
        condition = np.stack((image.selfieSeg_results.segmentation_mask,) * 3, axis=-1) > 0.1
        blurred = cv2.bilateralFilter(image.image, 5, 50, 50)
        cv2.imshow("blurred", blurred)
        return np.where(condition, blurred, image.image)

def drawLandmarks(img, landmarks):
    """Draws points at the landmarks"""
    height, width, _ = img.shape
    for faceLms in landmarks:
        for landmark in faceLms.landmark:
            point = denormalize(width, height, landmark)
            cv2.circle(img, point, 0, (255,0,0), 2)

def distinct_indices(indices):
    distinct = set()
    for poly in indices:
        for point in poly:
            distinct.add(point)
    return list(distinct)


def getPointCoordinates(height, width, landmarks, indices):
    points = []
    for faceLms in landmarks:
        for i, j in indices:
            point = denormalize(width, height, faceLms.landmark[i])
            points.append(point)
    return points

def drawLines(img, lines):
    for i, j in lines:
        cv2.line(img, i, j, (0,255,00), 1)

def drawPoints(img, lines):
    for i, j in lines:
        cv2.circle(img, i, 0, (255,0,0), 2)
        cv2.circle(img, j, 0, (255,0,0), 2)
