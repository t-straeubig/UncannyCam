import cv2
import numpy as np
import mediapipe as mp

def denormalize(width, height, landmark):
    x = int(width * landmark.x)
    y = int(height * landmark.y)
    return x, y


def getLines(height, width, landmarks, indices):
    """Returns a list of landmark lines in absolute coordinates"""
    lines = []
    for faceLms in landmarks:
        for i, j in indices:
            p1 = denormalize(width, height, faceLms.landmark[i])
            p2 = denormalize(width, height, faceLms.landmark[j])
            lines.append((p1, p2))
    return lines

def findPolygon(lines):
    """Converts a list of lines to a list of points forming a polygon"""
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

def getPolygon(height, width, landmarks, indices):
    lines = getLines(height, width, landmarks, indices)
    return findPolygon(lines)

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


def filterPolygon(img, landmarks, outline):
    """Applies a blur filter to the image inside the polygon"""
    height, width, _ = img.shape
    polygon = getPolygon(height, width, landmarks, outline)
    blurred = cv2.bilateralFilter(img, 10, 50, 50)
    mask = getMask(img.shape, polygon)
    return np.where(mask==np.array([255,255,255]), blurred, img)

def segmentationFilter(img, mask):
        background = np.zeros(img.shape, dtype=np.uint8)
        background[:] = (0,0,0)
        condition = np.stack((mask,) * 3, axis=-1) > 0.1
        blurred = cv2.bilateralFilter(img, 10, 50, 50)
        return np.where(condition, blurred, img)

def drawLandmarks(img, landmarks):
    """Draws points at the landmarks"""
    height, width, _ = img.shape
    for faceLms in landmarks:
        for landmark in faceLms.landmark:
            point = denormalize(width, height, landmark)
            cv2.circle(img, point, 0, (255,0,0), 2)
   
def getPointCoordinates(height, width, landmarks, indices):
    points = []
    for faceLms in landmarks:
        for i, j in indices:
            point = denormalize(width, height, faceLms.landmark[i])
            points.append(points)
    return points

def drawLines(img, lines):
    for i, j in lines:
        cv2.line(img, i, j, (0,255,00), 1)

