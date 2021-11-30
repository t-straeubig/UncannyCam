import cv2
import numpy as np
import mediapipe as mp

def findPolygon(lines):
    polygon = []
    polygon.append(lines[0][0])
    polygon.append(lines[0][1])
    while len(polygon) <= len(lines)*3:
        for line in lines:
            first, second = line
            if first == polygon[-1] and first != polygon[-2]:
                polygon.append(second)
            elif second == polygon[-1] and second != polygon[-2]:
                polygon.append(first)
    
    return polygon

def getMask(shape, polygon):
    poly = np.array(polygon)
    mask = np.zeros(shape)
    cv2.fillPoly(mask, np.int32([poly]), color=(255, 255, 255))
    return mask

def drawPolygon(img, polygon):
    for i, p in enumerate(polygon):
        x1,y1 = polygon[i]
        if i+1 < len(polygon):
            x2,y2 = polygon[i+1]
        cv2.line(img,(x1, y1), (x2, y2),(0,255,00),1)

def getPolygon(height, width, landmarks, indices):
    lines = getLines(height, width, landmarks, indices)
    return findPolygon(lines)

def filterFace(img, polygon):
    blurred = cv2.bilateralFilter(img, 10, 50, 50)
    mask = getMask(img.shape, polygon)
    return np.where(mask==np.array([255,255,255]), blurred, img)

def drawLandmarks(img, landmarks):
    for faceLms in landmarks:
        for lm in faceLms.landmark:
            ih, iw, ic = img.shape
            x, y = int(iw*lm.x), int(ih*lm.y)
            cv2.circle(img, (x,y), 0, (255,0,0), 2)

def getLines(height, width, landmarks, indices):
    lines = []
    for faceLms in landmarks:
        for line in indices:
            i, j = line
            x1, y1 = int(width*faceLms.landmark[i].x), int(height*faceLms.landmark[i].y)
            x2, y2 = int(width*faceLms.landmark[j].x), int(height*faceLms.landmark[j].y)
            lines.append(((x1, y1), (x2, y2)))
    return lines
   
def getPointCoordinates(height, width, landmarks, indices):
    points = []
    for faceLms in landmarks:
        for line in indices:
            i, j = line
            x, y = int(width*faceLms.landmark[i].x), int(height*faceLms.landmark[i].y)
            points.append((x,y))
    return points  

def drawLines(img, lines):
    for line in lines:
        i, j = line
        cv2.line(img, i, j,(0,255,00),1)

def segmentationFilter(img, mask):
    background = np.zeros(img.shape, dtype=np.uint8)
    background[:] = (0,0,0)
    condition = np.stack((mask,) * 3, axis=-1) > 0.1
    blurred =cv2.bilateralFilter(img, 10, 50, 50)
    return np.where(condition, blurred, img)

def displace(img, oldImg, mask):
    background = np.zeros(img.shape, dtype=np.uint8)
    background[:] = (0,0,0)
    return np.where(mask == np.array([255,255,255]), oldImg, img)

def getPoints(lines):
    points = set()
    for line in lines:
        i, j = line
        points.add(i)
        points.add(j)
    return points  

# def getTriangles(lines):
#     points = getPoints(lines)
#     triangles = []
