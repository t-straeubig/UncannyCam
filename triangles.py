import utils
import cv2
import numpy as np
import mediapipe as mp

#replace all new triangles with old triangles
def displace(img, oldImg, landmarks, landmarksOld, triangleIndices):
    trianglesOld = getTriangles(oldImg, landmarksOld, triangleIndices)
    trianglesNew = getTriangles(img, landmarks, triangleIndices)
    result = img

    for i, t in enumerate(trianglesNew):
        t2 = trianglesOld[i]
        
        tNew = np.float32([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]])
        tOld = np.float32([[t2[0], t2[1]], [t2[2], t2[3]], [t2[4], t2[5]]])
        mask = utils.getMask(img.shape, tNew)

        transform = cv2.getAffineTransform(tOld, tNew)
        warpedImg = cv2.warpAffine(oldImg, transform, (oldImg.shape[1], oldImg.shape[0]))
        result = np.where(mask == np.array([255,255,255]), warpedImg, result)

    return result

#get triangles as points given the indices of the triangulation
def getTriangles(img, landmarks, triangles):
    outTriangles = []
    for lm in landmarks:
        for t in triangles:
            x1, y1 = utils.denormalize(img.shape[1], img.shape[0], lm.landmark[t[0]])
            x2, y2 = utils.denormalize(img.shape[1], img.shape[0], lm.landmark[t[1]])
            x3, y3 = utils.denormalize(img.shape[1], img.shape[0], lm.landmark[t[2]])
            outTriangles.append([x1, y1, x2, y2, x3, y3])
    return outTriangles

#get indices of the triangulation
def getTriangleIndices(img, landmarks, indices):
    triangles = initialTriangles(img, landmarks, indices)
    indicesDict = pointsToIndices(img.shape[0], img.shape[1], landmarks, indices)
    triangleIndices = []
    for t in triangles:
        t = t.astype(int)
        i1 = indicesDict[(t[0], t[1])]
        i2 = indicesDict[(t[2], t[3])]
        i3 = indicesDict[(t[4], t[5])]
        triangleIndices.append([i1, i2, i3])
    return triangleIndices

#get triangulation as points
def initialTriangles(img, landmarks, indices):
    polygon = utils.getPolygon(img.shape[0], img.shape[1], landmarks, indices)
    rect = (0, 0, img.shape[1], img.shape[0])
    subDiv = cv2.Subdiv2D(rect)
    subDiv.insert(polygon)
    triangleList = subDiv.getTriangleList()              
    return triangleList 
    
#create dictionary mapping points to indices in facemesh
def pointsToIndices(height, width, landmarks, indices):
    points = {}
    for faceLms in landmarks:
        for line in indices:
            i, j = line
            x1, y1 = utils.denormalize(width, height, faceLms.landmark[i])
            x2, y2 = utils.denormalize(width, height, faceLms.landmark[j])
            if (x1,y1) not in points:
                points[(x1,y1)] = i
            if (x2,y2) not in points:
                points[(x2,y2)] = j
    return points  
