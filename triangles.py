import utils
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mpFaceMesh

   
def insertTriangles(img, swapImg, landmarks, landmarksOld, triangleIndices):
    trianglesOld = getTriangles(swapImg, landmarksOld, triangleIndices)
    trianglesNew = getTriangles(img, landmarks, triangleIndices)

    #newFace is the transformed part of the face with a black background
    newFace = np.zeros_like(img)
    for i, t in enumerate(trianglesNew):
        t2 = trianglesOld[i]
        triangle = np.float32([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]])
        triangleSwap = np.float32([[t2[0], t2[1]], [t2[2], t2[3]], [t2[4], t2[5]]])
        newFace = displace(newFace, swapImg, triangle, triangleSwap)

    points = utils.getPointCoordinates(img.shape[0], img.shape[1], landmarks, mpFaceMesh.FACEMESH_LEFT_EYE)
    return insertNewFace(img, newFace, points)


def insertNewFace(img, newFace, points, withSeamlessClone=False):
    convexhull = cv2.convexHull(np.array(points, np.int32))
    head_mask =  cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
    face_mask = cv2.fillConvexPoly(head_mask, convexhull, 255)
    head_mask = cv2.bitwise_not(face_mask)

    seam_clone = img.copy()
    noFace = cv2.bitwise_and(seam_clone, seam_clone, mask=head_mask)
    face = cv2.add(noFace, newFace)
    if withSeamlessClone:
        (x, y, w, h) = cv2.boundingRect(convexhull)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        face = cv2.seamlessClone(face, img, face_mask, center_face2, cv2.MIXED_CLONE)

    white_background = cv2.add(cv2.cvtColor(head_mask, cv2.COLOR_GRAY2BGR), newFace)
    _, mask = cv2.threshold(cv2.cvtColor(white_background, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY_INV)
    return np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) == np.array([255,255,255]), img, face)


#replace all new triangles with old triangles
def displace(img, swapImg, triangle, triangleSwap):
    rectSwap = cv2.boundingRect(triangleSwap)
    (x, y, w, h) = rectSwap
    croppedTriangle = swapImg[y: y + h, x: x + w]
    pointsOld = np.array([[triangleSwap[0][0] -x, triangleSwap[0][1] - y],
                            [triangleSwap[1][0] -x, triangleSwap[1][1] - y], 
                            [triangleSwap[2][0] -x, triangleSwap[2][1] - y]], np.int32)
    
    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect
    croppedMask = np.zeros((h,w), np.uint8)
    points = np.array([[triangle[0][0] -x, triangle[0][1] - y],
                        [triangle[1][0] -x, triangle[1][1] - y], 
                        [triangle[2][0] -x, triangle[2][1] - y]], np.int32)
    cv2.fillConvexPoly(croppedMask, points, 255)

    points = np.float32(points)
    pointsOld = np.float32(pointsOld)
    matrix = cv2.getAffineTransform(pointsOld, points)
    warpedTriangle = cv2.warpAffine(croppedTriangle, matrix, (w, h), flags=cv2.INTER_NEAREST)
    warpedTriangle = cv2.bitwise_and(warpedTriangle, warpedTriangle, mask = croppedMask)

    newFaceArea = img[y: y + h, x: x + w] 
    newFaceGray = cv2.cvtColor(newFaceArea, cv2.COLOR_BGR2GRAY)
    _, newFaceAreaMask = cv2.threshold(newFaceGray, 1, 255, cv2.THRESH_BINARY_INV)
    warpedTriangle = cv2.bitwise_and(warpedTriangle, warpedTriangle, mask=newFaceAreaMask)
    newFaceArea = cv2.add(newFaceArea, warpedTriangle)

    img[y: y + h, x: x + w] = newFaceArea
    return img


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
