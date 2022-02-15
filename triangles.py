import utils
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mpFaceMesh

   
def insertTriangles(img, swapImg, triangleIndices, pointIndices, leaveOutPoints=None, withSeamlessClone=False):
    swap_triangles = getTriangles(swapImg, triangleIndices)
    triangles = getTriangles(img, triangleIndices)

    #newFace is the transformed part of the face with a black background
    newFace = np.zeros_like(img.image)
    for triangle, swap_triangle in zip(triangles, swap_triangles):
        # triangle = np.array(triangle, np.int32)
        triangle = np.float32(triangle)
        swap_triangle = np.float32(swap_triangle)
        newFace = displace(newFace, swapImg.image, triangle, swap_triangle)
    points = img.get_denormalized_landmarks(pointIndices)
    if leaveOutPoints:
        leaveOutPoints = img.get_denormalized_landmarks(leaveOutPoints)
    return insertNewFace(img.image, newFace, points, leaveOutPoints, withSeamlessClone)


def insertNewFace(img, newFace, points, leaveOutPoints=None, withSeamlessClone=False):
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
    if leaveOutPoints is not None:
        head_mask = cv2.fillConvexPoly(mask, cv2.convexHull(np.array(leaveOutPoints, np.int32)), 255)
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
def getTriangles(image, triangles):
    outTriangles = []
    for faceId in range(len(image.landmarks_denormalized)):
        for triangle in triangles:
            outTriangles.append(image.get_denormalized_landmarks(triangle, faceId))
    return outTriangles


#get indices of the triangulation
def getTriangleIndices(img, indices):
    triangles = initialTriangles(img, indices)
    indicesDict = pointsToIndices(img, indices)
    triangleIndices = []
    for t in triangles:
        t = t.astype(int)
        i1 = indicesDict[(t[0], t[1])]
        i2 = indicesDict[(t[2], t[3])]
        i3 = indicesDict[(t[4], t[5])]
        triangleIndices.append([i1, i2, i3])
    return triangleIndices

#get triangulation as points
def initialTriangles(img, indices):
    polygon = img.get_denormalized_landmarks(utils.find_polygon(indices))
    rect = (0, 0, img.image.shape[1], img.image.shape[0])
    subDiv = cv2.Subdiv2D(rect)
    subDiv.insert(polygon)
    triangleList = subDiv.getTriangleList()              
    return triangleList 
    
#create dictionary mapping points to indices in facemesh
def pointsToIndices(img, indices):
    distinct = utils.distinct_indices(indices)
    points = {}
    for faceId in range(len(img.landmarks)):
        for index in distinct:
            landmark_denormalized = img.get_denormalized_landmark(index)
            if landmark_denormalized not in points:
                points[landmark_denormalized] = index
    return points  
