import utils
import cv2
import numpy as np


def insertTriangles(
    img,
    swapImg,
    triangleIndices,
    pointIndices,
    leaveOutPoints=None,
    withSeamlessClone=False,
):
    """Moves the triangles specified by triangleIndices from swapImg to the coresponding places in img"""
    swap_triangles = getTriangles(swapImg, triangleIndices)
    triangles = getTriangles(img, triangleIndices)

    # newFace is the transformed part of the face with a black background
    newFace = np.zeros_like(img._raw)
    for dst_triangle, swap_triangle in zip(triangles, swap_triangles):
        dst_triangle = np.float32(dst_triangle)
        swap_triangle = np.float32(swap_triangle)
        newFace = displace(newFace, swapImg._raw, dst_triangle, swap_triangle)
    points = img.get_denormalized_landmarks(pointIndices)
    if leaveOutPoints:
        leaveOutPoints = img.get_denormalized_landmarks_nested(leaveOutPoints)
    return insertNewFace(img._raw, newFace, points, leaveOutPoints, withSeamlessClone)


def insertNewFace(img, newFace, points, leaveOutPoints=None, withSeamlessClone=False):
    convexhull = cv2.convexHull(np.array(points, np.int32))
    head_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
    face_mask = cv2.fillConvexPoly(head_mask, convexhull, 255)
    head_mask = cv2.bitwise_not(face_mask)

    seam_clone = img.copy()
    noFace = cv2.bitwise_and(seam_clone, seam_clone, mask=head_mask)
    face = cv2.add(noFace, newFace)
    if withSeamlessClone:
        (x, y, w, h) = cv2.boundingRect(convexhull)
        if x >= 0 and x + w < face.shape[1] and y >= 0 and y + h < face.shape[0]:
            center_face2 = (int(x + w / 2), int(y + h / 2))
            face = cv2.seamlessClone(
                face, img, face_mask, center_face2, cv2.MIXED_CLONE
            )

    white_background = cv2.add(cv2.cvtColor(head_mask, cv2.COLOR_GRAY2BGR), newFace)
    _, mask = cv2.threshold(
        cv2.cvtColor(white_background, cv2.COLOR_BGR2GRAY),
        1,
        255,
        cv2.THRESH_BINARY_INV,
    )
    if leaveOutPoints:
        for points in leaveOutPoints:
            head_mask = cv2.fillConvexPoly(
                mask, cv2.convexHull(np.array(points, np.int32)), 255
            )
    return np.where(
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) == np.array([255, 255, 255]), img, face
    )


# replace all new triangles with old triangles
def displace(dest_img, src_img, dest_triangle, src_triangle):
    """Move a triangle of img_swap specified by triangleSwap to a triangle of img specified by triangle"""
    x, y, w, h = cv2.boundingRect(src_triangle)
    cropped_src_img = src_img[y : y + h, x : x + w]
    cropped_src_triangle = np.array(
        [
            [src_triangle[0][0] - x, src_triangle[0][1] - y],
            [src_triangle[1][0] - x, src_triangle[1][1] - y],
            [src_triangle[2][0] - x, src_triangle[2][1] - y],
        ],
        np.float32,
    )

    x, y, w, h = cv2.boundingRect(dest_triangle)
    cropped_dest_img = dest_img[y : y + h, x : x + w]
    cropped_dest_triangle = np.array(
        [
            [dest_triangle[0][0] - x, dest_triangle[0][1] - y],
            [dest_triangle[1][0] - x, dest_triangle[1][1] - y],
            [dest_triangle[2][0] - x, dest_triangle[2][1] - y],
        ],
        np.float32,
    )

    if x < 0 or x + w >= dest_img.shape[1] or y < 0 or y + h >= dest_img.shape[0]:
        return dest_img

    transform = cv2.getAffineTransform(cropped_src_triangle, cropped_dest_triangle)
    if cv2.determinant(transform[0:2, 0:2]) < 0:
        return dest_img

    warped_src = cv2.warpAffine(
        cropped_src_img, transform, (w, h), flags=cv2.INTER_NEAREST
    )

    mask = np.zeros((h, w, 3), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(cropped_dest_triangle), (1, 1, 1), 16, 0)
    warped_src = warped_src * mask
    newFaceGray = cv2.cvtColor(cropped_dest_img, cv2.COLOR_BGR2GRAY)
    _, newFaceAreaMask = cv2.threshold(newFaceGray, 1, 1, cv2.THRESH_BINARY_INV)
    warped_src = warped_src * mask
    # cropped_dest_img = cv2.add(cropped_dest_img, warped_src)

    dest_img[y : y + h, x : x + w] = (
        dest_img[y : y + h, x : x + w] * ((1.0, 1.0, 1.0) - mask) + warped_src
    )
    return dest_img


# get triangles as points given the indices of the triangulation
def getTriangles(image, triangles):
    outTriangles = []
    for faceId in range(len(image.landmarks_denormalized)):
        for triangle in triangles:
            outTriangles.append(image.get_denormalized_landmarks(triangle, faceId))
    return outTriangles


# get indices of the triangulation
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


# get triangulation as points
def initialTriangles(img, indices):
    polygon = img.get_denormalized_landmarks(utils.find_polygon(indices))
    rect = (0, 0, img._raw.shape[1], img._raw.shape[0])
    subDiv = cv2.Subdiv2D(rect)
    subDiv.insert(polygon)
    triangleList = subDiv.getTriangleList()
    return triangleList


# create dictionary mapping points to indices in facemesh
def pointsToIndices(img, indices):
    distinct = utils.distinct_indices(indices)
    points = {}
    for faceId in range(len(img.landmarks)):
        for index in distinct:
            landmark_denormalized = img.get_denormalized_landmark(index)
            if landmark_denormalized not in points:
                points[landmark_denormalized] = index
    return points
