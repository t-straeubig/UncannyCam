import cv2
import numpy as np
import utils


def insert_triangles(
    img,
    swap_img,
    triangle_indices,
    point_indices,
    leave_out_points=None,
    with_seamless_clone=False,
) -> np.ndarray:
    """Moves the triangles specified by triangleIndices from swapImg to the coresponding places in img"""
    swap_triangles = swap_img.get_denormalized_landmarks_nested(triangle_indices)
    triangles = img.get_denormalized_landmarks_nested(triangle_indices)

    # new_face is the transformed part of the face with a black background
    new_face = np.zeros_like(img.raw)
    for dst_triangle, swap_triangle in zip(triangles, swap_triangles):
        dst_triangle = np.float32(dst_triangle)
        swap_triangle = np.float32(swap_triangle)
        new_face = displace(new_face, swap_img.raw, dst_triangle, swap_triangle)
    points = img.get_denormalized_landmarks(point_indices)
    if leave_out_points:
        leave_out_points = img.get_denormalized_landmarks_nested(leave_out_points)
    return insert_new_face(
        img.raw, new_face, points, leave_out_points, with_seamless_clone
    )


def insert_new_face(
    img, new_face, point_list, leave_out_points=None, with_seamless_clone=False
) -> np.ndarray:
    convexhull = cv2.convexHull(np.array(point_list, np.int32))
    head_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
    face_mask = cv2.fillConvexPoly(head_mask, convexhull, 255)
    head_mask = cv2.bitwise_not(face_mask)

    seam_clone = img.copy()
    no_face = cv2.bitwise_and(seam_clone, seam_clone, mask=head_mask)
    face = cv2.add(no_face, new_face)
    if with_seamless_clone:
        (x, y, w, h) = cv2.boundingRect(convexhull)
        if x >= 0 and x + w < face.shape[1] and y >= 0 and y + h < face.shape[0]:
            center_face2 = (int(x + w / 2), int(y + h / 2))
            face = cv2.seamlessClone(
                face, img, face_mask, center_face2, cv2.MIXED_CLONE
            )

    white_background = cv2.add(cv2.cvtColor(head_mask, cv2.COLOR_GRAY2BGR), new_face)
    _, mask = cv2.threshold(
        cv2.cvtColor(white_background, cv2.COLOR_BGR2GRAY),
        1,
        255,
        cv2.THRESH_BINARY_INV,
    )
    if leave_out_points:
        for point_list in leave_out_points:
            head_mask = cv2.fillConvexPoly(
                mask, cv2.convexHull(np.array(point_list, np.int32)), 255
            )
    return np.where(
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) == np.array([255, 255, 255]), img, face
    )


def displace(dest_img, src_img, dest_triangle, src_triangle) -> np.ndarray:
    """Move a triangle of src_img specified by src_triangle to a triangle of dest_img specified by dest_triangle"""
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

    dest_img[y : y + h, x : x + w] = (
        dest_img[y : y + h, x : x + w] * ((1.0, 1.0, 1.0) - mask) + warped_src
    )
    return dest_img


# get indices of the triangulation
def get_triangle_indices(img, indices):
    triangles = initial_triangles(img, indices)
    indices_dict = points_to_indices(img, indices)
    triangle_indices = []
    for t in triangles:
        t = t.astype(int)
        i1 = indices_dict[(t[0], t[1])]
        i2 = indices_dict[(t[2], t[3])]
        i3 = indices_dict[(t[4], t[5])]
        triangle_indices.append([i1, i2, i3])
    return triangle_indices


# get triangulation as points
def initial_triangles(img, indices):
    polygon = img.get_denormalized_landmarks(utils.find_polygon(indices))
    rect = (0, 0, img.raw.shape[1], img.raw.shape[0])
    sub_div = cv2.Subdiv2D(rect)
    sub_div.insert(polygon)
    triangle_list = sub_div.getTriangleList()
    return triangle_list


# create dictionary mapping points to indices in facemesh
def points_to_indices(img, indices):
    distinct = utils.distinct_indices(indices)
    points = {}
    for face_id in range(len(img.landmarks)):
        for index in distinct:
            landmark_denormalized = img.get_denormalized_landmark(index)
            if landmark_denormalized not in points:
                points[landmark_denormalized] = index
    return points
