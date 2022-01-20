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

def getMask(shape, polygon):
    """Returns an np-array with specified shape where points inside the polygon are white"""
    poly = np.array(polygon)
    mask = np.zeros(shape)
    cv2.fillPoly(mask, np.int32([poly]), color=(255, 255, 255))
    return mask


def distinct_indices(indices):
    distinct = set()
    for poly in indices:
        for point in poly:
            distinct.add(point)
    return list(distinct)

