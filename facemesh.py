import cv2
import mediapipe as mp
import numpy as np
import utils as ut
import time

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mpSelfieSeg = mp.solutions.selfie_segmentation

def drawOnFace(img, landmarks, option):
    height, width, c = img.shape
    polygonOval = ut.getPolygon(height, width, landmarks, mpFaceMesh.FACEMESH_FACE_OVAL)
    polygonEye = ut.getPolygon(height, width, landmarks, mpFaceMesh.FACEMESH_LEFT_EYE)

    if option == "faceContour":
        ut.drawPolygon(img, polygonOval)
    elif option == "landmarks":
        ut.drawLandmarks(img, landmarks)
    elif option == "eyeContour":
        ut.drawPolygon(img, polygonEye)
    
        
def filterFace(img, landmarks):
    height, width, c = img.shape
    polygonOval = ut.getPolygon(height, width, landmarks, mpFaceMesh.FACEMESH_FACE_OVAL)
    return ut.filterFace(img, polygonOval)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened(): # try to get the first frame
        rval, frame = cap.read()
    else:
        rval = False

    faceMesh = mpFaceMesh.FaceMesh(refine_landmarks=True)
    selfieSeg = mpSelfieSeg.SelfieSegmentation(model_selection=0)
    startTime = time.time()
    old = None

    while rval:
        rval, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultsFM = faceMesh.process(imgRGB)
        resultsSS = selfieSeg.process(img)
        if resultsFM.multi_face_landmarks:
            if (time.time() - startTime) % 1 < 0.1:
                old = img
            #mask = ut.getMask(img.shape, ut.getPolygon(img.shape[0], img.shape[1], resultsFM.multi_face_landmarks, mpFaceMesh.FACEMESH_LEFT_EYE))
            #img = ut.displace(img, old, mask)

            #drawOnFace(img, resultsFM.multi_face_landmarks, 'faceContour')
            #img = filterFace(img, resultsFM.multi_face_landmarks)
            img = ut.segmentationFilter(img, resultsSS.segmentation_mask)

        cv2.imshow("Image", img)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break


if __name__ == "__main__":
    main()