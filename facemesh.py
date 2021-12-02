import cv2
import mediapipe as mp
import numpy as np
import utils as ut
import time
import pyvirtualcam
from mediapipe.python.solutions import \
    drawing_utils as mpDraw, \
    face_mesh as mpFaceMesh, \
    selfie_segmentation as mpSelfieSeg

def drawOnFace(img, landmarks, option):
    height, width, _ = img.shape
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
    startTime = 0
    old = None

    with mpFaceMesh.FaceMesh(refine_landmarks=True) as faceMesh, pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
        selfieSeg = mpSelfieSeg.SelfieSegmentation(model_selection=0)
        print(f'Using virtual camera: {cam.device}')
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("No Image could be captured")
                continue


            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resultsFM = faceMesh.process(img)
            resultsSS = selfieSeg.process(img)

            if resultsFM.multi_face_landmarks:
                if (time.time() - startTime) > 0.1:
                    startTime = time.time()
                    old = img
                
                mask = ut.getMask(img.shape, ut.getPolygon(img.shape[0], img.shape[1], resultsFM.multi_face_landmarks, mpFaceMesh.FACEMESH_LEFT_EYE))
                img = ut.displace(img, old, mask)

                drawOnFace(img, resultsFM.multi_face_landmarks, 'faceContour')
                # img = filterFace(img, resultsFM.multi_face_landmarks)
                # img = ut.segmentationFilter(img, resultsSS.segmentation_mask)
            
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cam.send(img)
            # cam.sleep_until_next_frame()
            cv2.imshow("Image", img)
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break


if __name__ == "__main__":
    main()