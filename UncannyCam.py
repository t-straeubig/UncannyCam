from typing import List
import cv2
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
from effects import Effect, EyeFreezer, FaceFilter


class UncannyCam():
    def __init__(self) -> None:
        self.img = None
        self.testMode = False
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.effects: List[Effect] = []
        self.effects.append(EyeFreezer(self))
        # self.effects.append(FaceFilter(self, 1))
        self.faceMesh = mpFaceMesh.FaceMesh(refine_landmarks=True)
        self.selfieSeg = mpSelfieSeg.SelfieSegmentation(model_selection=0)
        self.cam = pyvirtualcam.Camera(width=width, height=height, fps=20)
        print(f'Using virtual camera: {self.cam.device}')

    
    def mainloop(self) -> None:
        while self.cap.isOpened():
            success, self.img = self.cap.read()
            if not success:
                print("No Image could be captured")
                continue
            
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.faceMesh_results = self.faceMesh.process(self.img)
            self.selfieSeg_results = self.selfieSeg.process(self.img)

            for effect in self.effects:
                self.img = effect.apply()

            if self.testMode:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Image", self.img)
                key = cv2.waitKey(20)
                if key == 27: # exit on ESC
                    break
            else:
                self.cam.send(self.img)
                self.cam.sleep_until_next_frame()

        print("main loop terminated")

if __name__ == "__main__":
    uncannyCam = UncannyCam()
    uncannyCam.mainloop()
