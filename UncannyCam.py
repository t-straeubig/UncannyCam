from typing import List
import cv2
import pyvirtualcam
from effects import Effect, EyeFreezer, FaceFilter, FaceSwap, FaceSymmetry
from imagetools import Image

class UncannyCam():
    def __init__(self) -> None:
        self.img = None
        self.testMode = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.effects: List[Effect] = []
        # self.effects.append(FaceSwap(self))
        # self.effects.append(EyeFreezer(self))
        # self.effects.append(FaceFilter(self))
        self.effects.append(FaceSymmetry(self))
        self.cam = pyvirtualcam.Camera(width=width, height=height, fps=20)
        print(f'Using virtual camera: {self.cam.device}')

    
    def mainloop(self) -> None:
        while self.cap.isOpened():
            success, self.imgraw = self.cap.read()
            if not success:
                print("No Image could be captured")
                continue
            
            # self.imgraw = cv2.cvtColor(self.imgraw, cv2.COLOR_BGR2RGB)
            if not self.img:
                self.img = Image(self.imgraw, selfieseg=True)
            else:
                self.img.change_image(self.imgraw, reprocess=True)
            for effect in self.effects:
                self.img = effect.apply()

            if self.testMode:
                # self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Image", self.img.image)
                key = cv2.waitKey(20)
                if key == 27: # exit on ESC
                    break
                    
            else:
                img = cv2.cvtColor(self.img.image, cv2.COLOR_BGR2RGB)
                self.cam.send(img)
                self.cam.sleep_until_next_frame()

        print("main loop terminated")

if __name__ == "__main__":
    uncannyCam = UncannyCam()
    uncannyCam.mainloop()
