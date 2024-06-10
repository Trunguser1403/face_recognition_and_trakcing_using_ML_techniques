import cv2
import numpy as np


class Face_Detection(cv2.CascadeClassifier):
    def __init__(self, type = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
        super().__init__(type)
        if self.empty():
            raise IOError("Unable to load the face cascade classifier xml file")

    def detectMultiScale(self, frame):       
        faces = super().detectMultiScale(frame)
        if faces is tuple():
            return []
        faces[:,2:3] = faces[:,0:1] + faces[:,2:3]
        faces[:,3:4] = faces[:,1:2] + faces[:,3:4]
        return faces

