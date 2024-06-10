import cv2
from src.models.detect_model import Face_Detection
from src.models.recognize_model import load_model, Face_Recognition

from sort.tracker import SortTracker

import numpy as np
import time

detecter = Face_Detection()
recog = Face_Recognition()
recog.fit_model("face_database")
# recog = load_model("save_pickle/recog_svc_model")

if __name__ == "__main__":

    # vid = cv2.VideoCapture("Test img\Viral_ Biden's Staff Cuts Short His Vietnam Press Briefing As U.S. Pres Starts Rambling _ Watch.mp4") 
    vid = cv2.VideoCapture(0)
    start_time = time.time()

    while(True): 
        ret, frame = vid.read() 

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detecter.detectMultiScale(gray_frame)
        re = recog.recognize(gray_frame, faces)

        for (x1, y1, x2, y2,name_prob, name) in re:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
            cv2.putText(frame, name, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, name_prob, (x2 - 30, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
    vid.release() 
    cv2.destroyAllWindows()