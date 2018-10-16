
# coding: utf-8

# In[1]:


import cv2
import dlib
import imutils
import time
import numpy as np
from imutils import face_utils
#from playsound import playsound


def eye_aspect_ratio(eye):
    
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    
    ear = (A+B) / (2.0 * C)
    
    return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 35
EYE_COUNTER = 0
TOTAL = 0
ALARM = False

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("C:\\Users\\spandan\\Documents\\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video_capture = cv2.VideoCapture(0)
cam_w = 640
cam_h = 480

while True:
        
    _, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    for rect in rects:
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        temp = leftEye
        leftEye = rightEye
        rightEye = temp

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        diff_ear = np.abs(leftEAR - rightEAR)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        if ear < EYE_AR_THRESH:
            EYE_COUNTER += 1
            
            
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM:
                    ALARM = True
                    
                #playsound('C:\\Users\\spandan\\Downloads\\alarm.wav')
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
            
            
        else:
            EYE_COUNTER = 0
            ALARM = False
            
        
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame",frame)
    k = cv2.waitKey(1) & 0xFF
                             
    if k%256 == 27: 
        break

video_capture.release()
cv2.destroyAllWindows()

