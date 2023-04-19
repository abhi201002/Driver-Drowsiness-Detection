import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pygame import mixer
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load_model(r'C:\Users\pgupt\Documents\DSP Project\Model\model.h5')

mixer.init()
cap = cv2.VideoCapture(0)
sound = mixer.Sound(r'C:\Users\pgupt\Documents\DSP Project\alarm.wav')
score = 0
while True:
    _,frame = cap.read()
    height,width = frame.shape[0:2]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 3)
    eyes = eye_cascade.detectMultiScale(gray,scaleFactor = 2,minNeighbors = 3)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1 = (x,y),pt2 = (x + w,y + h),color = (255,0,0),thickness = 2)
    for (x,y,w,h) in eyes:
        #cv2.rectangle(frame,pt1 = (x,y),pt2 = (x + w,y + h),color = (0,255,0),thickness = 2)
        eye = frame[y:y+h,x:x+w]
        eye = cv2.resize(eye,(80,80))
        eye = eye/255
        eye = eye.reshape(80,80,3)
        eye = np.expand_dims(eye,axis = 0)
        # eye_gray = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
        prediction = model.predict(eye)
        cv2.putText(frame,str(score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color = (0,255,255),thickness = 1,lineType=cv2.LINE_AA)
        if(prediction[0][0] > 0.5):
            score = score + 1
            if(score > 2):
                cv2.putText(frame,'Closed !!!!',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color = (0,0,255),thickness = 2,lineType=cv2.LINE_AA)
            if(score > 5):
                sound.play()
                # cv2.putText(frame,'alert !!!!',(105,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color = (255,0,255),thickness = 2,lineType=cv2.LINE_AA)
        else:
            cv2.putText(frame,'Open :)',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color = (0,255,0),thickness = 2,lineType=cv2.LINE_AA)
            if(score <= 0):
                score = 0
            else:
                score = score - 1
    cv2.imshow('Frame',frame)
    
    key = cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()