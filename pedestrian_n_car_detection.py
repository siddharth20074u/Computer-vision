# Pedestrian detection

import cv2
import numpy as np

body_classifier = cv2.CascadeClassifier('/users/siddharthsmac/downloads/Haarcascades/haarcascade_fullbody.xml')
cap = cv2.VideoCapture('/users/siddharthsmac/downloads/walking.avi')
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    for (x,y,w,h) in bodies:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('pedestrians', frame)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()

# Car detection

import cv2
import numpy as np
import time

car_classifier = cv2.CascadeClassifier('/users/siddharthsmac/downloads/Haarcascades/haarcascade_car.xml')
cap = cv2.VideoCapture('/users/siddharthsmac/downloads/cars.avi')
while cap.isOpened():
    time.sleep(.05)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.2, 3)
    for (x,y,w,h) in cars:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()
