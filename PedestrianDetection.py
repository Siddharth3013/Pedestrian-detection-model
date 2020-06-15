import numpy as np
import cv2

video = cv2.VideoCapture("Resources/Pedestrians.mp4")

subject_cascade = cv2.CascadeClassifier("Resources/haarcascade_fullbody.xml")

while True:
    success, image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    human = subject_cascade.detectMultiScale(gray, 1.1, 4)

    for(x, y, w, h) in human:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("OUTPUT", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()