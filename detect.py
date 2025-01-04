import cv2
import numpy as np
import os
import sqlite3

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  #-> To Detect Faces in the Camera
cam = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdatas.yml")

def getprofile(id):
    conn = sqlite3.connect("database.db")   #->Connecting
    # with open("database.sql", "r") as f:
    #     sql_script = f.read()
    # conn.executescript(sql_script)
    cursor = conn.execute("Select * From Creato where id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        profile = getprofile(id)
        print(profile)
        if(profile!=None):
            cv2.putText(img, "Name:"+str(profile[1]), (x,y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Age:" + str(profile[2]), (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

    cv2.imshow("FACE", img)
    if(cv2.waitKey(1)==ord('g')):
        break

cam.release()
cv2.destroyAllWindows()
