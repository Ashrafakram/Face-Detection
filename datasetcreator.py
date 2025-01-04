#import Pakages
import cv2
#import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  #-> To Detect Faces in the Camera
cam = cv2.VideoCapture(0) #-> 0 is used for webcamera detection

def insertorupdate(Id,Name,age):   #-> Function for Sqlite Dataset
    conn = sqlite3.connect("database.db")   #->Connecting
    with open("database.sql", "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    cmd = "Select * From  Creato Where ID ="+str(Id)
    cursor = conn.execute(cmd)    #cursor to execute
    isRecordExist = 0              #-> Assume there is no record to our table
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist == 1):
        conn.execute("Update Creato set Name = ? Where Id=?", (Name, Id,))
        conn.execute("Update Creato set age = ? Where Id=?", (age, Id,))
    else:
        conn.execute("Insert into Creato (Id, Name, age) Values(?, ?, ?)",(Id, Name, age))

    conn.commit()
    conn.close()

#Insert User-Defined Values into Tables
Id = input("Enter User Id")
Name = input("Enter User Name")
age = input("Enter User Age")

insertorupdate(Id, Name, age)

#detecting face in the web camera

sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        sampleNum = sampleNum+1
        cv2.imwrite("dataset/user."+str(Id)+"."+str(sampleNum)+".jpg",gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)  # Correct function call
    cv2.imshow("Face Showing", img)
    cv2.waitKey(1)
    if(sampleNum>20):
        break

cam.release()
cv2.destroyAllWindows()



