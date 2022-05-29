from gettext import npgettext
import numpy np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

first_read = True

cap = cv2.VideoCapture9(0)
ret,img = cap.read()

while(ret):
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,5,1,1)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200,200))
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            roi_face = gray[y:y+h, x:x+w]
            roi_face_clr = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_face, 1.3,5, minSize=(50,50))

            if(len(eyes)>=2):

                if(first_read):
                    cv2.putText(img,
                    "Eye detected press s to begin",
                    (70,70),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0,255,0),2)
                else:
                    cv2.putText(img,
                    "Eyes open!", (70,70),
                    cv2.FONT_HARSHEY_PLAIN, 2,
                    (255,255,255),2)
            else:
                if(first_read):
                    cv2.putText(img,
                    "No eyes detected",(70,70),
                    cv2.FONT_HARSHEY_PLAIN,3,
                    (0,0,255),2)
                else:
                    print("Blink detected---------------")
                    cv2.waitKey(3000)
                    first_read = True
            

            else:
                cv2.putText(img,
                "No face detected", (100,100),
                cv2.FONT_HARSHEY_PLAIN,3,
                (0,255,0),2)
            

            cv2.imshow('img',img)
            a = cv2.waitKey(1)
            if(a == ord('q')):
                break
            elif(a==ord('s') and first_read):
                first_read = False


        cap.release()
        cv2.destroyAllWindows()                

