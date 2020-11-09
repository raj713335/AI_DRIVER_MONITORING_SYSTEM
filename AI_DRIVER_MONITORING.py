
#Import necessary libraries
import os
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2
import requests
import threading

def telegram():

    # pyautogui.screenshot(r"Fraud.png")

    print("start")

    try:

        time.sleep(2)


        token = "1362521589:AAETxO9b_8NLgVpCVe4yD4I5q9U2SwPeYbw"
        chat_id = '@intelegix'  # chat id
        file = 'Fraud.jpg'

        url = f"https://api.telegram.org/bot{token}/sendPhoto"

        print(url)
        files = {}
        files["photo"] = open(file, "rb")
        print(requests.get(url, params={"chat_id": chat_id}, files=files))
    except:
        pass

    print("end")


#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")









#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('haarcascades/shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

base_dir = os.getcwd()
base_dir = base_dir.replace('\\', '/')
# model_store_dir = base_dir + '/SMOKER_DETECTOR/Model/smoking_detector.model'

face_detector_caffe = base_dir + '/SMOKER_DETECTOR/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
prototxtPath = base_dir + '/SMOKER_DETECTOR/face_detector/deploy.prototxt'
weightsPath = face_detector_caffe
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
net=cv2.dnn.readNet("SMOKER_DETECTOR/Model/yolov4.weights","SMOKER_DETECTOR/Model/yolov4.cfg")

classes=["Cigarette","Mobile"]
# maskNet = load_model(model_store_dir)

#Start webcam video capture
video_capture = cv2.VideoCapture(0)



font = cv2.FONT_HERSHEY_SIMPLEX

drowsey_level="False"
Cigarette="False"
Mobile="False"
telegx=0
while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    (H, W) = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    #face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    # for (x,y,w,h) in face_rectangle:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                drowsey_level="True"
        else:
            pygame.mixer.music.stop()
            COUNTER = 0

    cv2.putText(frame, "INTELEGIX (Driver Monitoring System)", (110, 40),
                font, 0.7*2, (255, 255, 255), 2)
    cv2.rectangle(frame, (20, 50), (W - 20, 15), (255, 255, 255), 2)
    # cv2.putText(img, "RISK ANALYSIS", (30, 85),
    #             font, 0.5, (255, 255, 0), 1)
    # cv2.putText(img, "-- GREEN : SAFE", (H-100, 85),
    #             font, 0.5, (0, 255, 0), 1)
    # cv2.putText(img, "-- RED: UNSAFE", (H-200, 85),
    #             font, 0.5, (0, 0, 255), 1)



    tot_str = "Drowsy : " + str(drowsey_level)
    high_str = "Cigarette Detected : " + str(Cigarette)
    low_str = "Mobile Phone Detected : " + str(Mobile)
    safe_str = "Total Persons: " + str(0)


    Cigarette="False"
    Mobile="False"

    sub_img = frame[H - 100: H, 0:260]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

    res = cv2.addWeighted(sub_img, 0.8, black_rect, 0.2, 1.0)

    frame[H - 100:H, 0:260] = res

    cv2.putText(frame, tot_str, (10, H - 80),
                font, 0.5*2, (255, 255, 255), 1)
    cv2.putText(frame, high_str, (10, H - 55),
                font, 0.5*2, (0, 255, 0), 1)
    cv2.putText(frame, low_str, (10, H - 30),
                font, 0.5*2, (0, 120, 255), 1)
    cv2.putText(frame, safe_str, (10, H - 5),
                font, 0.5*2, (0, 0, 150), 1)

    ln=net.getLayerNames()


    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (224, 224), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # print("Frame Prediction Time : {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)










    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[classIDs[i]])
            color = (0,0,255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)
            print(label)
            if str(label)==str("Cigarette"):
                Cigarette="True"
            if str(label)==str("Mobile"):
                Mobile="True"





    if Cigarette == "False"  and Mobile=="False" and drowsey_level=="False":
        #image = draw_outputs(img, (boxes, scores, classes, nums), class_names, color=(0, 255, 0))
        cv2.circle(frame, (25, 80), 10, (0, 255, 0), -1)
        cv2.putText(frame, "All Ok", (50, 85),
                    font, 0.5*2, (0, 255, 0), 2)



    else:
        #image = draw_outputs(img, (boxes, scores, classes, nums), class_names, color=(0, 0, 255))
        cv2.circle(frame, (25, 80), 10, (0, 0, 255), -1)
        cv2.putText(frame, "Driver Rules Violation", (50, 85),
                    font, 0.5*2, (0, 0, 255), 2)


        telegx += 1
        print(telegx)
        if telegx > 2:
            cv2.imwrite("Fraud.jpg", frame)
            threading.Thread(target=telegram).start()
            telegx = 0

    drowsey_level = "False"

    #Show video feed
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Output", frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
