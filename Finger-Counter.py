import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
# print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (200, 200))
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

detector = htm.HandDetector(detectionCon=0.8)


tipIds = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            # Fingers
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        img[0:200, 0:200] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 20)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS: {int(fps)}',(400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2 )




    cv2.imshow('Video', img)
    cv2.waitKey(1)
