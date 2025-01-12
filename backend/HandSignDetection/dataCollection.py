import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)


detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300 # adjust to take larger hand size
    # can use try except if you want to avoid this...

folder = "Data/D Major Chord"
counter = 0

while True:
    success, img = cap.read()

    if not success:
        break
    img = cv2.flip(img,1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0] # crops around the first hand, relies on one hand (maxHands = 1)
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        imgCrop = img[y - offset: y + h+offset, x:x + w+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            wGap = math.ceil((imgSize - wCal) / 2) # gap to centre the image (width gap)

            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)  # gap to centre the image (width gap)

            imgWhite[hGap:hCal + hGap, : ] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("m"): # saving image, to train
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)



    #cv2.waitKey(1)

