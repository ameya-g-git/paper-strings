import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyglet
import time

cap = cv2.VideoCapture(0)

# Hand detector for max 2 hands
detector = HandDetector(maxHands=2)

# Initialize classifier
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Offset and image size settings
offset = 20
imgSize = 500

# Folder and counter (presumably for data collection)
folder = "Data/G Major Chord"
counter = 0

# Labels for chord types
labels = ["C Major Chord", "A Major Chord", "G Major Chord", "E Major Chord", "D Major Chord"]
audioPaths = ["audio/Cmaj.wav", "audio/Amaj.wav", "audio/Gmaj.wav", "audio/Emaj.wav", "audio/Dmaj.wav"]
audioObjects = []

for i in range (len(audioPaths)):
    audioObjects.append(pyglet.media.load(audioPaths[i], streaming=False))



# Initialize strumming variables
prev_right_y = 300
strumDifference = 20
predictedChordIndex = 0


# Play sound if strum detected
def playChord(predictedChordIndex):
    audioObjects[predictedChordIndex].play()
    time.sleep(1.8)
    print(labels[predictedChordIndex])

# Change index of labels to predicted chord
def setPredictedChordIndex(indexLeft):
    global predictedChordIndex
    predictedChordIndex = indexLeft



# Video feed loop
while True:
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # Flip image for mirror effect

    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hands (both if maxHands=2)

    if hands:
        left_hand = None
        right_hand = None

        # Separate left and right hands
        for hand in hands:
            if hand["type"] == "Left":
                right_hand = hand
            elif hand["type"] == "Right":
                left_hand = hand

        # Only process left hand for chord detection
        if left_hand:
            left_x, left_y, left_w, left_h = left_hand['bbox']
            imgWhiteLeft = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropLeft = img[left_y - offset: left_y + left_h + offset, left_x:left_x + left_w + offset]

            aspectRatioLeft = left_h / left_w

            if aspectRatioLeft > 1:
                k = imgSize / left_h
                wCalLeft = math.ceil(k * left_w)
                imgResizeLeft = cv2.resize(imgCropLeft, (wCalLeft, imgSize))
                wGapLeft = math.ceil((imgSize - wCalLeft) / 2)  # Gap to center the image (width gap)

                imgWhiteLeft[:, wGapLeft:wCalLeft + wGapLeft] = imgResizeLeft
            else:
                k = imgSize / left_w
                hCalLeft = math.ceil(k * left_h)
                imgResizeLeft = cv2.resize(imgCropLeft, (imgSize, hCalLeft))
                hGapLeft = math.ceil((imgSize - hCalLeft) / 2)  # Gap to center the image (height gap)

                imgWhiteLeft[hGapLeft:hCalLeft + hGapLeft, :] = imgResizeLeft

            # Get the prediction for chord detection for left hand
            predictionLeft, indexLeft = classifier.getPrediction(imgWhiteLeft, draw=False)
            cv2.putText(imgOutput, labels[indexLeft], (left_x, left_y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255),2)
            cv2.rectangle(imgOutput, (left_x - offset, left_y - offset),(left_x + left_w + offset, left_y + left_h + offset), (255, 0, 255), 4)
            setPredictedChordIndex(indexLeft)

        # Strum detection on right hand
        if right_hand:
            right_x, right_y, right_w, right_h = right_hand['bbox']
            # Draw the bounding box for the right hand
            cv2.rectangle(imgOutput, (right_x - offset, right_y - offset),(right_x + right_w + offset, right_y + right_h + offset), (0, 255, 255), 4)
            # print(right_y)
            # print(prev_right_y - right_y)
            if prev_right_y - right_y < -strumDifference:
                playChord(predictedChordIndex)
            prev_right_y = right_y

    # Show the output
    cv2.imshow('Paper Strings Demo - Copy 1', imgOutput)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()