import cv2
import numpy as np
import time
from my_modules import HandTrackingModule as htm
from cvzone.HandTrackingModule import HandDetector
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Size of the window
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.HandDetector(detection_con=0.7)
detector2 = HandDetector(detectionCon=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while cap.isOpened():
    success, img = cap.read()

    # to flip the frames horizontally
    img = cv2.flip(img, 1)
    # img = detector.find_hands(img)

    hands, img2 = detector2.findHands(img)
    # hands = True
    if hands:
        # Get the first hand detected
        hand = hands[0]
        # lmList = detector.find_position(img2)

        # Get landmark positions
        lmList = hand['lmList']

        if len(lmList) != 0:
            # print(lmList[4], lmList[8])

            # Get the tip positions of the thumb (id=4) and index finger (id=8)
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            x1, y1 = thumb_tip[:2]
            x2, y2 = index_tip[:2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (x1, y1), 5, (0, 0, 125), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 125), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 125), 3)
            cv2.circle(img, (cx, cy), 5, (0, 0, 125), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)
            # Finger distance min-20 & max-250
            # Volume range -65 to 0
            # to map with length
            vol = np.interp(length, [20, 250], [minVol, maxVol])
            # print(int(length), vol)
            volume.SetMasterVolumeLevel(vol, None)
            # Set System Volume
            # volume.SetMasterVolumeLevelScalar(vol, None)

    # To calculate the frames per second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
