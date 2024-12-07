import cv2
import numpy as np
import my_modules.HandTrackingModule as htm
import time
import pyautogui

wCam, hCam = 640, 480
frameR = 100
smoothening = 7
flag = True
cur_time = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
detector = htm.HandDetector(max_hands=1)
wScr, hScr = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

print("Starting...")

if not cap.isOpened():
    flag = False
    print("Something went wrong!!! \n\tPlease check your camera")


while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Something went wrong!!!")
        break

    img = detector.find_hands(img)
    lmList, bbox = detector.find_position(img)

    # tip of the fingers
    if len(lmList) != 0:
        x1, y1 = lmList[4][1:]
        x2, y2 = lmList[8][1:]
        x3, y3 = lmList[12][1:]

    # which finger is up
    fingers = detector.finger_up()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # Only Index Finger : Moving Mode
    if len(fingers) != 0 and fingers[1] == 1 and fingers[2] == 0:
        # mapping screen size with index finger position
        mx = np.interp(x2, (frameR, wCam - frameR), (0, wScr))
        my = np.interp(y2, (frameR, hCam - frameR), (0, hScr))
        clocX = plocX + (mx - plocX) / smoothening
        clocY = plocY + (my - plocY) / smoothening

        pyautogui.moveTo(wScr - clocX, clocY)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # Left Click
    if len(fingers) != 0 and fingers[0] == 1 and fingers[1] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.find_distance(4, 8, img, r=10)
        print(length)
        # 10. Click mouse if distance short
        if length < 80:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
            pyautogui.click()

    # Right Click
    if len(fingers) != 0 and fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.find_distance(8, 12, img, r=10)
        print(length)
        # 10. Click mouse if distance short
        if length < 20:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
            pyautogui.rightClick()

    # Frame rate
    cur_time = detector.frames_per_second(img, cur_time)
    # display
    cv2.imshow("Image", img)

    if flag:
        print("Now you can control cursor with you finger....")
        flag = False

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Program Terminated!!!")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
