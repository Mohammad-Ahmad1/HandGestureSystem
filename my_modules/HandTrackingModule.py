import math

import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, comp=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.comp = comp
        self.detectionCon = detection_con
        self.trackCon = track_con

        # Initialize hand tracking module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.comp, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    # set landmarks on hands
    def find_hands(self, img, draw=True):
        # to convert the BGR image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # find position of the landmarks
    def find_position(self, img, hand_no=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for lm_id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # calculating center of the landmark
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([lm_id, cx, cy])
                if draw:
                    # to make a circle on the landmark
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    # find which finger is up
    def finger_up(self):
        fingers = []
        # Thumb
        if len(self.lmList) != 0:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for lm_id in range(1, 5):
                if self.lmList[self.tipIds[lm_id]][2] < self.lmList[self.tipIds[lm_id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        # totalFingers = finger.count(1)
        return fingers

    # find distance between landmarks
    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def frames_per_second(self, img, prev_time=0):
        # To calculate the frames per second
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        # Displaying frames per sec
        cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 1)
        return cur_time


def main():
    # Set up video capture
    cap = cv2.VideoCapture(0)

    detector = HandDetector()
    cur_time = 0

    while cap.isOpened():
        success, img = cap.read()

        # to flip the frames horizontally
        img = cv2.flip(img, 1)

        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        if len(lmList) != 0:
            print(lmList[4])

        cur_time = detector.frames_per_second(img, cur_time)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
