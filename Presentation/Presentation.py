import os
import cv2
from cvzone.HandTrackingModule import HandDetector

# slide window size
width, height = 1280, 720
ws, hs = int(256 * 1), int(144 * 1)  # small window size
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Initialize swipe gesture variables
gesture_threshold = 300
swipe_threshold = 100  # Minimum swipe distance
swipe_start = None

gesture_check = False
gesture_counter = 0
gesture_delay = 10

image_folder = "..\\Presentation\\Resources"  # Image folder
image_list = sorted(
    [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
image_index = 0

print(image_list)  # Print the list of Slides

detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Hand detection
    hands, img = detector.findHands(img, flipType=False)
    # cv2.line(img, (0, swipe_threshold), (width, swipe_threshold), (0,255,0), 10)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        # print(fingers)

        if fingers == [0, 0, 0, 0, 0]:
            gesture_check = False
            gesture_counter = 0

        if gesture_check is False:
            # Left Gesture
            if hand['type'] == 'Left' and fingers == [1, 0, 0, 0, 0]:
                if image_index > 0:
                    gesture_check = True
                    # image_index = (image_index - 1) % len(image_list)
                    image_index -= 1
                    # print("Left")
                    print("Slide No. : ", image_index)


            # Right Gesture
            if hand['type'] == 'Right' and fingers == [1, 0, 0, 0, 0]:
                if image_index < len(image_list) - 1:
                    gesture_check = True
                    # image_index = (image_index + 1) % len(image_list)
                    image_index += 1
                    # print("Right")
                    print("Slide No. : ", image_index)


    # gesture_check iterations
    if gesture_check:
        gesture_counter += 1
        if gesture_counter > gesture_delay:
            gesture_counter = 0
            gesture_check = False

    # Display the current image
    image_to_show = cv2.imread(image_list[image_index])

    # To resize the image/slide
    resizeSlide = cv2.resize(image_to_show, (width, height))

    # webcam
    smlWebcam = cv2.resize(img, (ws, hs))  # Adding webcam image on the slide
    h, w, _ = resizeSlide.shape
    resizeSlide[0:hs, w - ws:w] = smlWebcam  # Setting on the slide
    # cv2.imshow("Image", img)

    cv2.imshow("Slides", resizeSlide)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
