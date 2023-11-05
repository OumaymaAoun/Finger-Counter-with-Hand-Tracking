import cv2
import time
import os
import mediapipe as mp

# Set webcam dimensions
wCam, hCam = 640, 480

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # Set webcam width
cap.set(4, hCam)  # Set webcam height

# Path to the folder containing overlay images
folderPath = 'FingerImages'
myList = os.listdir(folderPath)
print(myList)
overlaylist = []

# Load overlay images
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlaylist.append(image)
print(len(overlaylist))

pTime = 0

# Initialize the mediapipe hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the drawing module for visualization
mp_drawing = mp.solutions.drawing_utils

# List to store hand landmark positions
hand_positions = []
tipIds = [4 , 8 , 12 , 16 , 20]
while True:
    success, img = cap.read()

    # Convert the BGR frame to RGB for processing with mediapipe
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If hands are detected, draw landmarks on the frame and store positions
    if results.multi_hand_landmarks:
        hand_positions = []
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

            # Store landmark positions in the hand_positions list as rounded integers
            for landmark in landmarks.landmark:
                hand_positions.append([
                    int(landmark.x * img.shape[1]),  # Convert normalized coordinates to pixel values
                    int(landmark.y * img.shape[0]),  # Convert normalized coordinates to pixel values
                    int(landmark.z * img.shape[1])  # Convert normalized coordinates to pixel values
                ])
                #print(hand_positions)
         #use the right hand
        if len(hand_positions) != 0:
            fingers =[]
            #thumb ken bech tbadel left hand rodha akber y_tip > y_base - finger_open_threshold
            y_tip = hand_positions[tipIds[0]][1]
            y_base = hand_positions[tipIds[0] - 1][1]

            # Define a threshold for finger open/close detection
            finger_open_threshold = 20

            if y_tip < y_base - finger_open_threshold:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range (1,5):
                # Assuming index finger's tip is at index 8 and base is at index 5
                y_tip = hand_positions[tipIds[id]][1]
                y_base = hand_positions[tipIds[id]-3][1]

                # Define a threshold for finger open/close detection
                finger_open_threshold = 20

                if y_tip < y_base - finger_open_threshold:
                    fingers.append(1)
                else :
                    fingers.append(0)
            #print(fingers)
            TotalFingers = fingers.count(1)
            print(TotalFingers)

            target_width = 200  # Replace with your target width
            target_height = 200  # Replace with your target height

            overlay_img_resized = cv2.resize(overlaylist[TotalFingers-1], (target_width, target_height))
            img[0:target_height, 0:target_width] = overlay_img_resized
            cv2.rectangle(img, (20,225),(170,425) ,(0,255,0), cv2.FILLED)
            cv2.putText(img , str(TotalFingers), (45,375),cv2.FONT_HERSHEY_PLAIN ,10 , (255,0,0) , 25)

    # Calculate frames per second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Hand Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

