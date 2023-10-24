import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import pygame

# Load the gesture model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize Mediapipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize pygame for audio playback
pygame.init()
pygame.mixer.init()

# Define a dictionary that maps gestures to song file paths
gesture_to_song = {
    "hello": "songs/Brown_munde.mp3",
    "srk": "songs/ddlj.mp3",  
}
song_playing = False

while True:
    lst = []

    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        # Play songs based on the detected gesture
        if pred in gesture_to_song:
            song_path = gesture_to_song[pred]
            if not song_playing:
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()
                song_playing = True
        else:
            if song_playing:
                pygame.mixer.music.stop()
                song_playing = False

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        pygame.mixer.music.stop()  # Stop the song before exiting
        pygame.quit()  # Quit pygame
        break
