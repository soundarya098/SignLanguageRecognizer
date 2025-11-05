import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    return np.zeros(63)  # No hand detected
