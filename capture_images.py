import cv2
import os
import mediapipe as mp

label = input("Enter label for this sign (e.g., Hello): ")
save_dir = f"dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            xys = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            for x, y in xys:
                h, w = frame.shape[:2]
                cv2.circle(frame, (int(x * w), int(y * h)), 5, (255, 0, 0), -1)
            if frame_count % 5 == 0:
                cv2.imwrite(f"{save_dir}/{label}_{frame_count}.jpg", frame)
            frame_count += 1

    cv2.imshow("Capture - Press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
