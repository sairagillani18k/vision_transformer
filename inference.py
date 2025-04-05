import cv2
import torch
import numpy as np
from models.vit_model import GestureViT
from utils.mediapipe_utils import extract_hand_landmarks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureViT(num_classes=2).to(device)
model.load_state_dict(torch.load("gesture_model.pth", map_location=device))
model.eval()

cap = cv2.VideoCapture(0)
gesture_names = ["Fist", "Palm"]  # Adjust as per dataset

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_hand_landmarks(frame)
    if landmarks is not None:
        with torch.no_grad():
            input_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            gesture = gesture_names[pred]

            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
