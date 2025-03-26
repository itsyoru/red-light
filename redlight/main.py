import cv2
import torch
import numpy as np
import time
import math
from collections import defaultdict, deque

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

game_state = "Green Light"
last_switch = time.time()
switch_interval = 5
game_over = False

position_history = defaultdict(lambda: deque(maxlen=2))
movement_threshold = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if game_over:
        cv2.putText(frame, "YOU LOST! Press 'R' to restart", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Red Light, Green Light", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            game_over = False
            game_state = "Green Light"
            last_switch = time.time()
            position_history.clear()
        elif key == ord('q'):
            break
        continue
    
    results = model(frame)

    if time.time() - last_switch > switch_interval:
        game_state = "Red Light" if game_state == "Green Light" else "Green Light"
        last_switch = time.time()

    person_detected = False
    for i, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
        if results.names[int(cls)] == 'person':
            person_detected = True
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            color = (0, 255, 0) if game_state == "Green Light" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, color, -1)

            position_history[i].append((center_x, center_y))

            if game_state == "Red Light" and len(position_history[i]) == 2:
                prev_x, prev_y = position_history[i][0]
                distance = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

                if distance > movement_threshold:
                    print(f"Movement detected: {distance:.2f} pixels")
                    game_over = True

    if not person_detected:
        position_history.clear()

    state_color = (0, 255, 0) if game_state == "Green Light" else (0, 0, 255)
    cv2.putText(frame, f"State: {game_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)

    cv2.imshow("Red Light, Green Light", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
