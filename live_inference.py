import cv2
import time
import numpy as np
from ultralytics import YOLO

# base model
#model = YOLO(r'C:\Users\CHERIAN BIJU\Desktop\SafeGuard-CV\training_results\helmet_model\weights\best.pt', task='detect')

# edge model
model = YOLO(r'C:\Users\CHERIAN BIJU\Desktop\SafeGuard-CV\training_results\helmet_model\weights\best.onnx', task='detect')

video_path = 'test_video_1.mp4'
cap = cv2.VideoCapture(video_path)

COLORS = {
    'Helmet':    (255, 255, 255),
    'No-Helmet': (0, 0, 255),
    'Person':    (255, 255, 255)
}

# FPS tracking
fps_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-processing time
    t_pre = time.perf_counter()
    resized = cv2.resize(frame, (640, 640))
    t_pre_end = time.perf_counter()
    pre_ms = (t_pre_end - t_pre) * 1000

    # Inference time
    t_infer = time.perf_counter()
    results = model(resized, verbose=False, conf=0.6, device=0)
    t_infer_end = time.perf_counter()
    fps = 1 / (t_infer_end - t_infer)
    fps_list.append(fps)

     # Post-processing (NMS) time
    t_post = time.perf_counter()
    boxes = results[0].boxes
    names = [model.names[int(b.cls[0])] for b in boxes]  
    t_post_end = time.perf_counter()
    post_ms = (t_post_end - t_post) * 1000

    # Draw boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        color = COLORS.get(label, (255, 255, 255))

        cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(resized, f'{label} {conf:.2f}',(x1, y1 - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay metrics
    cv2.rectangle(resized, (0, 0), (280, 80), (0, 0, 0), -1)
    cv2.putText(resized, f'FPS: {fps:.1f}',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(resized, f'Preprocess: {pre_ms:.1f}ms',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(resized, f'Postprocess: {post_ms:.1f}ms',
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Helmet Detection - live_inference.py', resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# FPS statistics

print(f"FPS Statistics:")
print(f"Average FPS : {np.mean(fps_list):.1f}")
print(f"Min FPS     : {np.min(fps_list):.1f}")
print(f"Max FPS     : {np.max(fps_list):.1f}")
