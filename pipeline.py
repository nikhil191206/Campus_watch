import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

yolo = YOLO("yolov8n.pt")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_model.eval()

np.random.seed(42)
COLORS = np.random.randint(0, 255, (150, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)
frame_count = 0
seg_overlay = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 10 == 0 or seg_overlay is None:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = seg_model(**inputs)
        seg_map = outputs.logits.argmax(dim=1)[0].numpy()
        seg_map_resized = cv2.resize(
            seg_map.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        color_seg = COLORS[seg_map_resized]
        seg_overlay = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)

    results = yolo(frame, verbose=False)
    yolo_frame = results[0].plot()

    combined = cv2.addWeighted(yolo_frame, 0.6, seg_overlay, 0.4, 0)

    detected = [results[0].names[int(c)] for c in results[0].boxes.cls] if results[0].boxes else []
    person_present = "person" in detected
    status = "ALERT - Person Detected" if person_present else "All Clear"
    color = (0, 0, 255) if person_present else (0, 255, 0)
    cv2.putText(combined, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("VIT CampusWatch", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()