pipeline.pyimport cv2
import torch
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.eval()

np.random.seed(42)
COLORS = np.random.randint(0, 255, (150, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = outputs.logits.argmax(dim=1)[0].numpy()
    seg_map_resized = cv2.resize(
        seg_map.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    color_seg = COLORS[seg_map_resized]
    color_seg_bgr = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(frame, 0.5, color_seg_bgr, 0.5, 0)

    cv2.imshow("Segmentation", blended)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()