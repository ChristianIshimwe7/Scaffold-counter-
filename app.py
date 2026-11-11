"""Scaffold plank counter: web UI (Gradio).

Run locally:   python app.py
Colab share:   SHARE=1 python app.py
Needs a trained YOLO weights file named best.pt next to this file (or set WEIGHTS=/path/to/file.pt).
"""
import os
import time

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

WEIGHTS = os.environ.get("WEIGHTS", "best.pt")
IMGSZ = int(os.environ.get("IMGSZ", "512"))
IOU = 0.5
MAX_SIDE = 1600  # resize very large photos for speed

model = YOLO(WEIGHTS)
model.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, verbose=False)  # warm-up


def _prep(image):
    """Gradio gives RGB; return BGR, shrunk if huge."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def _roi_px(shape, x0, y0, x1, y1):
    h, w = shape[:2]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return int(w * x0 / 100), int(h * y0 / 100), int(w * x1 / 100), int(h * y1 / 100)


def preview(image, x0, y0, x1, y1):
    """Show the selected region (yellow) before counting."""
    if image is None:
        return None
    bgr = _prep(image)
    a, b, c, d = _roi_px(bgr.shape, x0, y0, x1, y1)
    cv2.rectangle(bgr, (a, b), (c, d), (0, 200, 255), 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def count(image, conf, aspect_max, x0, y0, x1, y1):
    if image is None:
        return None, "Upload a photo first."
    bgr = _prep(image)
    h, w = bgr.shape[:2]
    a, b, c, d = _roi_px(bgr.shape, x0, y0, x1, y1)
    t = time.perf_counter()
    r = model.predict(bgr, conf=conf, iou=IOU, imgsz=IMGSZ, verbose=False)[0]
    ms = (time.perf_counter() - t) * 1000
    keep = []
    for (cx, cy, bw, bh), box in zip(r.boxes.xywh.tolist(), r.boxes.xyxy.tolist()):
        if a <= cx <= c and b <= cy <= d and bw / bh <= aspect_max:
            keep.append((box, cy))
    keep.sort(key=lambda k: (round(k[0][0] / 40), k[1]))
    out = bgr.copy()
    cv2.rectangle(out, (a, b), (c, d), (0, 200, 255), 2)
    for i, (box, _) in enumerate(keep, 1):
        col = (60, 60, 230) if i % 2 else (230, 120, 40)
        cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 2)
    cv2.putText(out, f"{len(keep)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
    msg = (f"## {len(keep)} planks counted\n"
           f"Detections in the whole image: {len(r.boxes)} (only those inside the yellow box are counted)  \n"
           f"Inference time: {ms:.0f} ms")
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), msg


with gr.Blocks(title="Scaffold plank counter") as demo:
    gr.Markdown(
        "# Scaffold plank counter\n"
        "1. Upload a photo of stacked scaffold planks (front view works best).\n"
        "2. Move the box sliders so the **yellow box** covers only the stack you want to count.\n"
        "3. Press **Count**. Lower *Confidence* to find more planks, raise it to remove false ones.\n\n"
        "_Baseline model trained on synthetic images: check the result and the red/blue boxes before you trust a count._")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Photo", image_mode="RGB")
            conf = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confidence")
            asp = gr.Slider(2, 12, value=6, step=0.5, label="Ignore boxes wider than this x their height (removes side faces)")
            gr.Markdown("**Region to count (% of the image)**")
            x0 = gr.Slider(0, 100, value=0, step=1, label="Left")
            x1 = gr.Slider(0, 100, value=100, step=1, label="Right")
            y0 = gr.Slider(0, 100, value=0, step=1, label="Top")
            y1 = gr.Slider(0, 100, value=100, step=1, label="Bottom")
            btn = gr.Button("Count", variant="primary")
        with gr.Column():
            out = gr.Image(label="Result", interactive=False)
            txt = gr.Markdown()

    roi_inputs = [inp, x0, y0, x1, y1]
    for comp in (x0, x1, y0, y1):
        comp.change(preview, roi_inputs, out, show_progress="hidden")
    inp.change(preview, roi_inputs, out, show_progress="hidden")
    btn.click(count, [inp, conf, asp, x0, y0, x1, y1], [out, txt], api_name="count")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=os.environ.get("SHARE") == "1")
