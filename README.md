# Scaffold plank counter

Counts stacked scaffold planks (and round tube bundles) in a photo and draws numbered boxes.
Baseline built for a test task: a YOLOv8n detector trained on **synthetic images**.
## Demo
A temporary live demo (runs from a Colab notebook): https://94e06bbd94323fad08.gradio.live

Upload a photo, move the Left/Right/Top/Bottom sliders so the yellow box covers one stack, press **Count**.
Do not upload confidential images to a shared link.

## Contents
| Path | What |
|---|---|
| `notebooks/colab_scaffold_counter.ipynb` | Full pipeline on Colab (T4): training(on synthetic data), counting, any-image counting, export |
| `src/synth_data.py` | Synthetic stack/bundle generator with YOLO labels |
| `src/count_in_roi.py` | Count detections inside a region (JSON of boxes per image) |
| `src/predict_count.py` | Plain detection + count for a folder |
| `src/count_tubes.py` | Classical (Hough + lattice) counter for round tubes, no training |
| `src/count_planks.py` | Experimental classical plank-layer counter (unreliable, kept for the record) |
| `app/app.py` | Gradio web UI (upload, choose the stack with sliders, Count)

## Quick start
```bash
pip install -r requirements.txt
python src/synth_data.py --out data/synthetic --train 1000 --val 150 --size 512 --seed 1
yolo detect train data=data/synthetic/dataset.yaml model=yolov8n.pt imgsz=512 epochs=30 batch=32 name=synth
cp runs/detect/synth/weights/best.pt models/best.pt        # path may be runs/detect/runs/synth/... on some versions
python app/app.py                                          # opens the UI at http://localhost:7860
```
GPU training is much faster; the Colab notebook does all of this in one go.

## Export for iOS
```bash
pip install coremltools onnx
yolo export model=models/best.pt format=coreml nms=True imgsz=512   # -> best.mlpackage
yolo export model=models/best.pt format=onnx imgsz=512
```
The Core ML model takes a 512x512 image plus `confidenceThreshold` and `iouThreshold` and returns `confidence`
and `coordinates`. Keep only boxes inside the user's region and drop boxes wider than ~6x their height in app code.

## Limitations
- Trained on synthetic data only; real-photo accuracy needs annotated real images (CVAT, YOLO format) and fine-tuning.
- The region (ROI) is essential: counting the whole image also counts neighbouring stacks and background planks.
- Thresholds (confidence 0.5, IoU 0.5, aspect 6) were tuned by eye, not on a held-out real set.
- The client's test photos are confidential and are intentionally not in this repository.
