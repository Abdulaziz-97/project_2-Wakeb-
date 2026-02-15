Here’s a concrete end-to-end plan to deliver “detect + mask + car-type” with your 20 aerial images: bootstrap masks/boxes using YOLO26 + SAM3, fine-tune a YOLO26 segmentation model, then add a per-car type classifier on top. [docs.ultralytics](https://docs.ultralytics.com/tasks/segment/)

## Outputs we’ll ship
- **Instance segmentation** model: returns a mask and a bounding box per car instance (and can be trained with multiple classes if/when you have enough labels). [docs.ultralytics](https://docs.ultralytics.com/tasks/segment/)
- **Car-type classification** model: takes each detected car crop (box crop or mask-cutout) and predicts one of your 4 car types.  

## Environment setup
1) Install/upgrade Ultralytics so SAM3 integration is available (SAM3 is integrated as of `ultralytics` 8.3.237). [docs.ultralytics](https://docs.ultralytics.com/models/yolo26/)
2) Request/download `sam3.pt` separately (Ultralytics notes SAM3 weights are not auto-downloaded), and place it in your project folder or reference it by full path. [docs.ultralytics](https://docs.ultralytics.com/models/yolo26/)

## Data + annotation plan
1) Create a folder of raw images, e.g. `data/raw_images/` (optionally tile into patches first if cars are tiny; keep overlaps so you don’t cut cars).  
2) Run **auto-annotation** to generate initial segmentation labels: `auto_annotate()` detects with a YOLO model then runs a SAM model to produce segments saved as YOLO-format text labels. [openreview](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf)
3) Use SAM3 weights inside auto-annotation by setting `sam_model="sam3.pt"` (auto_annotate instantiates `SAM(sam_model)` and then calls it with `bboxes`). [openreview](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf)

Example bootstrap script:
```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data="data/raw_images",
    det_model="yolo26x.pt",
    sam_model="sam3.pt",
    conf=0.25,
    imgsz=1024,   # raise if aerial cars are small
    device="0",
)
```
`auto_annotate()` supports `classes=[...]` if you want to filter detections to certain class IDs from the detector. [openreview](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf)

## Training plan (2-stage, reliable)
1) **Stage A: Segment “car” (1 class)** — train a YOLO26 segmentation model on the bootstrapped labels first, because you’ll get far more stable masks/boxes before you try fine-grained types. [github](https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/segment/index.md)
2) **Stage B: Classify types on crops** — for each predicted instance, crop (box crop or mask-cutout) and train a small classifier to predict the 4 car types; this isolates the “fine-grained” problem from localization noise.  
3) When you have more labeled instances, optionally collapse into **one 4-class segmentation model** (each instance mask labeled by type), using the YOLO segmentation polygon label format and a dataset YAML that lists your class names. [github](https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/segment/index.md)

Minimal YOLO-seg training (syntax from Ultralytics segment task docs):
```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")  # pretrained recommended
model.train(data="your_dataset-seg.yaml", epochs=100, imgsz=1024)
```


## Iteration loop (how you win with 20 images)
1) After the first training, run inference on your 20 images (and any extra unlabeled aerial frames you can collect), then **select the worst predictions** for manual correction and add them back into training.  
2) Re-run `auto_annotate()` occasionally with a stronger `det_model` or updated thresholds to keep generating better pseudo-labels (it’s explicitly designed as YOLO-detect → SAM-segment → save labels). [openreview](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf)
3) Only start labeling car types heavily once your segmentation is stable, because type labels on bad crops waste time.

If you tell me (a) your 4 car types and (b) average car size in pixels in the aerial images, I’ll pick the best `imgsz`, tiling size/stride, and whether mask-cutout crops or plain box crops will work better for the classifier.
