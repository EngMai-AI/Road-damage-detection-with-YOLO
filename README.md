# Road Damage Detection with YOLO

Detect and localize road damage in photos with YOLO object detection. Three YOLO models were trained and compared (YOLOv8n, YOLOv8s, YOLO11s), tuned, evaluated, and the best one was deployed as a **FastAPI** web app with an image-upload page and a REST API.


## Damage classes
`alligator`, `block`, `crack`, `edge`, `longitudinal`, `pothole`, `transverse`

## Dataset
[[Roboflow Universe: road-damage-l1ju7-ftmwi (version 1)](https://universe.roboflow.com/trafficsignssafeway/road-damage-l1ju7/browse?queryText=&pageSize=100&startingIndex=0&browseQuery=true)](https://universe.roboflow.com/), YOLOv8 format.

| Split | Images | Boxes |
|---|---|---|
| Train | 4,930 | 15,024 |
| Validation | 1,146 | 3,588 |
| Test | 681 | 2,004 |

35 label files contained polygons; they were converted to bounding boxes so all models see identical labels. The `edge` class has almost no validation or test examples.

## Results (test split)

All models: 100 epochs, image size 640, batch 16, AdamW (lr0 0.001), seed 0, pretrained COCO weights, Tesla T4.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8n | 0.605 | 0.565 | 0.580 | 0.357 |
| YOLOv8s | 0.613 | 0.539 | 0.590 | 0.361 |
| **YOLO11s** | **0.647** | 0.556 | **0.604** | **0.376** |
| YOLO11s (tuned) | 0.632 | 0.572 | 0.603 | 0.371 |
| YOLOv8n (tuned) | 0.602 | 0.556 | 0.576 | 0.356 |

### Speed, size and deployment efficiency

| Model | Params (M) | .pt (MB) | ONNX (MB) | GPU FPS (T4, batch 1) | CPU FPS (ONNX) |
|---|---|---|---|---|---|
| YOLOv8n | 3.14 | 6.5 | 12.8 | 128.5 | 15.1 |
| YOLOv8s | 11.14 | 22.5 | 44.8 | 86.1 | 5.4 |
| YOLO11s | 9.43 | 19.2 | 37.9 | 89.4 | 7.5 |

Speed and size figures come from notebook `04` (GPU on a Tesla T4, CPU via ONNX) and were not independently re-measured.

### Per-class AP50-95 (test)

| Class | YOLOv8n | YOLOv8s | YOLO11s |
|---|---|---|---|
| block | 0.598 | 0.593 | 0.642 |
| crack | 0.584 | 0.599 | 0.626 |
| pothole | 0.375 | 0.382 | 0.391 |
| transverse | 0.358 | 0.362 | 0.350 |
| alligator | 0.150 | 0.142 | 0.150 |
| longitudinal | 0.078 | 0.090 | 0.095 |

### Key findings
- **YOLO11s is the most accurate** and beats YOLOv8s on every metric while being smaller and faster.
- **YOLOv8n is the efficiency pick:** 3x fewer parameters, a 6.5 MB checkpoint and about 2x the CPU speed, for about 2 points less mAP50-95.
- **Hyperparameter tuning did not help at full length.** A 6-trial grid (30 epochs, YOLOv8n) picked `lr0=0.0005`, but the +0.005 gain was within noise and vanished at 100 epochs. Image size 800 was worse and 49% slower to train.
- **Alligator and longitudinal cracks are the hard classes.** Errors are mostly misses and extra boxes on background, not confusion between damage types.
- Differences between models are small and come from a single seed, so treat the ranking as suggestive.

## Deployment app (FastAPI)

```bash
cd api
pip install -r requirements.txt
uvicorn main:app
```

Open http://127.0.0.1:8000/ for the upload page, or http://127.0.0.1:8000/docs for the interactive API docs. Put your `.pt` weights in `api/models/` (the YOLO11s weights are the default).

| Endpoint | Description |
|---|---|
| `GET /health`, `GET /models` | Service status and available weights |
| `POST /predict` | JSON with class, confidence and box per detection, counts, inference time, optional base64 annotated image |
| `POST /predict/image` | Annotated JPEG only |

Query parameters: `model` (weight file name without `.pt`), `conf` (default 0.25). Uploads up to 10 MB.

```bash
curl -X POST "http://127.0.0.1:8000/predict?model=yolo11s_e100_best&conf=0.3" -F "file=@road.jpg"
```

Optional environment variables: `MODEL_DIR`, `DEFAULT_MODEL`, `DEFAULT_CONF`, `IMGSZ`.

![App screenshot](<img width="836" height="830" alt="image" src="https://github.com/user-attachments/assets/7d98f94f-f0a3-4d56-ba81-1c00bd94cfd1" />
<img width="886" height="873" alt="image" src="https://github.com/user-attachments/assets/140619da-1302-4724-b6b7-826feb61ac21" />
<img width="870" height="825" alt="image" src="https://github.com/user-attachments/assets/c9d6af20-8140-43c8-b3e3-6ad861c26e54" />
)

## Reproducing the experiments

The notebooks were run on Kaggle (Tesla T4):

| Notebook | Purpose |
|---|---|
| `01` / `02` / `03` | Train YOLOv8s / YOLOv8n / YOLO11s |
| `04` | Evaluation and final comparison |
| `05` | Manual hyperparameter tuning |
| `06` / `07` | Final training with the tuned settings (YOLO11s / YOLOv8n) |

**Roboflow API key:** do not put it in the code. On Kaggle, add it under *Add-ons → Secrets* as `ROBOFLOW_API_KEY` and read it with:

```python
from kaggle_secrets import UserSecretsClient
api_key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
```

## Repository structure
```
├── README.md
├── notebooks/        # 01-07
├── api/              # main.py, requirements.txt, static/, models/
├── report/           # comparative analysis (PDF)
└── screenshots/
```

## Limitations
- One training seed per model; gaps under about 0.02 mAP50-95 are not statistically established.
- Tuning was run on YOLOv8n only and its settings were reused for YOLO11s; YOLOv8s was not tuned.
- Training images are mostly asphalt roads and dashcam photos; accuracy drops on very different surfaces or heavy watermarks.
- The service handles single images; video and webcam streaming are not included.
- This is a demonstration project, not a replacement for a professional road inspection.

## Acknowledgements
Thanks to my instructor Jana Hatem, my mentor Sohaila ElSayed, and Instant Software Solutions.
