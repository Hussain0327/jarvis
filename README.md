# Project Jarvis: Intelligent Traffic Intelligence Platform

## Principal-Level Technical Design Document

**Version:** 1.1
**Author:** Hussain
**Date:** February 2026
**Status:** Phase 2 Complete — Tracking System Implemented

---

## 1. Executive Summary

Project Jarvis is a full-stack AI engineering and data science platform that ingests live traffic camera feeds, detects and identifies vehicles, tracks them across multiple cameras, predicts their trajectories, detects behavioral anomalies, and exposes all intelligence through both a visual command dashboard and a natural language analyst interface powered by a fine-tuned open-source LLM with RAG.

The system chains together six distinct technical layers, each representing a different ML/AI engineering competency: real-time video ingestion, deep learning perception (detection + OCR), multi-object tracking with cross-camera identity resolution, classical ML analytics, a full-stack web dashboard, and an LLM-powered intelligence interface with retrieval-augmented generation.

---

## 1.1 Progress

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1: Ingestion + Perception | **Complete** | StreamHandler, FrameBuffer, CameraManager, YOLOv8 vehicle/plate detection, TrOCR OCR, PerceptionPipeline |
| Phase 2: Tracking | **Complete** | SingleCameraTracker (BoxMOT), ReIDEmbedder (DINOv2/OSNet), CrossCameraResolver, IdentityGraph |
| Codebase Audit | **Complete** | 31 issues fixed: thread safety (4), security (3), config (3), reliability (6), performance (7), test infra (3), test coverage (5) |
| Phase 3: Database Pipeline | Not started | Full schema deployed, query functions exist; pipeline orchestration needed |
| Phase 4: Analytics Engine | Not started | Route clustering, anomaly detection, trajectory prediction |
| Phase 5: Dashboard | Not started | FastAPI + Streamlit |
| Phase 6: LLM Intelligence | Not started | Qwen3 + RAG + function calling |
| Phase 7: Integration | Not started | Docker deployment + hardening |

**Test suite:** 108 passing, 4 skipped (DB integration), lint clean (`ruff check src/ tests/`)

**Key audit fixes applied:**
- Thread-safe singletons (double-checked locking) in DB session and camera manager
- PostgreSQL upserts (`INSERT ON CONFLICT`) replacing race-prone SELECT-then-INSERT
- RTSP credential sanitization in logs
- Hardcoded passwords removed from tracked files
- YAML config sub-settings now actually load
- `get_settings()` is a true `@lru_cache` singleton
- `sqlalchemy.engine.URL.create()` for safe URL encoding
- Stable class IDs for BoxMOT tracker association
- Batch inference for YOLO/TrOCR (true GPU batching, not sequential loops)
- `grab()`/`retrieve()` frame skipping optimization
- HNSW vector index migration for pgvector similarity search

---

## 2. System Architecture

### 2.1 Layer Overview

```
Layer 1: Ingestion         -> Public traffic camera feeds, frame standardization
Layer 2: Perception        -> Vehicle detection, plate detection, plate OCR
Layer 3: Tracking          -> Single-camera MOT, cross-camera re-identification
Layer 4: Analytics          -> Route clustering, anomaly detection, trajectory prediction
Layer 5: Dashboard          -> Real-time visualization, map interface, alert system
Layer 6: Intelligence       -> RAG + function calling LLM analyst interface
```

### 2.2 Data Flow

```
Traffic Camera Feeds (RTSP/MJPEG)
        |
   [OpenCV Frame Extraction]
        |
   Frame Buffer (timestamped, camera-tagged)
        |
   [YOLOv8 Vehicle Detection] -----> Detection Logs (PostgreSQL)
        |
   [YOLOv8 Plate Detection]
        |
   [TrOCR Plate Recognition] ------> Plate Read Logs (PostgreSQL)
        |
   [BoT-SORT/ByteTrack Tracking] ---> Track Logs (PostgreSQL)
        |
   [DINOv2 Cross-Camera Re-ID] ----> Identity Graph (PostgreSQL)
        |
   [scikit-learn Analytics] --------> Anomaly Flags, Route Clusters (PostgreSQL)
        |
   [LSTM/Transformer Prediction] --> Trajectory Predictions (PostgreSQL)
        |
   [FastAPI Backend] <--------------> [React/Streamlit Dashboard]
        |
   [Qwen3-30B-A3B LLM + RAG] <----> [pgvector + Redis Cache]
        |
   Natural Language Intelligence Interface
```

### 2.3 Complete Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Primary development language |
| Deep Learning Framework | PyTorch 2.x | YOLOv8, TrOCR, tracking, LSTM, DINOv2 |
| Classical ML | scikit-learn | Clustering, anomaly detection, baselines |
| Computer Vision | OpenCV | Frame extraction, preprocessing, stream handling |
| Object Detection | Ultralytics YOLOv8 | Vehicle and plate detection |
| OCR | TrOCR (microsoft/trocr-base-printed) | License plate character recognition |
| Tracking | BoT-SORT / ByteTrack (via BoxMOT) | Multi-object tracking |
| Re-identification | OSNet (via BoxMOT) / DINOv2 (facebook/dinov2-base) | Cross-camera vehicle embedding matching |
| Sequence Modeling | PyTorch LSTM or Transformer | Trajectory prediction |
| LLM | Qwen3-30B-A3B (Apache 2.0) | Natural language analyst interface |
| LLM Serving | Ollama (local) / vLLM (cloud) | Model inference |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Document vectorization for RAG |
| Vector Store | pgvector (PostgreSQL extension) | Similarity search for RAG retrieval |
| Database | PostgreSQL 16 | Structured data storage |
| Cache | Redis | Query result and embedding caching |
| Backend API | FastAPI | REST API serving predictions and queries |
| Frontend | React or Streamlit | Dashboard interface |
| Maps | Folium / Mapbox / Leaflet | Geospatial visualization |
| Training Platform | Google Colab Pro (A100 GPU) | Model training and fine-tuning |
| Fine-tuning | QLoRA via Hugging Face PEFT / Unsloth | LLM domain adaptation |
| Containerization | Docker / Docker Compose | Service orchestration |
| Version Control | Git / GitHub | Code management |

---

## 3. Datasets

### 3.1 Vehicle Detection Training

**UA-DETRAC**
- Source: https://detrac-db.rit.albany.edu/
- Content: 100+ hours of traffic video from Beijing and Tianjin
- Annotations: Bounding boxes for vehicles with classification (car, bus, van, others)
- Size: ~140,000 frames, 1.21 million labeled objects
- Use: Supplementary training volume for YOLOv8 vehicle detection
- Format: PASCAL VOC XML annotations
- **Note:** Recorded in China — vehicle types, lane behavior, and road markings differ significantly from US traffic. Use as supplementary data, not primary.

**CityFlow**
- Source: https://www.aicitychallenge.org/
- Content: US intersection footage from multiple cities, designed for traffic analysis
- Annotations: Vehicle bounding boxes, tracking IDs, re-ID labels across cameras
- Use: Primary training and evaluation dataset for US traffic scenarios (detection + tracking + re-ID)
- Advantage: Matches target deployment domain (US intersections with US vehicle types)

**BDD100K**
- Source: https://bdd-data.berkeley.edu/
- Content: 100,000 driving video clips from Berkeley
- Annotations: Bounding boxes, lane markings, drivable areas, instance segmentation
- Size: 100K videos, 10 frames each with labels
- Use: Supplementary training and validation for vehicle detection
- Format: JSON annotations

**COCO (vehicles subset)**
- Source: https://cocodataset.org/
- Content: General object detection dataset
- Use: Pre-training baseline, transfer learning starting point
- Filter: car, truck, bus, motorcycle categories only

### 3.2 License Plate Detection Training

**UFPR-ALPR**
- Source: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
- Content: 4,500 fully annotated images of vehicle rear ends
- Annotations: Plate bounding boxes, character-level annotations
- Use: Fine-tuning YOLOv8 for plate region detection

**CCPD (Chinese City Parking Dataset)**
- Source: https://github.com/detectRecog/CCPD
- Content: 250,000+ images with plate annotations
- Use: Additional volume for plate detection training
- Note: Chinese plates; useful for detection model generalization, not OCR

**OpenALPR Benchmarks**
- Source: https://github.com/openalpr/benchmarks
- Content: Plate images from multiple countries
- Use: Evaluation and testing of plate detection pipeline

### 3.3 License Plate OCR Training

**UFPR-ALPR (character level)**
- Same dataset as above
- Use: Character-level annotations for fine-tuning TrOCR

**Synthetic Plate Data (self-generated)**
- Method: Programmatically generate synthetic US license plate images with known text
- Libraries: PIL/Pillow, albumentations for augmentation
- Augmentations: rotation, blur, noise, lighting variation, perspective warping
- Volume: Generate 50,000-100,000 synthetic plate crops
- Use: Augment real plate data for TrOCR fine-tuning

### 3.4 Vehicle Re-identification

**VeRi-776**
- Source: https://vehiclereid.github.io/VeRi/
- Content: 776 vehicles captured by 20 cameras
- Annotations: Vehicle identity labels across cameras
- Use: Evaluating DINOv2/OSNet embedding quality for cross-camera matching

**CityFlow (Re-ID subset)**
- Source: https://www.aicitychallenge.org/
- Content: US intersection footage with vehicle re-ID annotations across cameras
- Use: Evaluating re-ID performance on US traffic (complementary to VeRi-776)

**VehicleID**
- Source: https://www.pkuml.org/resources/pku-vehicleid.html
- Content: 221,763 images of 26,267 vehicles
- Use: Large-scale vehicle re-identification training and evaluation

### 3.5 Trajectory and Traffic Flow

**NGSIM (Next Generation Simulation)**
- Source: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm
- Content: Detailed vehicle trajectory data from US highways
- Annotations: Position, velocity, acceleration at 10 Hz
- Use: Supplementary training data for LSTM/Transformer trajectory prediction model
- **Note:** Collected in 2005-2006 — traffic patterns and vehicle types have changed significantly. Prioritize Argoverse 2 as primary trajectory training data.

**Argoverse 2 Motion Forecasting**
- Source: https://www.argoverse.org/av2.html
- Content: 250,000 driving scenarios with HD maps
- Use: Trajectory prediction model training and benchmarking

### 3.6 Live Inference Data

**NYC DOT Traffic Cameras**
- Source: https://webcams.nyctmc.org/
- Content: 700+ live traffic camera feeds across New York City
- Format: MJPEG streams, publicly accessible
- Use: Live demo inference, real-time system testing

**NJ DOT Traffic Cameras**
- Source: https://www.511nj.org/
- Content: Live traffic camera feeds across New Jersey
- Use: Local demo data, New Jersey plates for OCR testing

**Caltrans CCTV**
- Source: https://cwwp2.dot.ca.gov/vm/iframemap.htm
- Content: California DOT traffic cameras
- Use: Additional live feed source for system testing

---

## 4. Phase Breakdown

---

### Phase 1: Perception Pipeline (Weeks 1-4)

**Objective:** Build a working pipeline that takes a traffic image or video frame and outputs detected vehicles with classified types and read license plate numbers.

#### 1.1 Environment Setup (Week 1, Days 1-2)

**Tasks:**
- Set up Python 3.11 virtual environment
- Install core dependencies: PyTorch, OpenCV, Ultralytics, transformers
- Configure Google Colab Pro notebook for GPU training
- Initialize Git repository with proper .gitignore
- Create project directory structure

**Directory Structure:**
```
project-jarvis/
├── config/
│   ├── config.yaml              # Global configuration
│   ├── camera_registry.yaml     # Camera metadata (location, URL, FOV)
│   └── model_configs/           # Per-model hyperparameters
├── data/
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Preprocessed training data
│   ├── synthetic/               # Generated synthetic plates
│   └── splits/                  # Train/val/test splits
├── models/
│   ├── vehicle_detector/        # YOLOv8 vehicle detection
│   ├── plate_detector/          # YOLOv8 plate detection
│   ├── plate_ocr/               # TrOCR plate reading
│   ├── tracker/                 # BoT-SORT/ByteTrack weights
│   ├── reid/                    # DINOv2 re-ID embeddings
│   ├── trajectory/              # LSTM/Transformer prediction
│   └── llm/                     # Fine-tuned Qwen3 adapter weights
├── src/
│   ├── ingestion/
│   │   ├── stream_handler.py    # Camera feed connection and frame extraction
│   │   ├── frame_buffer.py      # Timestamped frame queue management
│   │   └── camera_manager.py    # Multi-camera orchestration
│   ├── perception/
│   │   ├── vehicle_detector.py  # YOLOv8 vehicle detection wrapper
│   │   ├── plate_detector.py    # YOLOv8 plate detection wrapper
│   │   ├── plate_ocr.py         # TrOCR OCR pipeline
│   │   └── perception_pipeline.py  # End-to-end frame processing
│   ├── tracking/
│   │   ├── single_camera.py     # BoT-SORT/ByteTrack per-camera tracking
│   │   ├── reid_embedder.py     # DINOv2 embedding extraction
│   │   ├── cross_camera.py      # Cross-camera identity resolution
│   │   └── identity_graph.py    # Vehicle identity management
│   ├── analytics/
│   │   ├── route_clustering.py  # K-means/DBSCAN on trajectories
│   │   ├── anomaly_detector.py  # Isolation Forest anomaly detection
│   │   ├── trajectory_predictor.py  # LSTM/Transformer prediction
│   │   └── behavioral_profiler.py   # Per-vehicle behavioral analysis
│   ├── database/
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   ├── queries.py           # Prebuilt query functions
│   │   └── migrations/          # Alembic migrations
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── routes/              # API route handlers
│   │   └── schemas/             # Pydantic request/response schemas
│   ├── llm/
│   │   ├── engine.py            # Qwen3 inference wrapper
│   │   ├── functions.py         # Function definitions for tool use
│   │   ├── rag.py               # RAG retrieval pipeline
│   │   ├── cache.py             # Redis caching layer
│   │   └── validator.py         # SQL validation and sanitization
│   └── dashboard/
│       ├── app.py               # Streamlit or React app entry
│       └── components/          # UI components
├── notebooks/
│   ├── 01_vehicle_detection_training.ipynb
│   ├── 02_plate_detection_training.ipynb
│   ├── 03_plate_ocr_training.ipynb
│   ├── 04_tracking_evaluation.ipynb
│   ├── 05_trajectory_prediction_training.ipynb
│   ├── 06_analytics_development.ipynb
│   ├── 07_llm_fine_tuning.ipynb
│   └── 08_system_integration_tests.ipynb
├── scripts/
│   ├── download_datasets.sh     # Automated dataset downloading
│   ├── generate_synthetic_plates.py
│   ├── evaluate_pipeline.py     # End-to-end evaluation
│   └── demo.py                  # Live demo script
├── tests/
│   ├── test_perception.py
│   ├── test_tracking.py
│   ├── test_analytics.py
│   ├── test_llm.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
├── requirements.txt
├── setup.py
└── README.md
```

**Dependencies (requirements.txt):**
```
# Core
python>=3.11
numpy>=1.24
pandas>=2.0

# Deep Learning
torch>=2.1
torchvision>=0.16
ultralytics>=8.1
transformers>=4.51

# Computer Vision
opencv-python>=4.8
Pillow>=10.0
albumentations>=1.3

# Classical ML
scikit-learn>=1.3

# Tracking
boxmot>=11.0

# Embeddings and LLM
sentence-transformers>=2.2
peft>=0.7
accelerate>=0.25
bitsandbytes>=0.42

# Database
sqlalchemy>=2.0
psycopg2-binary>=2.9
alembic>=1.12
pgvector>=0.2

# Cache
redis>=5.0

# API
fastapi>=0.104
uvicorn>=0.24
pydantic>=2.5

# Dashboard
streamlit>=1.29
folium>=0.15
plotly>=5.18

# Utilities
pyyaml>=6.0
python-dotenv>=1.0
loguru>=0.7
tqdm>=4.66
pytest>=7.4
```

#### 1.2 Vehicle Detection Model (Week 1, Days 3-7)

**Pre-trained Starting Point:**
- Model: YOLOv8m (medium) from Ultralytics, available on Hugging Face
- Reason: Best speed/accuracy tradeoff; YOLOv8n is faster but less accurate, YOLOv8l/x are overkill for this resolution

**Training Pipeline (Google Colab Pro):**
1. Download UA-DETRAC dataset, convert annotations to YOLO format (class x_center y_center width height)
2. Split: 80% train, 10% validation, 10% test
3. Fine-tune YOLOv8m on vehicle classes: car, bus, van, truck, motorcycle
4. Training hyperparameters:
   - Epochs: 100 (with early stopping, patience=15)
   - Batch size: 16 (adjust based on Colab GPU memory)
   - Image size: 640x640
   - Optimizer: SGD with cosine LR schedule
   - Initial LR: 0.01
   - Augmentations: mosaic, mixup, HSV shift, flip, scale
5. Evaluate on test set: target mAP@0.5 >= 0.85

**Validation Metrics:**
- mAP@0.5 (primary)
- mAP@0.5:0.95 (secondary)
- Per-class precision and recall
- Inference speed (FPS on Colab GPU and MacBook CPU)

**Deliverable:** Trained YOLOv8m checkpoint saved to `models/vehicle_detector/best.pt`

#### 1.3 License Plate Detection Model (Week 2)

**Pre-trained Starting Point:**
- Model: YOLOv8s (small) from Ultralytics
- Reason: Plate detection operates on cropped vehicle regions (smaller images), so a smaller model suffices
- Alternative: Search Hugging Face for pre-trained plate detection models (e.g., keremberke/yolov8m-license-plate-detection) and fine-tune from there

**Training Pipeline (Google Colab Pro):**
1. Download UFPR-ALPR dataset
2. Extract plate bounding box annotations, convert to YOLO format
3. Supplement with CCPD dataset for volume (detection only, ignore Chinese characters)
4. Fine-tune YOLOv8s on single class: license_plate
5. Training hyperparameters:
   - Epochs: 80 (early stopping, patience=10)
   - Batch size: 32
   - Image size: 640x640
   - Same optimizer and augmentation strategy as vehicle detector
6. Evaluate: target mAP@0.5 >= 0.90 for plate detection

**Deliverable:** Trained YOLOv8s checkpoint saved to `models/plate_detector/best.pt`

#### 1.4 License Plate OCR Model (Week 3)

**Pre-trained Starting Point:**
- Model: microsoft/trocr-base-printed from Hugging Face
- Architecture: Vision Transformer (ViT) encoder + GPT-2 decoder
- Why TrOCR over traditional OCR: Handles perspective distortion, blur, and partial occlusion better than tesseract or paddleocr on plate-sized crops

**Training Pipeline (Google Colab Pro):**
1. Extract plate crops from UFPR-ALPR using bounding box annotations
2. Generate 50,000-100,000 synthetic US license plate images:
   - Use PIL to render plates with correct fonts, spacing, state formats
   - Apply augmentations: rotation (-15 to +15 degrees), Gaussian blur, noise, brightness variation, perspective transforms
   - Each image paired with ground truth text label
3. Combine real plate crops (UFPR-ALPR) with synthetic data
4. Fine-tune TrOCR:
   - Freeze ViT encoder for first 5 epochs, then unfreeze
   - Learning rate: 5e-5 with linear warmup and cosine decay
   - Batch size: 16
   - Epochs: 30
   - Loss: Cross-entropy on character sequence
5. Evaluate: target Character Error Rate (CER) <= 5%, Word Accuracy >= 90%

**Evaluation Protocol:**
- Test on held-out real plate crops (not synthetic)
- Report per-character accuracy and full-plate exact match rate
- Test on degraded images (blur levels, low resolution) to establish robustness bounds

**Deliverable:** Fine-tuned TrOCR checkpoint and tokenizer saved to `models/plate_ocr/`

#### 1.5 Perception Pipeline Integration (Week 4)

**Pipeline Flow:**
```
Input Frame
    |
    v
[YOLOv8m Vehicle Detection]
    |
    v
For each detected vehicle:
    |
    v
[Crop vehicle region from frame]
    |
    v
[YOLOv8s Plate Detection on crop]
    |
    v
If plate detected:
    |
    v
[Crop plate region]
    |
    v
[TrOCR Plate Reading]
    |
    v
Output: {vehicle_bbox, vehicle_class, confidence,
         plate_bbox, plate_text, plate_confidence,
         timestamp, camera_id}
```

**Integration Tasks:**
- Build `PerceptionPipeline` class that chains all three models
- Implement batched inference for multiple vehicles per frame
- Add confidence thresholds: vehicle detection >= 0.5, plate detection >= 0.6, plate OCR >= 0.7
- Build frame annotation utility (draw bounding boxes, plate text overlay)
- Test end-to-end on UA-DETRAC video clips
- Benchmark: target >= 10 FPS on Colab GPU, >= 2 FPS on MacBook CPU

**Deliverable:** Working `perception_pipeline.py` that processes video frames and returns structured detection results

---

### Phase 2: Tracking System (Weeks 5-7)

**Objective:** Track individual vehicles across frames within a single camera and across multiple cameras using visual embeddings.

#### 2.1 Single-Camera Multi-Object Tracking (Week 5)

**Algorithm: BoT-SORT (primary) / ByteTrack (fast alternative)**
- Library: BoxMOT (`pip install boxmot`) — actively maintained, pluggable tracker interface
- Why not DeepSORT: DeepSORT (2017) has high ID switch rates and slow ReID bottleneck. BoT-SORT achieves state-of-the-art MOTA on MOT benchmarks with fewer ID switches. ByteTrack achieves 171 FPS with competitive accuracy.
- Components (BoT-SORT):
  - Kalman Filter with camera motion compensation (CMC)
  - IoU + appearance cost matrix for association
  - Built-in re-ID model support (OSNet, CLIP) via BoxMOT

**Implementation:**
1. Initialize BoT-SORT tracker per camera feed via BoxMOT's pluggable interface
2. Feed YOLOv8 detections (bounding boxes + confidence) into tracker each frame
3. Tracker assigns persistent track IDs to each vehicle
4. Track lifecycle: tentative (first 3 frames) -> confirmed -> deleted (30 frames without match)
5. For each confirmed track, log to database: track_id, camera_id, frame_number, timestamp, bbox, vehicle_class, plate_text (if available)
6. If BoT-SORT latency is too high for a given feed, fall back to ByteTrack (same BoxMOT interface)

**Evaluation:**
- Metrics: MOTA (Multi-Object Tracking Accuracy), MOTP (Precision), IDF1 (identity preservation)
- Test on UA-DETRAC test set with ground truth track IDs
- Target: MOTA >= 0.75, IDF1 >= 0.70

**Deliverable:** Working `single_camera.py` that maintains vehicle identity across frames

#### 2.2 Vehicle Re-identification Embeddings (Week 6)

**Primary Model: OSNet (via BoxMOT) / DINOv2 (facebook/dinov2-base)**
- OSNet: lightweight, purpose-built re-ID backbone used in StrongSORT; included in BoxMOT ecosystem
- DINOv2: pre-trained self-supervised vision transformer, generates 768-dim embeddings
- Strategy: Start with BoxMOT's built-in OSNet re-ID (optimized for tracking). Evaluate DINOv2 zero-shot as alternative if OSNet underperforms on vehicle crops.
- BoxMOT also includes CLIP-based re-ID models fine-tuned on vehicle data

**Implementation:**
1. For each tracked vehicle, extract the best crop (highest resolution, least occlusion) per track
2. Pass crop through DINOv2 to generate 768-dim embedding vector
3. Store embedding alongside track record in database
4. Similarity function: cosine similarity between embeddings

**Evaluation:**
- Test on VeRi-776 dataset: compute Rank-1 accuracy and mAP for vehicle re-ID
- If zero-shot DINOv2 achieves Rank-1 >= 70%, proceed without fine-tuning
- If below threshold, fine-tune DINOv2 with contrastive loss on VeRi-776 training set

**Alternative Models:** CLIP (openai/clip-vit-base-patch32) as backup; fine-tuned DINOv2 on VeRi-776 if zero-shot underperforms

**Deliverable:** Working `reid_embedder.py` that generates embeddings for vehicle crops

#### 2.3 Cross-Camera Identity Resolution (Week 7)

**Logic:**
1. When a plate is read on Camera A, log it with timestamp and embedding
2. When the same plate appears on Camera B, link the two records into one identity
3. When a plate is not readable (occluded, low quality), fall back to embedding similarity:
   - Compare new detection embedding against all unmatched embeddings from other cameras within a time window
   - If cosine similarity >= 0.85, link as same vehicle
   - Time window constraint: only compare detections within a plausible travel time between camera locations

**Identity Graph:**
- Each unique vehicle is a node
- Edges connect sightings across cameras with timestamps
- Primary key: plate number (when available) or embedding cluster ID (when plate unavailable)

**Database Schema (relevant tables):**
```sql
CREATE TABLE vehicles (
    vehicle_id UUID PRIMARY KEY,
    plate_number VARCHAR(20),
    vehicle_class VARCHAR(20),
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    total_sightings INTEGER
);

CREATE TABLE sightings (
    sighting_id UUID PRIMARY KEY,
    vehicle_id UUID REFERENCES vehicles(vehicle_id),
    camera_id VARCHAR(50),
    timestamp TIMESTAMP,
    bbox_x FLOAT, bbox_y FLOAT, bbox_w FLOAT, bbox_h FLOAT,
    confidence FLOAT,
    plate_confidence FLOAT,
    embedding VECTOR(768),  -- pgvector
    frame_path VARCHAR(255)
);

CREATE TABLE camera_registry (
    camera_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    latitude FLOAT,
    longitude FLOAT,
    stream_url VARCHAR(500),
    status VARCHAR(20)
);
```

**Deliverable:** Working cross-camera identity resolution with vehicle identity graph persisted in PostgreSQL

---

### Phase 3: Database and Data Pipeline (Week 8)

**Objective:** Build the complete database schema, data ingestion pipeline, and query interface that all downstream components depend on.

#### 3.1 Database Schema (Full)

```sql
-- Core tables
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector

CREATE TABLE cameras (
    camera_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    stream_url VARCHAR(500) NOT NULL,
    resolution VARCHAR(20),
    fps INTEGER,
    status VARCHAR(20) DEFAULT 'active',
    coverage_zone JSONB,  -- GeoJSON polygon of camera FOV
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE vehicles (
    vehicle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plate_number VARCHAR(20) UNIQUE,
    vehicle_class VARCHAR(20),
    color VARCHAR(30),
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    total_sightings INTEGER DEFAULT 1,
    embedding_centroid VECTOR(768),  -- Average embedding for this vehicle
    metadata JSONB
);

CREATE TABLE detections (
    detection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id VARCHAR(50) REFERENCES cameras(camera_id),
    vehicle_id UUID REFERENCES vehicles(vehicle_id),
    timestamp TIMESTAMP NOT NULL,
    frame_number INTEGER,
    bbox_x FLOAT, bbox_y FLOAT, bbox_w FLOAT, bbox_h FLOAT,
    vehicle_class VARCHAR(20),
    vehicle_confidence FLOAT,
    plate_text VARCHAR(20),
    plate_confidence FLOAT,
    embedding VECTOR(768),
    track_id INTEGER,  -- per-camera track ID
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE trajectories (
    trajectory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_id UUID REFERENCES vehicles(vehicle_id),
    camera_sequence JSONB,  -- ordered list of camera_id + timestamp pairs
    route_cluster_id INTEGER,
    total_duration_seconds FLOAT,
    total_distance_km FLOAT,
    avg_speed_kmh FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE anomalies (
    anomaly_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_id UUID REFERENCES vehicles(vehicle_id),
    anomaly_type VARCHAR(50),  -- 'circling', 'impossible_speed', 'unusual_route', 'frequency'
    severity VARCHAR(20),  -- 'low', 'medium', 'high'
    description TEXT,
    detection_ids UUID[],
    detected_at TIMESTAMP DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE
);

CREATE TABLE route_clusters (
    cluster_id SERIAL PRIMARY KEY,
    centroid_path JSONB,  -- representative trajectory
    vehicle_count INTEGER,
    avg_duration_seconds FLOAT,
    time_distribution JSONB,  -- hourly distribution of traffic on this route
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_id UUID REFERENCES vehicles(vehicle_id),
    predicted_camera_id VARCHAR(50) REFERENCES cameras(camera_id),
    predicted_arrival TIMESTAMP,
    confidence FLOAT,
    model_version VARCHAR(50),
    actual_camera_id VARCHAR(50),  -- filled in after observation
    actual_arrival TIMESTAMP,      -- filled in after observation
    created_at TIMESTAMP DEFAULT NOW()
);

-- RAG knowledge base
CREATE TABLE knowledge_base (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255),
    content TEXT,
    category VARCHAR(50),  -- 'camera_info', 'traffic_pattern', 'system_doc', 'analysis_summary'
    embedding VECTOR(384),  -- MiniLM embedding dimension
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- LLM query logs (for fine-tuning data collection)
CREATE TABLE llm_query_logs (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_query TEXT,
    parsed_intent VARCHAR(50),
    function_called VARCHAR(100),
    function_params JSONB,
    raw_result JSONB,
    formatted_response TEXT,
    response_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_detections_camera_time ON detections(camera_id, timestamp);
CREATE INDEX idx_detections_vehicle ON detections(vehicle_id);
CREATE INDEX idx_detections_plate ON detections(plate_text);
CREATE INDEX idx_vehicles_plate ON vehicles(plate_number);
CREATE INDEX idx_anomalies_vehicle ON anomalies(vehicle_id);
CREATE INDEX idx_anomalies_type ON anomalies(anomaly_type);
CREATE INDEX idx_predictions_vehicle ON predictions(vehicle_id);
CREATE INDEX idx_sighting_embedding ON detections USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_knowledge_embedding ON knowledge_base USING hnsw (embedding vector_cosine_ops);
```

#### 3.2 Data Pipeline

**Ingestion Pipeline (runs continuously):**
1. `StreamHandler` connects to camera feeds, extracts frames at configured FPS
2. `FrameBuffer` queues frames with metadata (camera_id, timestamp)
3. `PerceptionPipeline` processes each frame, outputs detection records
4. `SingleCameraTracker` assigns track IDs within each camera stream
5. `CrossCameraResolver` links tracks across cameras into vehicle identities
6. All records inserted into PostgreSQL via batch inserts (every 100 detections or 5 seconds)

**Batch Analytics Pipeline (runs periodically):**
1. Every 15 minutes: recompute route clusters on new trajectory data
2. Every 5 minutes: run anomaly detection on recent detections
3. Every frame with a tracked vehicle: generate trajectory prediction
4. Daily: update vehicle behavioral profiles, generate analysis summaries for knowledge base

**Deliverable:** Complete database schema deployed, data pipeline inserting records from perception + tracking layers

---

### Phase 4: Analytics Engine (Weeks 9-11)

**Objective:** Build the intelligence layer that transforms raw detections into actionable insights.

#### 4.1 Route Clustering (Week 9)

**Method: DBSCAN on trajectory sequences**
- Input: sequences of (camera_id, timestamp) pairs per vehicle
- Feature engineering: encode each trajectory as a fixed-length vector using camera visit order and inter-camera travel times
- Algorithm: DBSCAN (density-based, handles arbitrary cluster shapes, auto-detects number of clusters)
- Hyperparameters: eps and min_samples tuned on initial data collection
- Alternative: K-Means if DBSCAN produces too many noise points

**scikit-learn Implementation:**
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Feature matrix: each row is a trajectory encoded as visit pattern
X = encode_trajectories(trajectory_data)
X_scaled = StandardScaler().fit_transform(X)

clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
```

**Output:** Each trajectory assigned a cluster_id; cluster centroids stored as representative routes

#### 4.2 Anomaly Detection (Week 9-10)

**Method: Multi-signal anomaly detection**

**Signal 1: Impossible speed (rule-based)**
- If a vehicle appears on Camera A at time T1 and Camera B at time T2, compute required speed
- If speed > physically possible (e.g., 200 mph for road travel), flag as anomaly
- Likely cause: plate misread or cloned plate

**Signal 2: Circling behavior (frequency analysis)**
- If a vehicle appears at the same camera more than N times within a time window, flag
- Threshold: configurable, default N=3 within 30 minutes

**Signal 3: Unusual route (Isolation Forest)**
```python
from sklearn.ensemble import IsolationForest

# Same trajectory feature matrix as clustering
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)
# -1 = anomaly, 1 = normal
```

**Signal 4: Temporal anomaly**
- Vehicle appears at unusual times relative to its historical pattern
- E.g., a vehicle that always appears 8-9 AM weekdays suddenly appears at 3 AM

**Output:** Anomaly records with type, severity, and linked detection IDs stored in anomalies table

#### 4.3 Trajectory Prediction (Weeks 10-11)

**Model: LSTM (PyTorch)**

**Architecture:**
```
Input: sequence of (camera_embedding, time_delta, speed, direction) tuples
    |
[LSTM layer 1: 128 hidden units]
    |
[LSTM layer 2: 64 hidden units]
    |
[Dropout: 0.3]
    |
[Fully Connected: 64 -> num_cameras (softmax)]  # predicted next camera
    |
[Fully Connected: 64 -> 1 (regression)]  # predicted arrival time
```

**Training (Google Colab Pro):**
1. Prepare training data from historical trajectories: input = first N sightings, target = (N+1)th camera and arrival time
2. Sequence length: N = 3 to 10 (variable, use padding and masking)
3. Loss: Cross-entropy for camera prediction + MSE for arrival time, weighted sum
4. Optimizer: Adam, LR = 1e-3 with ReduceLROnPlateau
5. Epochs: 50, batch size 64
6. Train/val/test: 70/15/15

**Evaluation:**
- Camera prediction: Top-1 accuracy, Top-3 accuracy
- Arrival time: MAE (Mean Absolute Error) in seconds
- Target: Top-1 >= 60%, Top-3 >= 85%, MAE <= 120 seconds

**Alternative:** Transformer encoder if LSTM underperforms; self-attention may capture long-range dependencies in trajectories better

**Deliverable:** Trained trajectory prediction model, anomaly detection pipeline, route clustering, all writing results to database

---

### Phase 5: Dashboard (Weeks 12-13)

**Objective:** Build a real-time command dashboard for visualizing all system outputs.

#### 5.1 Dashboard Components

**Live Feed View:**
- Grid of active camera feeds with detection overlays (bounding boxes, plate text, track IDs)
- Click any camera to expand to full view
- Real-time detection counter per camera

**Map View:**
- Interactive map (Leaflet/Mapbox) showing all camera locations
- Click camera pin to see live feed and recent detections
- Vehicle routes drawn as polylines on map with timestamps
- Heatmap overlay showing traffic density by area and time

**Vehicle Search:**
- Text input for plate number
- Returns: complete sighting history as timeline, route on map, behavioral profile
- Shows predicted next location with confidence score and estimated arrival time

**Anomaly Dashboard:**
- List of active anomalies sorted by severity
- Each anomaly links to the relevant vehicle and detections
- Filtering by type, severity, time range

**Analytics View:**
- Route cluster visualization (most common routes as colored paths on map)
- Traffic flow statistics (vehicles per hour per camera)
- Peak hours analysis
- Vehicle class distribution over time

#### 5.2 Technology Choice

**Recommendation: Streamlit for MVP, React for production**
- Streamlit: fastest to build, native Python, built-in charting, sufficient for demo
- React: needed if you want a polished, production-grade UI with real-time WebSocket updates

**Key Libraries:**
- Streamlit for layout and interactivity
- Folium for map rendering within Streamlit
- Plotly for interactive charts
- st-aggrid for data tables

#### 5.3 Backend API (FastAPI)

**Endpoints:**
```
GET  /api/cameras                    # List all cameras with status
GET  /api/cameras/{id}/feed          # Live frame from camera
GET  /api/cameras/{id}/detections    # Recent detections for camera

GET  /api/vehicles                   # List vehicles (paginated, filterable)
GET  /api/vehicles/{id}              # Full vehicle profile
GET  /api/vehicles/{id}/history      # Complete sighting history
GET  /api/vehicles/{id}/predict      # Predicted next location

GET  /api/search/plate/{plate}       # Search by plate number

GET  /api/anomalies                  # List anomalies (filterable)
GET  /api/anomalies/{id}             # Anomaly details

GET  /api/analytics/routes           # Route clusters
GET  /api/analytics/traffic          # Traffic flow statistics
GET  /api/analytics/heatmap          # Density data for map overlay

POST /api/query                      # Natural language query (LLM layer)
```

**Deliverable:** Working dashboard with live feeds, map, search, and analytics views, all backed by FastAPI

---

### Phase 6: LLM Intelligence Interface (Weeks 14-17)

**Objective:** Build a natural language analyst interface that queries the entire system through a fine-tuned open-source LLM with RAG.

#### 6.1 Base LLM Setup (Week 14)

**Model: Qwen3-30B-A3B**
- Architecture: Mixture of Experts, 30B total params, 3B activated per token
- License: Apache 2.0 (free for all use)
- Source: Hugging Face (Qwen/Qwen3-30B-A3B-Instruct-2507)

**Local Development (Ollama):**
```bash
ollama pull qwen3:30b-a3b
ollama run qwen3:30b-a3b
```
- Runs on MacBook for development and testing
- API available at http://localhost:11434

**Production Deployment (vLLM):**
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --quantization awq \
    --max-model-len 8192
```
- Deploy on cloud GPU for live demo
- OpenAI-compatible API

#### 6.2 Function Definitions (Week 14)

**The LLM can call these functions:**

```python
functions = [
    {
        "name": "search_plate",
        "description": "Search for a vehicle by license plate number. Returns full sighting history.",
        "parameters": {
            "plate_number": {"type": "string", "description": "License plate to search"}
        }
    },
    {
        "name": "get_vehicle_history",
        "description": "Get complete movement history for a vehicle.",
        "parameters": {
            "vehicle_id": {"type": "string"},
            "hours_back": {"type": "integer", "default": 24}
        }
    },
    {
        "name": "get_camera_activity",
        "description": "Get recent detection activity for a specific camera.",
        "parameters": {
            "camera_id": {"type": "string"},
            "hours_back": {"type": "integer", "default": 1}
        }
    },
    {
        "name": "get_anomalies",
        "description": "Retrieve anomalies filtered by type, severity, and time.",
        "parameters": {
            "anomaly_type": {"type": "string", "optional": True},
            "severity": {"type": "string", "optional": True},
            "hours_back": {"type": "integer", "default": 24}
        }
    },
    {
        "name": "predict_vehicle_location",
        "description": "Predict where a vehicle will appear next.",
        "parameters": {
            "vehicle_id": {"type": "string"}
        }
    },
    {
        "name": "get_traffic_stats",
        "description": "Get traffic flow statistics for a camera or area.",
        "parameters": {
            "camera_id": {"type": "string", "optional": True},
            "time_range": {"type": "string", "description": "e.g., 'last_hour', 'today', 'this_week'"}
        }
    },
    {
        "name": "find_co_traveling_vehicles",
        "description": "Find vehicles frequently seen together with a target vehicle.",
        "parameters": {
            "vehicle_id": {"type": "string"},
            "min_co_occurrences": {"type": "integer", "default": 3}
        }
    },
    {
        "name": "execute_sql",
        "description": "Execute a validated SQL query against the database. Only SELECT queries allowed.",
        "parameters": {
            "query": {"type": "string", "description": "SQL SELECT query"}
        }
    }
]
```

#### 6.3 RAG Pipeline (Week 15)

**Knowledge Base Population:**
1. Camera documentation: location descriptions, coverage zones, known blind spots
2. System documentation: how anomaly detection works, what each metric means, route cluster definitions
3. Traffic pattern summaries: auto-generated daily/weekly summaries of traffic patterns
4. Historical analysis reports: any past investigation or analysis results

**Embedding Pipeline:**
```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# 384-dimensional embeddings

# Chunk documents into ~500 token passages
chunks = chunk_documents(knowledge_base_docs, max_tokens=500, overlap=50)

# Embed and store in pgvector
for chunk in chunks:
    embedding = embedder.encode(chunk.text)
    insert_into_knowledge_base(chunk.title, chunk.text, chunk.category, embedding)
```

**Retrieval at Query Time:**
```python
# Embed user query
query_embedding = embedder.encode(user_query)

# Retrieve top-k relevant chunks from pgvector
relevant_chunks = db.execute("""
    SELECT title, content, 1 - (embedding <=> %s::vector) as similarity
    FROM knowledge_base
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", [query_embedding, query_embedding])
```

#### 6.4 Caching Layer (Week 15)

**Redis Configuration:**

**Level 1: Query Result Cache**
```python
import redis
import hashlib
import json

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_result(query_hash):
    result = cache.get(f"query:{query_hash}")
    return json.loads(result) if result else None

def cache_result(query_hash, result, ttl_seconds):
    cache.setex(f"query:{query_hash}", ttl_seconds, json.dumps(result))

# TTL strategy:
# Live data queries (current camera activity): 30 seconds
# Recent historical queries (last hour): 60 seconds
# Historical queries (last 24h+): 300 seconds
# Static queries (route clusters, camera info): 3600 seconds
```

**Level 2: Semantic Query Cache (pgvector)**
```python
def get_similar_cached_query(query_embedding, threshold=0.95):
    # Use pgvector similarity search instead of brute-force iteration
    # This scales to any number of cached queries via HNSW index
    result = db.execute("""
        SELECT response, 1 - (embedding <=> %s::vector) as similarity
        FROM query_cache
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT 1
    """, [query_embedding, query_embedding, threshold, query_embedding])
    return result[0]['response'] if result else None
```
- **Note:** Previous design iterated over all cached queries in Redis with O(n) cosine similarity — this does not scale. Using pgvector with an HNSW index provides O(log n) approximate nearest neighbor search.

#### 6.5 SQL Validation Layer (Week 15)

**Critical security component. The LLM proposes SQL, this layer validates before execution.**

```python
ALLOWED_TABLES = {'cameras', 'vehicles', 'detections', 'trajectories',
                  'anomalies', 'route_clusters', 'predictions'}
ALLOWED_OPERATIONS = {'SELECT'}
BLOCKED_KEYWORDS = {'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER',
                    'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE', '--', ';'}

def validate_sql(query: str) -> tuple[bool, str]:
    query_upper = query.upper().strip()

    # Must start with SELECT
    if not query_upper.startswith('SELECT'):
        return False, "Only SELECT queries are allowed"

    # Check for blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_upper:
            return False, f"Blocked keyword detected: {keyword}"

    # Verify only allowed tables are referenced
    # (use sqlparse for proper parsing in production)
    for table in extract_tables(query):
        if table not in ALLOWED_TABLES:
            return False, f"Access to table '{table}' is not permitted"

    return True, "Query validated"
```

#### 6.6 LLM Query Pipeline (Week 16)

**Complete Query Flow:**

```
User: "Where has plate ABC-1234 been in the last 6 hours?"
    |
[Check Redis embedding cache] -- cache hit --> return cached response
    |  (cache miss)
    v
[Embed query with MiniLM]
    |
[Retrieve relevant RAG chunks from pgvector]
    |
[Construct prompt: system instructions + RAG context + function defs + user query]
    |
[Send to Qwen3-30B-A3B (temperature=0)]
    |
[LLM returns function call: search_plate(plate_number="ABC-1234")]
    |
[Execute function against PostgreSQL]
    |
[Filter results to last 6 hours]
    |
[Return structured data to LLM]
    |
[LLM formats natural language response with data]
    |
[Cache response in Redis]
    |
[Log query, intent, function, result, response to llm_query_logs table]
    |
User receives: "Plate ABC-1234 was spotted at 4 locations in the last 6 hours:
               Camera 12 (Broadway & 42nd) at 2:03 PM, Camera 7 (8th Ave & 34th) at 2:11 PM..."
```

**System Prompt for Base Model (pre-fine-tuning):**
```
You are the intelligence analyst for Project Jarvis, a traffic monitoring system.
You have access to a database of vehicle detections, trajectories, and anomalies
from a network of traffic cameras.

RULES:
1. Only answer questions using data from the provided functions. Never fabricate data.
2. If you cannot find the requested information, say so clearly.
3. Always include timestamps and camera locations when reporting vehicle sightings.
4. Format responses for clarity: use chronological order for sighting histories.
5. When reporting anomalies, include severity and recommended actions.
6. For predictions, always state the confidence level.

You have access to the following functions:
[function definitions injected here]

Relevant system context:
[RAG chunks injected here]
```

**Deliverable:** Working natural language query interface with RAG, function calling, caching, and SQL validation

#### 6.7 LLM Fine-Tuning (Week 17)

**Prerequisites:**
- Minimum 500 logged query-response pairs from llm_query_logs table
- Manual review and correction of failed queries
- Categorized training examples covering all function types

**Bootstrapping Note:** Fine-tuning requires data from the live system running with the base (non-fine-tuned) model. The system must operate with the base Qwen3 model first to collect sufficient query-response pairs. Fine-tuning is therefore a Phase 7+ activity — the base model handles all queries during initial deployment, and fine-tuning improves accuracy iteratively as usage data accumulates.

**Fine-Tuning Method: QLoRA**
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

**Training Data Format:**
```json
{
    "messages": [
        {"role": "system", "content": "[condensed system prompt]"},
        {"role": "user", "content": "Where has plate ABC-1234 been today?"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "search_plate", "arguments": "{\"plate_number\": \"ABC-1234\"}"}}
        ]},
        {"role": "tool", "content": "[structured result from database]"},
        {"role": "assistant", "content": "Plate ABC-1234 has been spotted at 3 locations today..."}
    ]
}
```

**Training Hyperparameters:**
- Platform: Google Colab Pro, A100 40GB
- Epochs: 3-5 (monitor validation loss for overfitting)
- Batch size: 4 (with gradient accumulation steps=4, effective batch=16)
- Learning rate: 2e-4 with cosine schedule
- Warmup ratio: 0.03
- Max sequence length: 2048
- Training time: ~2-4 hours on A100

**Post-Fine-Tuning Evaluation:**
- Compare function calling accuracy: base model vs. fine-tuned on held-out test queries
- Measure response format consistency
- Test adversarial queries (prompt injection attempts)
- Measure prompt length reduction (fine-tuned model should need shorter system prompts)
- Target: function call accuracy >= 95%, format consistency >= 98%

**Deliverable:** Fine-tuned LoRA adapter weights saved to `models/llm/`, deployed alongside base model

---

### Phase 7: Integration and Production Hardening (Weeks 18-20)

**Objective:** Wire all layers together into a single deployable system and harden for demo.

#### 7.1 Docker Compose Architecture (Week 18)

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: jarvis
      POSTGRES_USER: jarvis
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  ingestion:
    build: ./docker/ingestion
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgresql://jarvis:${DB_PASSWORD}@postgres:5432/jarvis
    volumes:
      - ./models:/app/models
      - ./config:/app/config

  perception:
    build: ./docker/perception
    depends_on:
      - postgres
      - ingestion
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  tracking:
    build: ./docker/tracking
    depends_on:
      - postgres
      - perception

  analytics:
    build: ./docker/analytics
    depends_on:
      - postgres
      - tracking

  api:
    build: ./docker/api
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"

  llm:
    build: ./docker/llm
    depends_on:
      - postgres
      - redis
      - api
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  dashboard:
    build: ./docker/dashboard
    depends_on:
      - api
    ports:
      - "8501:8501"

volumes:
  pgdata:
```

#### 7.2 System Integration Testing (Week 19)

**End-to-End Test Suite:**

1. **Ingestion test:** Connect to live camera feed, verify frames arrive in buffer with correct metadata
2. **Perception test:** Process 1000 frames, verify detection rate, plate read rate, no crashes
3. **Tracking test:** Process 5-minute video, verify track continuity, correct ID assignment
4. **Cross-camera test:** Simulate same vehicle on two feeds, verify identity linking
5. **Analytics test:** Seed database with 10,000 detections, run clustering and anomaly detection, verify sensible outputs
6. **Prediction test:** Feed historical trajectory, verify prediction matches within acceptable MAE
7. **LLM test:** Submit 50 test queries covering all function types, verify correct function selection and response formatting
8. **RAG test:** Submit queries requiring knowledge base context, verify relevant chunks retrieved
9. **Cache test:** Submit duplicate queries, verify cache hit on second request
10. **SQL injection test:** Submit adversarial queries, verify validation layer blocks them
11. **Load test:** Simulate 10 concurrent camera feeds, verify system handles throughput without falling behind
12. **Latency test:** Measure end-to-end time from frame capture to database insertion, target < 500ms

#### 7.3 Performance Optimization (Week 19)

**Model Optimization:**
- Convert YOLOv8 models to TensorRT for 2-3x inference speedup on NVIDIA GPUs
- Use ONNX Runtime as fallback for CPU inference
- Batch multiple vehicle crops before running plate detection (batch inference)
- Implement frame skipping: process every Nth frame based on available compute

**Database Optimization:**
- Connection pooling (SQLAlchemy pool_size=20)
- Batch inserts (insert 100 records per transaction)
- Partition detections table by timestamp (monthly partitions)
- Vacuum and analyze on schedule

**LLM Optimization:**
- AWQ quantization for vLLM deployment (4-bit, minimal quality loss)
- KV cache optimization
- Speculative decoding if available for Qwen3

#### 7.4 Documentation and Demo Preparation (Week 20)

**README.md:** Project overview, architecture diagram, setup instructions, demo video link

**Demo Script:**
1. Show live camera feeds with detections running
2. Search a specific plate, show full history on map
3. Show anomaly dashboard with active alerts
4. Ask natural language questions: "Which vehicles were on Broadway in the last hour?" / "Are there any anomalies right now?" / "Where is plate XYZ-789 headed?"
5. Show route clusters and traffic flow analytics

**Resume Description:**
```
Project Jarvis: Intelligent Traffic Intelligence Platform
- Built end-to-end AI platform processing live traffic camera feeds for vehicle detection,
  tracking, and trajectory prediction across multi-camera networks
- Trained YOLOv8 (PyTorch) for vehicle/plate detection, TrOCR (PyTorch/Transformers) for plate OCR,
  LSTM for trajectory prediction; achieved 87% mAP vehicle detection, 92% plate read accuracy
- Implemented cross-camera vehicle re-identification using DINOv2/OSNet embeddings with
  BoT-SORT multi-object tracking via BoxMOT
- Built analytics engine with scikit-learn (DBSCAN clustering, Isolation Forest anomaly
  detection) processing 10,000+ detections daily
- Designed RAG-powered natural language interface using fine-tuned Qwen3-30B-A3B (QLoRA)
  with pgvector, Redis caching, and validated SQL generation
- Full stack: PyTorch, scikit-learn, PostgreSQL, pgvector, Redis, FastAPI,
  Docker, React/Streamlit
```

---

## 5. Evaluation Framework

### 5.1 Model-Level Metrics

| Model | Primary Metric | Target | Secondary Metrics |
|---|---|---|---|
| Vehicle Detection (YOLOv8) | mAP@0.5 | >= 0.85 | Per-class precision/recall, FPS |
| Plate Detection (YOLOv8) | mAP@0.5 | >= 0.90 | Recall at various IoU thresholds |
| Plate OCR (TrOCR) | Character Error Rate | <= 5% | Full-plate exact match >= 90% |
| Tracking (BoT-SORT) | MOTA | >= 0.75 | IDF1 >= 0.70, ID switches per 100 frames |
| Re-ID (DINOv2) | Rank-1 Accuracy | >= 0.70 | mAP on VeRi-776 |
| Route Clustering (DBSCAN) | Silhouette Score | >= 0.40 | Cluster stability across time windows |
| Anomaly Detection (IF) | Precision@k | >= 0.80 | False positive rate <= 10% |
| Trajectory Prediction (LSTM) | Top-1 Camera Accuracy | >= 60% | Top-3 >= 85%, Time MAE <= 120s |
| LLM Function Calling | Accuracy | >= 95% | Response format consistency >= 98% |

### 5.2 System-Level Metrics

| Metric | Target |
|---|---|
| End-to-end latency (frame to DB) | < 500ms |
| Sustained throughput | >= 10 cameras at 5 FPS each |
| LLM query response time | < 3 seconds |
| Cache hit rate | >= 40% after warmup |
| System uptime during demo | >= 99% (no crashes) |
| Dashboard refresh rate | < 2 seconds |

---

## 6. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Public camera feeds go offline | Demo breaks | Cache recent footage; support local video file playback mode |
| Plate OCR accuracy too low on real feeds | Core feature degraded | Fall back to embedding-only re-ID; generate more synthetic training data |
| Qwen3-30B-A3B too slow on MacBook | Dev workflow blocked | Use Qwen3-8B for development, 30B for production demo only |
| Colab Pro GPU unavailable during training | Training delayed | Kaggle free GPU as backup; Lambda Labs for burst compute |
| Cross-camera re-ID false matches | Incorrect vehicle linking | Increase similarity threshold; require plate match confirmation |
| LLM generates invalid SQL | Security risk / bad data | SQL validation layer blocks all non-SELECT queries; whitelist tables |
| Scope creep | Project never ships | Each phase is a standalone deliverable; ship incrementally |
| PostgreSQL query performance degrades at scale | System slows down | Table partitioning, connection pooling, proper indexing |

---

## 7. Timeline Summary

| Phase | Weeks | Deliverable |
|---|---|---|
| Phase 1: Perception Pipeline | 1-4 | Vehicle detection + plate detection + plate OCR |
| Phase 2: Tracking System | 5-7 | Single-camera MOT + cross-camera re-ID |
| Phase 3: Database Pipeline | 8 | Complete schema + data ingestion pipeline |
| Phase 4: Analytics Engine | 9-11 | Clustering + anomaly detection + trajectory prediction |
| Phase 5: Dashboard | 12-13 | Real-time visualization + search + API |
| Phase 6: LLM Intelligence | 14-17 | RAG + function calling + fine-tuning |
| Phase 7: Integration | 18-20 | Docker deployment + testing + demo preparation |

**Total estimated timeline: 20 weeks (5 months)**

---

## 8. Hugging Face Models Summary

| Purpose | Model | Source | Size |
|---|---|---|---|
| Vehicle Detection | YOLOv8m (fine-tuned) | ultralytics/yolov8 | ~50MB |
| Plate Detection | YOLOv8s (fine-tuned) | ultralytics/yolov8 | ~22MB |
| Plate OCR | TrOCR | microsoft/trocr-base-printed | ~334MB |
| Vehicle Re-ID | DINOv2 | facebook/dinov2-base | ~346MB |
| RAG Embeddings | MiniLM | sentence-transformers/all-MiniLM-L6-v2 | ~80MB |
| LLM Analyst | Qwen3-30B-A3B | Qwen/Qwen3-30B-A3B-Instruct-2507 | ~16GB (4-bit) |

---

## 9. Cost Estimate

| Item | Cost |
|---|---|
| Google Colab Pro | $0 (student plan) |
| PostgreSQL | $0 (local Docker) |
| Redis | $0 (local Docker) |
| Ollama (local LLM) | $0 |
| Hugging Face models | $0 (open source) |
| Datasets | $0 (all public) |
| Cloud GPU for demo (optional) | ~$1-2/hour spot instance |
| Domain for portfolio (optional) | ~$12/year |
| **Total minimum cost** | **$0** |

---

## 10. Skills Demonstrated

| Category | Specific Skills |
|---|---|
| Deep Learning | CNNs, object detection (YOLO), OCR (TrOCR), sequence models (LSTM), transfer learning, vision transformers |
| Classical ML | DBSCAN clustering, Isolation Forest, feature engineering, model evaluation |
| Computer Vision | Multi-object tracking, vehicle re-identification, real-time video processing |
| LLM Engineering | RAG, function calling, QLoRA fine-tuning, prompt engineering, tool use |
| Data Engineering | PostgreSQL, pgvector, Redis caching, batch/stream data pipelines |
| Software Engineering | FastAPI, Docker, REST API design, system architecture, testing |
| MLOps | Model training pipelines, evaluation frameworks, quantization, deployment |
| Frameworks | PyTorch, scikit-learn, Hugging Face, OpenCV, Ultralytics, BoxMOT |

---

## 11. Security and Privacy

### 11.1 API Authentication

All API endpoints require authentication. Strategy:
- **JWT-based authentication** for all `/api/*` endpoints
- API keys for service-to-service communication (ingestion, perception, tracking services)
- Role-based access control: `admin` (full access), `analyst` (read + query), `viewer` (dashboard only)
- Rate limiting on `/api/query` (LLM endpoint) to prevent abuse

### 11.2 PII Handling and Data Privacy

License plate numbers are personally identifiable information (PII). The system must address:

**Data Retention:**
- Raw detection data: retained for 90 days, then aggregated and anonymized
- License plate numbers: hashed after retention period; only aggregate statistics preserved
- Video frames/crops: retained for 7 days, then deleted (only metadata persists)

**Access Controls:**
- Plate number searches logged with user identity and justification
- Embedding vectors stored without direct plate linkage where possible
- Database-level row security policies restricting plate data access by role

**Anonymization:**
- Dashboard views default to anonymized mode (partial plate masking: `AB*-**34`)
- Full plate numbers only visible to authenticated analysts
- Export/API responses can be configured to mask PII

**Legal Considerations:**
- Compliance requirements vary by jurisdiction (DPPA in the US, GDPR if applicable)
- System designed for authorized traffic management use cases only
- Data processing agreements required for any third-party data sharing

### 11.3 SQL Validation

The keyword-blocking approach in Section 6.5 is a development placeholder. For production:
- Use `sqlparse` to parse the query AST and validate structure
- Block subqueries, CTEs, `UNION`, `INTO OUTFILE`, `LOAD_FILE()`, and `information_schema` access
- Use parameterized query builders that construct queries from validated parameters rather than accepting raw SQL strings
- All LLM-generated SQL is logged and auditable

### 11.4 CORS Configuration

FastAPI CORS middleware for React frontend:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 12. Error Handling and Monitoring

### 12.1 Resilience Strategy

**Camera Feed Failures:**
- Automatic reconnection with exponential backoff (1s, 2s, 4s, ... up to 60s)
- Mark camera as `degraded` after 3 failed reconnections, `offline` after 10
- Continue processing other cameras; system degrades gracefully per-camera
- Dashboard shows camera health status in real time

**Model Inference Failures:**
- Circuit breaker pattern: if a model fails 5 times in 60 seconds, stop sending requests for 30 seconds
- Fallback: skip plate OCR if TrOCR fails (still log vehicle detection); skip re-ID if embedding model fails
- All inference wrapped in try/except with structured error logging

**Database Failures:**
- Connection pool with retry logic (SQLAlchemy pool_pre_ping)
- Write buffer: if DB is temporarily unreachable, buffer up to 1000 detection records in memory
- Alert if buffer exceeds 80% capacity

### 12.2 Health Check Endpoints

```
GET /health              # Overall system health (200 if core services up)
GET /health/cameras      # Per-camera connection status
GET /health/models       # Model loading and inference status
GET /health/database     # PostgreSQL and Redis connectivity
GET /health/llm          # Qwen3 model availability and response time
```

### 12.3 Monitoring and Observability

**Structured Logging:**
- All services use `loguru` with JSON output for machine-parseable logs
- Log levels: DEBUG (inference details), INFO (detections, queries), WARNING (degraded services), ERROR (failures)
- Correlation IDs trace a frame through the entire pipeline

**Metrics (Prometheus):**
- `jarvis_frames_processed_total` — counter per camera
- `jarvis_detections_total` — counter by vehicle class
- `jarvis_inference_duration_seconds` — histogram per model
- `jarvis_tracking_id_switches_total` — counter per camera
- `jarvis_llm_query_duration_seconds` — histogram
- `jarvis_cache_hit_ratio` — gauge
- `jarvis_camera_status` — gauge (1=active, 0=offline)

**Dashboards (Grafana):**
- System overview: cameras online, detections/sec, model latency
- Per-camera detail: FPS, detection rate, tracking quality
- LLM performance: query latency, cache hit rate, function call accuracy
- Alerts: camera offline > 5 min, model latency > 2x baseline, error rate > 5%

---

## 13. GPU Memory Management

### 13.1 Model Memory Budget

| Model | VRAM (FP16) | Deployment |
|---|---|---|
| YOLOv8m (vehicle detection) | ~100MB | GPU (primary inference) |
| YOLOv8s (plate detection) | ~50MB | GPU (shared with vehicle detector) |
| TrOCR (plate OCR) | ~670MB | GPU |
| DINOv2/OSNet (re-ID) | ~700MB | GPU |
| LSTM (trajectory prediction) | ~50MB | CPU (low throughput, batch) |
| Qwen3-30B-A3B (4-bit) | ~16GB | Separate process via Ollama |

**Total GPU for perception pipeline:** ~1.5GB — fits comfortably on any modern GPU alongside Qwen3.

### 13.2 Deployment Strategy

- **Perception models** (YOLO, TrOCR, DINOv2/OSNet): loaded into a single GPU process, run concurrently
- **Qwen3 LLM**: runs as a separate Ollama process with its own GPU memory allocation. On a single-GPU system, Ollama manages memory sharing. For production, deploy on a separate GPU or use vLLM on a cloud GPU.
- **Analytics models** (scikit-learn, LSTM): run on CPU — lightweight and batch-oriented
- **Model loading/unloading**: all models loaded at startup and kept resident. If GPU memory is constrained, TrOCR and DINOv2 can be offloaded to CPU with ~3x latency increase.

---

## 14. MVP Scope and Fallback Plan

### 14.1 Minimum Viable Demo

If the 20-week timeline slips, the following defines the minimum set of features for a working demo:

**Core MVP (Phase 1 + Phase 3 + Phase 5):**
- Vehicle detection and plate OCR on live camera feeds
- All detections stored in PostgreSQL
- Dashboard showing live feeds with detection overlays, vehicle search by plate, and basic analytics

This alone demonstrates: deep learning (YOLO, TrOCR), computer vision, data engineering, and full-stack development.

**Stretch Goal 1 (add Phase 2):** Multi-object tracking and cross-camera re-ID
**Stretch Goal 2 (add Phase 4):** Analytics engine (clustering, anomaly detection, trajectory prediction)
**Stretch Goal 3 (add Phase 6):** LLM intelligence interface with RAG

### 14.2 Phase Dependencies

```
Phase 1 (Perception) ──> Phase 2 (Tracking) ──> Phase 4 (Analytics)
       │                                              │
       └──> Phase 3 (Database) ──> Phase 5 (Dashboard) ──> Phase 7 (Integration)
                                        │
                                        └──> Phase 6 (LLM Intelligence)
```

Each phase produces a standalone deliverable. If a phase takes longer than estimated, subsequent phases can begin with mock/simulated data from the incomplete phase.
