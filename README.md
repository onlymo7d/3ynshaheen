# 3YNSHAHEEN \| عين شاهين

**Real-Time AI Perimeter Detection & Threat Classification System**

3YNSHAHEEN (عين شاهين) is an AI-powered perimeter monitoring system that
detects, tracks, classifies, and assesses potential security threats in
real time. It combines multiple computer-vision models with tracking,
temporal reasoning, object association, threat assessment, event
logging, and a live web dashboard.

> **Pipeline:** Camera → Detection → Tracking → Association → Threat
> Assessment → Meaningful Events

------------------------------------------------------------------------

## Overview

3YNSHAHEEN was developed as a real-time perimeter detection and
classification project. Instead of treating every detection as an
isolated result, the system combines detections over time to build a
more stable understanding of the scene.

The system can recognize:

-   Agents and non-agents
-   Civilian and military vehicles
-   Drones
-   Visible weapons
-   Armed persons through weapon-to-person association

The system uses four threat levels:

**LOW → MEDIUM → HIGH → CRITICAL**

All detections contribute to live statistics, while important HIGH and
CRITICAL security events can be permanently logged with snapshots.

------------------------------------------------------------------------

## Key Features

-   Real-time multi-model object detection
-   Agent vs. non-agent classification
-   Civilian vs. military vehicle classification
-   Drone detection
-   Visible weapon detection
-   Multi-object tracking with persistent IDs
-   Temporal filtering and classification smoothing
-   Weapon-to-person association
-   Armed-person confirmation
-   Scene-level threat assessment
-   LOW, MEDIUM, HIGH, and CRITICAL threat levels
-   HIGH/CRITICAL event logging and snapshots
-   Date/session-based log organization
-   Live FPS and AI latency monitoring
-   Camera and storage health monitoring
-   Integrated Flask web dashboard
-   Local real-time processing

------------------------------------------------------------------------

## System Pipeline

``` text
Camera Feed
    │
    ▼
Object Detection
    ├── Person Model
    ├── Vehicle Model
    ├── Drone Model
    └── Weapon Model
    │
    ▼
Object Tracking
    │
    ▼
Temporal Filtering & Classification Smoothing
    │
    ▼
Weapon-to-Person Association
    │
    ▼
Threat Assessment
    │
    ▼
Meaningful Event Management
    ├── Live Statistics
    ├── HIGH / CRITICAL Alerts
    ├── JSONL Logs
    └── Event Snapshots
    │
    ▼
3YNSHAHEEN Web Dashboard
```

------------------------------------------------------------------------

## Model Performance

The individual models were trained and evaluated before integration into
the live pipeline.

  Model                 Precision   Recall    mAP50
  ------------------- ----------- -------- --------
  Drone Detection           \~92%    \~91%    \~94%
  Weapon Detection         85.13%   74.05%   80.84%
  Vehicle Detection         \~84%      ---    \~81%

The weapon detector achieved **58.22% mAP50-95**.

> Evaluation metrics reflect performance on the respective
> test/evaluation data. Live performance can vary with lighting,
> distance, object size, camera angle, occlusion, and environment.

------------------------------------------------------------------------

## Threat Assessment

Example threat logic used by the system:

  Situation                    Threat Level
  ---------------------------- --------------
  Normal non-agent             LOW
  Agent                        MEDIUM
  Confirmed drone              HIGH
  Confirmed military vehicle   HIGH
  Armed agent                  HIGH
  Armed non-agent              CRITICAL

The highest active threat can be used as the overall **scene threat**.

------------------------------------------------------------------------

## Meaningful Event Management

A live camera can observe the same object across hundreds of frames.
Recording every detection as a separate event would create large amounts
of repetitive data.

3YNSHAHEEN therefore uses tracking and object state to distinguish
continuous detections from meaningful security events.

Important alerts can include:

-   Timestamp
-   Object/classification
-   Confidence
-   Tracking ID
-   Threat level
-   Weapon status
-   Snapshot

The final configuration focuses permanent event storage and snapshots on
**HIGH** and **CRITICAL** alerts.

------------------------------------------------------------------------

## Web Dashboard

The Flask-based dashboard provides a single interface for:

-   Live annotated camera feed
-   Current scene threat
-   Real-time FPS
-   AI latency
-   Camera status
-   Storage status
-   Live object counts
-   Recent HIGH/CRITICAL alerts
-   Alert snapshots

### Dashboard Preview

Place a screenshot at `assets/dashboard_preview.png` and add:

``` markdown
![3YNSHAHEEN Dashboard](assets/dashboard_preview.png)
```

------------------------------------------------------------------------

## Project Structure

``` text
3ynshaheen/
│
├── Models/                 # Trained model weights
├── Logs/                   # Runtime event logs
├── Snapshots/              # HIGH/CRITICAL snapshots
├── static/                 # Logo and web assets
│
├── live_pipeline_v1.py     # Main AI/backend pipeline
├── requirements.txt        # Python dependencies
└── README.md
```

If `dashboard.py` is a legacy file and the final dashboard is fully
integrated into Flask, it can be removed from the final repository.

------------------------------------------------------------------------

## Installation

### Clone the repository

``` bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>
```

### Create a virtual environment

**Windows**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure all required trained model weights are available inside the
`Models/` directory and that their filenames match the paths configured
in the pipeline.

------------------------------------------------------------------------

## Running 3YNSHAHEEN

``` bash
python live_pipeline_v1.py
```

After the models load and the camera starts, open:

``` text
http://127.0.0.1:5000
```

The dashboard will display the live detection feed, statistics, system
health, threat level, and security alerts.

------------------------------------------------------------------------

## Hardware Acceleration

The pipeline can use available hardware acceleration:

``` text
NVIDIA GPU → CUDA
Apple Silicon → MPS
Fallback → CPU
```

Performance depends on the computer, model sizes, camera resolution, and
number of models running simultaneously.

------------------------------------------------------------------------

## Data Preparation

### People

Approximately **3,800 images** were used for two classes:

-   Agent
-   Non-agent

### Vehicles

Vehicle data was prepared for:

-   Civilian vehicle
-   Military vehicle

Irrelevant classes from the original dataset were removed during
preprocessing.

### Drones

Approximately **9,400 images** were used, covering different sizes,
angles, distances, and backgrounds.

### Weapons

The source dataset contained:

-   Handgun
-   Knife
-   Rifle
-   Sword

These were consolidated into a single **Weapon** class because the
system's primary objective is to determine whether a visible weapon is
present and whether it can be associated with a tracked person.

------------------------------------------------------------------------

## Technology

-   Python
-   Ultralytics YOLO
-   PyTorch
-   OpenCV
-   Flask
-   Multi-object tracking
-   JSON/JSONL event storage
-   CUDA / Apple MPS acceleration where available

------------------------------------------------------------------------

## Limitations

3YNSHAHEEN is a prototype/research project and should not be treated as
a production security system. Detection can be affected by poor
lighting, occlusion, long distances, small objects, unusual camera
angles, motion blur, and environments different from the training data.

AI detections and threat classifications can contain false positives and
false negatives and should not be the sole basis for consequential
security decisions.

------------------------------------------------------------------------

## Project Goal

The goal of 3YNSHAHEEN is not simply to draw bounding boxes. It
demonstrates a complete transition from:

**Raw Data → Trained Models → Real-Time Detection → Tracking → Context →
Threat Understanding → Meaningful Security Events**

This transforms raw computer-vision outputs into structured information
that an operator can understand and review.

------------------------------------------------------------------------

## Acknowledgments & Licensing

This project uses third-party datasets, libraries, and model frameworks.
Their respective licenses and attribution requirements remain
applicable.

Before publicly redistributing trained weights or datasets, verify the
redistribution terms of each original source.

The weapon dataset used during development was sourced from Roboflow
Universe and provided under **CC BY 4.0**.

------------------------------------------------------------------------

## Disclaimer

This repository is intended for educational, research, and demonstration
purposes. 3YNSHAHEEN is an AI-assisted monitoring prototype and is not a
substitute for professionally validated security infrastructure or human
judgment.
