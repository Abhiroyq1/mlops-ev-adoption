# Global EV Adoption Prediction Service

A fully productionized, end-to-end Machine Learning service that predicts Electric Vehicle (EV) adoption likelihood — **High**, **Medium**, or **Low** — based on a person's demographic, financial, behavioural, and infrastructure profile.

Dataset - [text](https://www.kaggle.com/datasets/aiexplorer77/from-fuel-to-electric-the-ev-transition-dataset/data)

Built with **FastAPI** and **scikit-learn**. Packaged with **Docker**. Tested with **pytest**.

> **7 models supported:** Random Forest, Gradient Boosting, Logistic Regression, Extra Trees, AdaBoost, SVM (RBF), and Soft Voting Ensemble. Every model can be trained, cross-validated, predicted, and evaluated through HTTP endpoints.

---

## Table of Contents

- [What This Repository Is For](#what-this-repository-is-for)
- [How This Works — The Big Picture](#how-this-works--the-big-picture)
- [If This Were a Plain Script](#if-this-were-a-plain-script)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Quick Start — Run Locally](#quick-start--run-locally)
- [Testing with Postman — Step by Step](#testing-with-postman--step-by-step)
- [Docker — Build and Run](#docker--build-and-run)
- [Testing with Postman on Docker](#testing-with-postman-on-docker)
- [API Endpoints — Full Reference](#api-endpoints--full-reference)
- [How the Endpoints Connect](#how-the-endpoints-connect)
- [Running the Test Suite](#running-the-test-suite)
- [Configuration](#configuration)
- [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
- [Dataset](#dataset)
- [Model Performance](#model-performance)

---

## What This Repository Is For

Most ML projects live in Jupyter notebooks — great for exploration, but not something you can call from another system or deploy to a server.

This repository takes that same ML work and wraps it into a **proper application**:

- Every step of the ML lifecycle (data exploration → training → cross-validation → prediction → evaluation) is a **callable HTTP API endpoint**
- The app runs as a **server** — anything that can make an HTTP request (Postman, a browser, another service, a frontend) can use it
- It is packaged as a **Docker container** — one command to build, one command to run, works the same on any machine

Think of it as: *"what does a data science notebook look like after it grows up into a production service?"*

---

## How This Works — The Big Picture

```
You (or any client)          FastAPI Server                   Files on disk
─────────────────            ──────────────                   ─────────────
                    HTTP
  Postman    ─────────────►  /eda/summary          reads ──► CSV dataset
  Browser    ◄─────────────  /eda/analysis
  curl                       /data-eng/splits
                             /model/train           saves ──► models/*.joblib
                             /model/cross-validate  (no artifact — scores only)
                             /model/predict         loads ◄── models/*.joblib
                             /model/list
                             /metrics/evaluate      loads ◄── models/*.joblib
```

The server holds the dataset in memory after the first read. Trained models are saved as `.joblib` files and loaded on demand. Cross-validation runs on the training split in memory and returns scores without saving anything. You do not need to write any Python — you just make HTTP requests.

---

## If This Were a Plain Script

To understand what each API endpoint does, here is the entire application logic written as a single Python script. Each comment block corresponds directly to one or more endpoints.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ── STEP 1: DATA READING ───────────────────────────────────────────────────────
# API: GET /eda/summary  and  GET /eda/analysis
# The CSV is read once and held in memory for all future requests.

df = pd.read_csv("data/global_ev_adoption_behavior_2026.csv")
df.columns = df.columns.str.strip().str.lower()

# ── STEP 2: EXPLORATORY DATA ANALYSIS ─────────────────────────────────────────
# API: GET /eda/summary — rows, columns, missing values, class balance
# API: GET /eda/analysis — describe(), correlation matrix, value counts

print(df.shape)                                          # (50000, 23)
print(df["ev_adoption_likelihood"].value_counts())       # High / Medium / Low counts
print(df.select_dtypes("number").describe())             # numeric stats
print(df.select_dtypes("number").corr())                 # correlation matrix

# ── STEP 3: FEATURE ENGINEERING & SELECTION ───────────────────────────────────
# API: GET /data-eng/splits — see which features are used and split sizes

NUMERIC_FEATURES = [
    "age", "annual_income", "daily_commute_km", "weekly_travel_distance_km",
    "vehicle_age_years", "fuel_expense_per_month", "charging_station_accessibility",
    "nearest_charging_station_km", "electricity_cost_per_kwh",
    "environmental_awareness_score", "government_incentive_awareness",
    "technology_affinity_score", "range_anxiety_score", "battery_replacement_concern",
    "ev_knowledge_score", "monthly_energy_consumption_kwh", "monthly_charging_cost",
]
CATEGORICAL_FEATURES = [
    "education_level", "city_type", "current_vehicle_type",
    "home_charging_available", "previous_ev_experience",
]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["ev_adoption_likelihood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# X_train → 40,000 rows   X_test → 10,000 rows

# ── STEP 4: TRAIN THE MODEL ────────────────────────────────────────────────────
# API: POST /model/train?model_name=random_forest
# Builds a Pipeline (preprocessing + classifier), fits it, saves to disk.
# clone() gives each call a fresh estimator — no shared-instance state bugs.

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", clone(RandomForestClassifier(n_estimators=100, random_state=42))),
])

pipeline.fit(X_train, y_train)
print("Train accuracy:", pipeline.score(X_train, y_train))  # ~98.9%
print("Test  accuracy:", pipeline.score(X_test,  y_test))   # ~85.25%
joblib.dump(pipeline, "models/random_forest.joblib")

# ── STEP 5: CROSS-VALIDATE ────────────────────────────────────────────────────
# API: POST /model/cross-validate?model_name=extra_trees&cv=5
# Runs StratifiedKFold CV on the training split. No artifact is saved.
# Use this to check variance and generalisation before committing to a model.

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(pipeline, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=1)
scores = results["test_score"]
print(f"CV mean: {scores.mean():.4f}  std: {scores.std():.4f}")
print(f"Fold scores: {scores.round(4)}")
print(f"Mean fit time: {results['fit_time'].mean():.2f}s")

# ── STEP 6: PREDICT FOR A NEW PERSON ──────────────────────────────────────────
# API: POST /model/predict?model_name=random_forest
# Loads the saved artifact, runs it on one new record.

pipeline = joblib.load("models/random_forest.joblib")
new_person = pd.DataFrame([{"age": 35, "annual_income": 75000, ...}])
print(pipeline.predict(new_person)[0])        # "High"
print(pipeline.predict_proba(new_person)[0])  # [0.72, 0.08, 0.20]

# ── STEP 7: EVALUATE ON THE TEST SET ──────────────────────────────────────────
# API: GET /metrics/evaluate?model_name=random_forest
# Loads the artifact, scores on the held-out test set — no re-training.

pipeline = joblib.load("models/random_forest.joblib")
y_pred = pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))             # 0.8525
print(classification_report(y_test, y_pred))      # per-class P / R / F1
print(confusion_matrix(y_test, y_pred))           # 3×3 matrix
```

**The API is this exact script, broken into named HTTP-accessible pieces.**

| Script step | API endpoint |
|---|---|
| Step 1 — data reading | Implicit in all endpoints (cached after first call) |
| Step 2 — EDA (shape) | `GET /eda/summary` |
| Step 2 — EDA (stats) | `GET /eda/analysis` |
| Step 3 — feature selection | `GET /data-eng/splits` |
| Step 4 — train | `POST /model/train?model_name=random_forest` |
| Step 5 — cross-validate | `POST /model/cross-validate?model_name=extra_trees&cv=5` |
| Step 6 — predict | `POST /model/predict?model_name=random_forest` |
| Step 7 — evaluate | `GET /metrics/evaluate?model_name=random_forest` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                          │
│                       app/main.py                                │
│                                                                  │
│   ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌──────────┐  │
│   │  /eda/*  │  │ /data-eng/*   │  │ /model/* │  │/metrics/*│  │
│   │  Router  │  │    Router     │  │  Router  │  │  Router  │  │
│   └────┬─────┘  └──────┬────────┘  └────┬─────┘  └────┬─────┘  │
│        ▼               ▼                ▼              ▼         │
│   ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌──────────┐  │
│   │   EDA    │  │ Preprocessing │  │  Model   │  │ Metrics  │  │
│   │ Service  │  │   Service     │  │ Service  │  │ Service  │  │
│   └────┬─────┘  └──────┬────────┘  └────┬─────┘  └────┬─────┘  │
│        └───────────────┴────────────────┴──────────────┘        │
│                                │                                 │
│                                ▼                                 │
│              ┌─────────────────────────────────┐                 │
│              │   Data Service  (@lru_cache)     │                 │
│              │   Reads CSV once, holds in RAM   │                 │
│              └────────────────┬────────────────┘                 │
└───────────────────────────────┼──────────────────────────────────┘
                                ▼
            data/global_ev_adoption_behavior_2026.csv

            app/ml/pipeline.py
            sklearn Pipeline → clone() → joblib.dump / joblib.load
                                ▼
            models/random_forest.joblib
            models/gradient_boosting.joblib
            models/logistic_regression.joblib
            models/extra_trees.joblib
            models/adaboost.joblib
            models/svm.joblib
            models/voting_soft.joblib
```

`main.py` mounts four routers via `app.include_router()`. Every route defined in each router file is automatically registered on the `app` object — which is why running `uvicorn app.main:app` gives you all endpoints even though `main.py` itself only has the `/health` route written directly in it.

---

## Folder Structure

```
mlops_practise/
│
├── app/                              # All application code
│   ├── main.py                       # Entry point — creates app, mounts routers
│   ├── config.py                     # All settings (paths, split size, random seed)
│   │
│   ├── routers/                      # HTTP layer — one file per route group
│   │   ├── eda.py                    # GET /eda/summary, GET /eda/analysis
│   │   ├── data_engineering.py       # GET /data-eng/splits
│   │   ├── modeling.py               # POST /model/train, /model/predict,
│   │   │                             #      /model/cross-validate, GET /model/list
│   │   └── metrics.py                # GET /metrics/evaluate
│   │
│   ├── services/                     # Business logic — one file per concern
│   │   ├── data_service.py           # CSV loader with @lru_cache
│   │   ├── eda_service.py            # Statistical summaries
│   │   ├── preprocessing.py          # Feature definitions + train/test split
│   │   ├── model_service.py          # Train, predict, cross-validate, list artifacts
│   │   └── metrics_service.py        # Accuracy, classification report, confusion matrix
│   │
│   ├── ml/
│   │   └── pipeline.py               # sklearn Pipeline builder + 7-model registry
│   │
│   └── schemas/
│       ├── requests.py               # Pydantic model for POST /predict body
│       └── responses.py              # Pydantic models for all API responses
│
├── data/
│   └── global_ev_adoption_behavior_2026.csv   # 50,000 records, 23 columns
│
├── models/                           # Auto-created on startup; .joblib files saved here
│
├── tests/
│   ├── conftest.py                   # Shared fixtures (sample DataFrame, test client)
│   ├── unit/                         # Fast tests — no CSV, all dependencies mocked
│   ├── api/                          # HTTP layer tests — services mocked
│   └── integration/                  # Full-stack tests against real CSV
│
├── docker/
│   └── Dockerfile
│
├── requirements.txt                  # All dependencies (app + testing)
├── pytest.ini
├── .env.example
└── README.md
```

---

## Prerequisites

Before you start, make sure you have the following installed on your machine.

### To run locally (without Docker)

- **Python 3.11 or later** — check with `python --version`
- **pip** — comes with Python
- The dataset file at `data/global_ev_adoption_behavior_2026.csv`

### To run with Docker

- **Docker Desktop** — download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

> **Check Docker Desktop is running before any Docker command.**
> Open your system tray (bottom-right on Windows) and look for the Docker whale icon. If it is not there, open Docker Desktop from the Start menu and wait until it says **"Docker Desktop is running"**. Running `docker` commands while Docker Desktop is closed will give you errors like `"error during connect"` or `"Cannot connect to the Docker daemon"`.

---

## Quick Start — Run Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/Abhiroyq1/mlops-ev-adoption.git
cd mlops-ev-adoption
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Start the server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You will see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

The server is now running. **Keep this terminal window open** — closing it stops the server.

**Step 4 — Confirm it works**

Open your browser and go to `http://localhost:8000/health`. You should see:
```json
{"status": "ok"}
```

Or open `http://localhost:8000/docs` to see the full interactive API documentation (Swagger UI).

---

## Testing with Postman — Step by Step

Postman is a tool that lets you make HTTP requests without writing any code. Download it free from [postman.com](https://www.postman.com/downloads/).

> **Important:** The server must be running (`uvicorn` or Docker container) before you make any Postman requests. You should see `{"status": "ok"}` from `/health` before proceeding.

---

### Step 1 — Health Check

Confirm the server is alive.

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/health` |
| Body | None |

Click **Send**. Expected response:
```json
{"status": "ok"}
```

---

### Step 2 — Explore the Dataset

See a summary of the data the model will train on.

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/eda/summary` |
| Body | None |

Expected response (abbreviated):
```json
{
  "num_rows": 50000,
  "num_columns": 23,
  "target_distribution": {"High": 29670, "Medium": 12078, "Low": 8252}
}
```

For deeper statistics (correlations, value counts):

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/eda/analysis` |

---

### Step 3 — Inspect the Feature Set and Split Sizes

See which 22 features will be used and how the data will be divided before training.

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/data-eng/splits` |

Expected response:
```json
{
  "train_size": 40000,
  "test_size": 10000,
  "feature_columns": ["age", "annual_income", "...22 features total..."]
}
```

---

### Step 4 — Train a Model

This is a POST request with **no body** — the model name goes in the URL as a query parameter.

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://localhost:8000/model/train?model_name=random_forest` |
| Body | None (select **None** in the Body tab) |

> **Note:** Random Forest on 40,000 rows takes **30–60 seconds**. Postman will appear to hang — this is normal. Wait for the response.

Expected response:
```json
{
  "model_name": "random_forest",
  "artifact_path": "models/random_forest.joblib",
  "train_accuracy": 0.9891,
  "test_accuracy": 0.8525,
  "classes": ["High", "Low", "Medium"]
}
```

**All supported models** — swap the `model_name` query parameter:

| Model | URL | Training time (approx) |
|---|---|---|
| `random_forest` | `POST .../model/train?model_name=random_forest` | 30–60 seconds |
| `gradient_boosting` | `POST .../model/train?model_name=gradient_boosting` | ~3 minutes |
| `logistic_regression` | `POST .../model/train?model_name=logistic_regression` | ~10 seconds |
| `extra_trees` | `POST .../model/train?model_name=extra_trees` | 20–40 seconds |
| `adaboost` | `POST .../model/train?model_name=adaboost` | ~2 minutes |
| `svm` | `POST .../model/train?model_name=svm` | **5–15 minutes** — SVM uses Platt scaling internally |
| `voting_soft` | `POST .../model/train?model_name=voting_soft` | ~5 minutes (trains RF + GB + LR together) |

> **SVM note:** `svm` uses RBF kernel with `probability=True`. On 40,000 rows this triggers internal 5-fold cross-validation (Platt scaling) during training — budget extra time. Start with `extra_trees` or `adaboost` first.

---

### Step 5 — List Trained Models

Confirm which models are ready to use.

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/model/list` |

Expected response:
```json
["random_forest", "extra_trees"]
```

---

### Step 6 — Cross-Validate a Model

Cross-validation runs **Stratified K-Fold** on the training split and returns per-fold accuracy scores. This is different from train/test accuracy — it shows how stable the model is across different data slices, which is a much better indicator of real-world performance.

No artifact is saved. You can run this before or after training.

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://localhost:8000/model/cross-validate?model_name=extra_trees&cv=5` |
| Body | None |

The `cv` parameter sets the number of folds (2–10, default 5).

Expected response:
```json
{
  "model_name": "extra_trees",
  "cv_folds": 5,
  "cv_scores": [0.9312, 0.9298, 0.9341, 0.9287, 0.9320],
  "mean_accuracy": 0.9312,
  "std_accuracy": 0.0019,
  "min_accuracy": 0.9287,
  "max_accuracy": 0.9341,
  "mean_fit_time_seconds": 4.23
}
```

**How to read this:**
- `mean_accuracy` — average accuracy across all folds — your headline number
- `std_accuracy` — low std (< 0.005) means the model is stable and not sensitive to which data it sees
- `min_accuracy` / `max_accuracy` — the worst and best fold — big gaps indicate instability
- `mean_fit_time_seconds` — how long each fold takes to train

> **SVM warning:** Cross-validating `svm` runs 5 full SVM training cycles. On 40K rows this can take 30–60 minutes. Use `extra_trees`, `adaboost`, or `logistic_regression` for quick CV comparisons.

You can compare models by running CV on each:
```
POST http://localhost:8000/model/cross-validate?model_name=extra_trees
POST http://localhost:8000/model/cross-validate?model_name=adaboost
POST http://localhost:8000/model/cross-validate?model_name=logistic_regression
POST http://localhost:8000/model/cross-validate?model_name=voting_soft
```

---

### Step 7 — Predict for a New Person

This is a POST request **with a JSON body**. In Postman:
1. Set Method to `POST`
2. Set URL to `http://localhost:8000/model/predict?model_name=random_forest`
3. Click the **Body** tab
4. Select **raw**
5. Select **JSON** from the dropdown on the right
6. Paste the JSON below into the body box
7. Click **Send**

```json
{
  "features": {
    "age": 35,
    "annual_income": 75000,
    "daily_commute_km": 25,
    "weekly_travel_distance_km": 180,
    "vehicle_age_years": 5,
    "fuel_expense_per_month": 200,
    "charging_station_accessibility": 7,
    "nearest_charging_station_km": 3.5,
    "electricity_cost_per_kwh": 0.12,
    "environmental_awareness_score": 8,
    "government_incentive_awareness": 6,
    "technology_affinity_score": 9,
    "range_anxiety_score": 3,
    "battery_replacement_concern": 4,
    "ev_knowledge_score": 7,
    "monthly_energy_consumption_kwh": 350,
    "monthly_charging_cost": 45,
    "education_level": "Bachelor",
    "city_type": "Urban",
    "current_vehicle_type": "Gasoline",
    "home_charging_available": "Yes",
    "previous_ev_experience": "No"
  }
}
```

Expected response:
```json
{
  "prediction": "High",
  "probabilities": {
    "High": 0.7214,
    "Medium": 0.1893,
    "Low": 0.0893
  }
}
```

> **Common mistakes:**
> - Getting `404` → you have not trained the model yet. Run Step 4 first.
> - Getting `405 Method Not Allowed` → you sent a GET instead of POST. Change the method to POST.
> - Getting `422 Unprocessable Entity` → your JSON is malformed or a field name is wrong.

---

### Step 8 — Evaluate Model Metrics

Get accuracy, classification report, and confusion matrix for a trained model.

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/metrics/evaluate?model_name=random_forest` |
| Body | None |

To evaluate a different model, change the query parameter:
```
GET http://localhost:8000/metrics/evaluate?model_name=gradient_boosting
GET http://localhost:8000/metrics/evaluate?model_name=extra_trees
GET http://localhost:8000/metrics/evaluate?model_name=adaboost
GET http://localhost:8000/metrics/evaluate?model_name=voting_soft
```

Expected response (abbreviated):
```json
{
  "model_name": "random_forest",
  "accuracy": 0.8525,
  "confusion_matrix": [[3001, 180, 153], [197, 2700, 436], [141, 398, 2794]],
  "classes": ["High", "Low", "Medium"]
}
```

---

## Docker — Build and Run

Docker packages the entire app — Python, dependencies, code, and data — into a self-contained image that runs the same way on any machine.

### Step 1 — Make sure Docker Desktop is running

Open Docker Desktop from the Start menu. Wait until the bottom-left corner says **"Engine running"** or the whale icon in the system tray is steady (not animated).

You can also verify in a terminal:
```cmd
docker info
```
If you see server details, Docker is running. If you see `"error during connect"`, Docker Desktop is not started yet.

---

### Step 2 — Build the Docker image

Open a terminal (CMD or PowerShell), navigate to the project root (`mlops_practise`), and run:

```cmd
docker build -f docker/Dockerfile -t evadoptionmodel .
```

What each part means:
- `docker build` — create an image
- `-f docker/Dockerfile` — the Dockerfile is inside the `docker/` folder (not the project root)
- `-t evadoptionmodel` — name the image `evadoptionmodel`
- `.` — use the current folder as the build context (so Docker can access `app/` and `data/`)

This will take 2–5 minutes the first time (downloading the Python base image and installing packages). You will see many lines of output — this is normal.

When it finishes you will see:
```
Successfully built <image-id>
Successfully tagged evadoptionmodel:latest
```

---

### Step 3 — Start the container

```cmd
docker run -p 8000:8000 evadoptionmodel
```

What each part means:
- `docker run` — create and start a container from the image
- `-p 8000:8000` — map port 8000 on your machine to port 8000 inside the container
- `evadoptionmodel` — the image name we gave in the build step

You will see the same uvicorn startup messages as running locally:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

**Keep this terminal window open.** If you close it or press Ctrl+C, the container stops.

---

### Step 4 — Verify the container is working

Open your browser and go to `http://localhost:8000/health`. You should see:
```json
{"status": "ok"}
```

The container is now running and accessible exactly like the local server.

---

### Useful Docker commands

Run in background (so your terminal is free):
```cmd
docker run -d -p 8000:8000 --name ev-service evadoptionmodel
```

Check if it is running:
```cmd
docker ps
```

View logs when running in background:
```cmd
docker logs ev-service
```

Stop the container:
```cmd
docker stop ev-service
```

Restart a stopped container:
```cmd
docker start ev-service
```

> **Trained models inside the container are lost when the container is removed.** If you want the `.joblib` files to survive container restarts, use a volume mount:
> ```cmd
> docker run -p 8000:8000 -v %cd%\models:/app/models evadoptionmodel
> ```
> This saves the model files to your local `models/` folder instead of inside the container.

---

## Testing with Postman on Docker

Once the container is running (`docker run -p 8000:8000 evadoptionmodel`), **Postman works identically to local testing**. Use the exact same URLs:

```
http://localhost:8000/health
http://localhost:8000/model/train?model_name=extra_trees
http://localhost:8000/model/cross-validate?model_name=extra_trees&cv=5
http://localhost:8000/model/predict?model_name=extra_trees
http://localhost:8000/metrics/evaluate?model_name=extra_trees
```

The `-p 8000:8000` flag in the run command is what makes this work — it forwards your machine's port 8000 to the container's port 8000. Postman sees no difference.

Follow the same [Step-by-Step Postman guide](#testing-with-postman--step-by-step) above. The only difference: **you must train the model again** after starting a fresh container (unless you used the volume mount), because a new container has no `.joblib` files inside it.

---

## API Endpoints — Full Reference

| Method | Endpoint | Body required | Description |
|---|---|---|---|
| GET | `/health` | No | Service health check |
| GET | `/eda/summary` | No | Dataset shape, types, missing values, class balance |
| GET | `/eda/analysis` | No | Describe, correlation matrix, category value counts |
| GET | `/data-eng/splits` | No | Feature lists and 40k/10k split sizes |
| POST | `/model/train` | No | Train a model, save artifact |
| POST | `/model/cross-validate` | No | Stratified K-Fold CV on training split — no artifact saved |
| POST | `/model/predict` | Yes — features JSON | Predict EV adoption for one person |
| GET | `/model/list` | No | List trained model artifacts on disk |
| GET | `/metrics/evaluate` | No | Accuracy, F1, confusion matrix on test split |

**Supported `model_name` values:** `random_forest` (default), `gradient_boosting`, `logistic_regression`, `extra_trees`, `adaboost`, `svm`, `voting_soft`

**Interactive docs:** `http://localhost:8000/docs` — Swagger UI with Try it out buttons for every endpoint.

---

### Full Endpoint Details

#### `GET /health`
```json
{"status": "ok"}
```

#### `GET /eda/summary`
```json
{
  "num_rows": 50000,
  "num_columns": 23,
  "columns": ["age", "annual_income", "..."],
  "dtypes": {"age": "int64", "education_level": "object"},
  "missing_values": {"age": 0, "annual_income": 0},
  "target_distribution": {"High": 29670, "Medium": 12078, "Low": 8252}
}
```

#### `GET /data-eng/splits`
```json
{
  "train_size": 40000,
  "test_size": 10000,
  "feature_columns": ["age", "annual_income", "...22 total..."],
  "numeric_features": ["age", "annual_income", "...17 total..."],
  "categorical_features": ["education_level", "city_type", "...5 total..."]
}
```

#### `POST /model/train?model_name=random_forest`
```json
{
  "model_name": "random_forest",
  "artifact_path": "models/random_forest.joblib",
  "train_accuracy": 0.9891,
  "test_accuracy": 0.8525,
  "classes": ["High", "Low", "Medium"]
}
```

#### `POST /model/cross-validate?model_name=extra_trees&cv=5`
```json
{
  "model_name": "extra_trees",
  "cv_folds": 5,
  "cv_scores": [0.9312, 0.9298, 0.9341, 0.9287, 0.9320],
  "mean_accuracy": 0.9312,
  "std_accuracy": 0.0019,
  "min_accuracy": 0.9287,
  "max_accuracy": 0.9341,
  "mean_fit_time_seconds": 4.23
}
```

#### `POST /model/predict?model_name=random_forest`
Request body:
```json
{
  "features": {
    "age": 35, "annual_income": 75000, "daily_commute_km": 25,
    "weekly_travel_distance_km": 180, "vehicle_age_years": 5,
    "fuel_expense_per_month": 200, "charging_station_accessibility": 7,
    "nearest_charging_station_km": 3.5, "electricity_cost_per_kwh": 0.12,
    "environmental_awareness_score": 8, "government_incentive_awareness": 6,
    "technology_affinity_score": 9, "range_anxiety_score": 3,
    "battery_replacement_concern": 4, "ev_knowledge_score": 7,
    "monthly_energy_consumption_kwh": 350, "monthly_charging_cost": 45,
    "education_level": "Bachelor", "city_type": "Urban",
    "current_vehicle_type": "Gasoline", "home_charging_available": "Yes",
    "previous_ev_experience": "No"
  }
}
```
Response:
```json
{
  "prediction": "High",
  "probabilities": {"High": 0.7214, "Medium": 0.1893, "Low": 0.0893}
}
```

#### `GET /metrics/evaluate?model_name=random_forest`
```json
{
  "model_name": "random_forest",
  "accuracy": 0.8525,
  "classification_report": {
    "High":   {"precision": 0.88, "recall": 0.90, "f1-score": 0.89, "support": 3334},
    "Medium": {"precision": 0.82, "recall": 0.81, "f1-score": 0.81, "support": 3333},
    "Low":    {"precision": 0.84, "recall": 0.83, "f1-score": 0.83, "support": 3333}
  },
  "confusion_matrix": [
    [3001, 180, 153],
    [197,  2700, 436],
    [141,  398,  2794]
  ],
  "classes": ["High", "Low", "Medium"]
}
```

---

## How the Endpoints Connect

You do not have to call endpoints in any fixed order, but these dependencies exist:

```
GET /health                          no dependencies — always works

GET /eda/summary    ─┐
GET /eda/analysis   ─┤──► loads CSV (cached after first call)
GET /data-eng/splits─┘

POST /model/train  ──────────────► loads CSV → splits → trains → saves .joblib

POST /model/cross-validate ──────► loads CSV → splits → CV on X_train (no artifact)

POST /model/predict ─────────────► requires .joblib file (train first)
GET  /metrics/evaluate ──────────► requires .joblib file (train first) + loads test split

GET  /model/list  ───────────────► reads models/ directory on disk
```

**Recommended order for a full run:**
```
1.  GET  /health
2.  GET  /eda/summary
3.  GET  /eda/analysis
4.  GET  /data-eng/splits
5.  POST /model/train?model_name=extra_trees
6.  POST /model/cross-validate?model_name=extra_trees&cv=5   ← check stability
7.  POST /model/train?model_name=adaboost                    (optional — compare)
8.  POST /model/cross-validate?model_name=adaboost           ← compare CV scores
9.  POST /model/train?model_name=random_forest               (optional)
10. POST /model/train?model_name=voting_soft                 (optional — ensemble)
11. GET  /model/list
12. POST /model/predict?model_name=extra_trees
13. GET  /metrics/evaluate?model_name=extra_trees
14. GET  /metrics/evaluate?model_name=random_forest          (compare test accuracy)
```

---

## Running the Test Suite

### What each tier does

| Tier | Folder | Tests | Speed | Needs CSV | What it checks |
|---|---|---|---|---|---|
| Unit | `tests/unit/` | ~120 | < 5 s | No — all deps mocked | Individual functions in isolation |
| API | `tests/api/` | ~60 | < 5 s | No — services mocked | HTTP routing, status codes, request validation |
| Integration | `tests/integration/` | ~10 | 2–5 min | Yes — real 50k-row CSV | Full stack: CSV → train → predict → evaluate |

**Total: 189 tests.** Unit and API run in ~8 seconds. Integration tests train real models so they are slow — run them before a release, not on every change.

The `-m integration` marker is registered in `pytest.ini`, which is why the filter works without any extra config.

---

### Step 0 — Activate your environment

```bash
# conda (used in this project)
conda activate mlops_test

# or if using venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows PowerShell
```

If you skip this step and run bare `pytest`, you will get `ModuleNotFoundError: No module named 'fastapi'`.

---

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs both the application packages and `pytest` / `httpx` (used by the test client).

---

### Step 2 — Run the fast tests (unit + API)

Run this during development after every code change. Takes ~8 seconds.

```bash
# Unit + API together (recommended default)
pytest tests/unit/ tests/api/ -v --tb=short

# Unit tests only
pytest tests/unit/ -v --tb=short

# API (HTTP layer) tests only
pytest tests/api/ -v --tb=short

# Same result using the marker — excludes integration
pytest -m "not integration" -v --tb=short
```

Expected output:
```
189 passed in 8.26s
```

---

### Step 3 — Run integration tests (full stack)

Requires the CSV file at `data/global_ev_adoption_behavior_2026.csv`. Trains real models on 40,000 rows — expect 2–5 minutes.

```bash
pytest tests/integration/ -v --tb=short
```

Run this before merging to `main` or after any change to the data pipeline, preprocessing, or model artifacts.

---

### Step 4 — Run everything at once

```bash
pytest -v --tb=short
```
---

### Useful flags

| Flag | Effect |
|---|---|
| `-v` | Verbose — shows each test name as it runs |
| `--tb=short` | Short traceback on failure — easier to read than the default |
| `-x` | Stop after the first failure — useful when debugging |
| `-k "test_train"` | Run only tests whose name matches the pattern |
| `--co` | Collect and list all tests without running them |

Example — run only tests related to cross-validation:
```bash
pytest -k "cross_validate" -v --tb=short
```

---

## Configuration

Settings are read from environment variables or a `.env` file. Copy `.env.example` to `.env` to customise:

```env
DATA_PATH=data/global_ev_adoption_behavior_2026.csv
MODELS_DIR=models
TARGET_COLUMN=ev_adoption_likelihood
TEST_SIZE=0.2
RANDOM_STATE=42
```

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/global_ev_adoption_behavior_2026.csv` | Path to the CSV dataset |
| `MODELS_DIR` | `models` | Where `.joblib` artifacts are saved |
| `TARGET_COLUMN` | `ev_adoption_likelihood` | Name of the prediction target column |
| `TEST_SIZE` | `0.2` | Fraction held out for evaluation (0.2 = 20%) |
| `RANDOM_STATE` | `42` | Seed for reproducible splits and models |

---

## Layer-by-Layer Breakdown

### `app/config.py` — Settings
Central configuration. All paths and parameters come from here. `MODELS_DIR` is created automatically on startup if it does not exist.

### `app/services/data_service.py` — Data Reading
`load_data()` reads the CSV once using `@lru_cache`. Every subsequent call from any endpoint gets the same in-memory DataFrame — no repeated disk reads.

### `app/services/eda_service.py` — Exploratory Analysis
Produces `describe()`, Pearson correlation matrix, and value counts for categorical columns. Answers: *"What does this data look like?"*

### `app/services/preprocessing.py` — Feature Engineering
Declares the 17 numeric and 5 categorical features. Runs a stratified 80/20 `train_test_split`. Stratification ensures all three target classes appear in both train and test in the same ratio.

### `app/ml/pipeline.py` — ML Pipeline
`build_pipeline(model_name)` returns an sklearn `Pipeline`:
- Numeric path: `SimpleImputer(median)` → `StandardScaler`
- Categorical path: `SimpleImputer(most_frequent)` → `OrdinalEncoder`
- Classifier: one of 7 supported models (see table below)

`clone()` is called on the estimator before adding it to the Pipeline — this gives every training call a fresh unfitted copy, preventing state corruption when the same model is trained multiple times in the same process.

The entire fitted pipeline (scaler + encoder + model) is saved as one `.joblib` file. Loading it for prediction gives back the same transformation chain — no train/serve mismatch.

| Model key | Algorithm | Parallelism | Notes |
|---|---|---|---|
| `random_forest` | RandomForestClassifier | `n_jobs=-1` | Robust, fast, good default |
| `gradient_boosting` | GradientBoostingClassifier | single-threaded | Highest accuracy, slowest |
| `logistic_regression` | LogisticRegression | single-threaded | Fastest, least overfit |
| `extra_trees` | ExtraTreesClassifier | `n_jobs=-1` | Faster than RF, similar accuracy |
| `adaboost` | AdaBoostClassifier | single-threaded | Classic sequential boosting |
| `svm` | SVC (rbf, probability=True) | single-threaded | Slow on large data (Platt scaling) |
| `voting_soft` | VotingClassifier (RF+GB+LR) | mixed | Often beats any single model |

### `app/services/model_service.py` — Training, Inference & Cross-Validation
- `train_model()` — fits the pipeline, scores on both splits, saves artifact
- `predict()` — loads artifact, wraps input in DataFrame, returns class + probabilities
- `cross_validate_model()` — runs StratifiedKFold CV on X_train, returns per-fold accuracy, mean, std, and average fit time; no artifact saved
- `list_trained_models()` — globs `models/*.joblib` for artifact names

### `app/services/metrics_service.py` — Evaluation
`compute_metrics()` loads the artifact, runs it on the held-out test split, returns accuracy, classification report, and confusion matrix. Never re-trains.

### `app/routers/` — HTTP Layer
Routers are thin wrappers. Each function: parses request → calls one service function → returns typed response. Error mapping: `FileNotFoundError` → 404, any other exception → 500, invalid input → 422 (automatic).

---

## Dataset

**File:** `data/global_ev_adoption_behavior_2026.csv` | **50,000 records** | **23 columns**

**Target:** `ev_adoption_likelihood` — `High` / `Medium` / `Low`

**17 numeric features:** `age`, `annual_income`, `daily_commute_km`, `weekly_travel_distance_km`, `vehicle_age_years`, `fuel_expense_per_month`, `charging_station_accessibility`, `nearest_charging_station_km`, `electricity_cost_per_kwh`, `environmental_awareness_score`, `government_incentive_awareness`, `technology_affinity_score`, `range_anxiety_score`, `battery_replacement_concern`, `ev_knowledge_score`, `monthly_energy_consumption_kwh`, `monthly_charging_cost`

**5 categorical features:** `education_level`, `city_type`, `current_vehicle_type`, `home_charging_available`, `previous_ev_experience`

---

## Model Performance

Evaluated on the stratified 10,000-record held-out test split. Train first, then call `/metrics/evaluate` to get your exact numbers — the values below are indicative.

| Model | Test Accuracy (approx) | Training Time (approx) | Recommended CV folds |
|---|---|---|---|
| Random Forest | ~85.3% | 30–60 seconds | 5 |
| Gradient Boosting | ~84–87% | ~3 minutes | 3 (slower per fold) |
| Logistic Regression | ~75–78% | ~10 seconds | 5 |
| Extra Trees | ~84–86% | 20–40 seconds | 5 |
| AdaBoost | ~80–83% | ~2 minutes | 5 |
| SVM (RBF) | ~82–85% | 5–15 minutes | 2 (very slow per fold) |
| Soft Voting | ~85–88% | ~5 minutes | 3 |

**Reading CV vs test accuracy:** CV mean accuracy runs on the training split (40K rows, fewer per fold). Test accuracy runs on the full held-out 10K. A high CV mean with low std (< 0.005) is a stronger signal than train accuracy alone — it means the model generalises consistently, not just on one lucky split.
