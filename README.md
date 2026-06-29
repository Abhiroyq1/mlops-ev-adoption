# Global EV Adoption Prediction Service

A fully productionized, end-to-end Machine Learning service that predicts Electric Vehicle (EV) adoption likelihood — **High**, **Medium**, or **Low** — based on a person's demographic, financial, behavioural, and infrastructure profile.

Built with **FastAPI** and **scikit-learn**. Packaged with **Docker**. Tested with **pytest**.

> **Best model accuracy: 85.25%** — Random Forest on a held-out 10,000-record test split.

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

- Every step of the ML lifecycle (data exploration → training → prediction → evaluation) is a **callable HTTP API endpoint**
- The app runs as a **server** — anything that can make an HTTP request (Postman, a browser, another service, a frontend) can use it
- It is packaged as a **Docker container** — one command to build, one command to run, works the same on any machine

Think of it as: *"what does a data science notebook look like after it grows up into a production service?"*

---

## How This Works — The Big Picture

```
You (or any client)          FastAPI Server               Files on disk
─────────────────            ──────────────               ─────────────
                    HTTP
  Postman    ─────────────►  /eda/summary       reads ──► CSV dataset
  Browser    ◄─────────────  /eda/analysis
  curl                       /data-eng/splits
                             /model/train       saves ──► models/*.joblib
                             /model/predict     loads ◄── models/*.joblib
                             /model/list
                             /metrics/evaluate  loads ◄── models/*.joblib
```

The server holds the dataset in memory after the first read. Trained models are saved as `.joblib` files and loaded on demand. You do not need to write any Python — you just make HTTP requests.

---

## If This Were a Plain Script

To understand what each API endpoint does, here is the entire application logic written as a single Python script. Each comment block corresponds directly to one or more endpoints.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
])

pipeline.fit(X_train, y_train)
print("Train accuracy:", pipeline.score(X_train, y_train))  # ~98.9%
print("Test  accuracy:", pipeline.score(X_test,  y_test))   # ~85.25%
joblib.dump(pipeline, "models/random_forest.joblib")

# ── STEP 5: PREDICT FOR A NEW PERSON ──────────────────────────────────────────
# API: POST /model/predict?model_name=random_forest
# Loads the saved artifact, runs it on one new record.

pipeline = joblib.load("models/random_forest.joblib")
new_person = pd.DataFrame([{"age": 35, "annual_income": 75000, ...}])
print(pipeline.predict(new_person)[0])        # "High"
print(pipeline.predict_proba(new_person)[0])  # [0.72, 0.08, 0.20]

# ── STEP 6: EVALUATE ON THE TEST SET ──────────────────────────────────────────
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
| Step 5 — predict | `POST /model/predict?model_name=random_forest` |
| Step 6 — evaluate | `GET /metrics/evaluate?model_name=random_forest` |

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
            sklearn Pipeline → joblib.dump / joblib.load
                                ▼
            models/random_forest.joblib
            models/gradient_boosting.joblib
            models/logistic_regression.joblib
```

`main.py` mounts four routers via `app.include_router()`. Every route defined in each router file is automatically registered on the `app` object — which is why running `uvicorn app.main:app` gives you all 8 endpoints even though `main.py` itself only has the `/health` route written directly in it.

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
│   │   ├── modeling.py               # POST /model/train, POST /model/predict, GET /model/list
│   │   └── metrics.py                # GET /metrics/evaluate
│   │
│   ├── services/                     # Business logic — one file per concern
│   │   ├── data_service.py           # CSV loader with @lru_cache
│   │   ├── eda_service.py            # Statistical summaries
│   │   ├── preprocessing.py          # Feature definitions + train/test split
│   │   ├── model_service.py          # Train, predict, list artifacts
│   │   └── metrics_service.py        # Accuracy, classification report, confusion matrix
│   │
│   ├── ml/
│   │   └── pipeline.py               # sklearn Pipeline builder + model registry
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
git clone <repo-url>
cd mlops_practise
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

You can also train the other two models (faster):

```
POST http://localhost:8000/model/train?model_name=gradient_boosting
POST http://localhost:8000/model/train?model_name=logistic_regression
```

---

### Step 5 — List Trained Models

Confirm which models are ready to use.

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/model/list` |

Expected response:
```json
["random_forest", "logistic_regression"]
```

---

### Step 6 — Predict for a New Person

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

### Step 7 — Evaluate Model Metrics

Get accuracy, classification report, and confusion matrix for a trained model.

| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/metrics/evaluate?model_name=random_forest` |
| Body | None |

To evaluate a different model, change the query parameter:
```
GET http://localhost:8000/metrics/evaluate?model_name=gradient_boosting
GET http://localhost:8000/metrics/evaluate?model_name=logistic_regression
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
http://localhost:8000/model/train?model_name=random_forest
http://localhost:8000/model/predict?model_name=random_forest
http://localhost:8000/metrics/evaluate?model_name=random_forest
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
| POST | `/model/predict` | Yes — features JSON | Predict EV adoption for one person |
| GET | `/model/list` | No | List trained model artifacts on disk |
| GET | `/metrics/evaluate` | No | Accuracy, F1, confusion matrix on test split |

**Supported `model_name` values:** `random_forest` (default), `gradient_boosting`, `logistic_regression`

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
GET /health                     no dependencies — always works

GET /eda/summary    ─┐
GET /eda/analysis   ─┤──► loads CSV (cached after first call)
GET /data-eng/splits─┘

POST /model/train  ──────────► loads CSV → splits data → trains → saves .joblib

POST /model/predict ─────────► requires .joblib file (train first)
GET  /metrics/evaluate ──────► requires .joblib file (train first) + loads test split

GET  /model/list  ───────────► reads models/ directory on disk
```

**Recommended order for a full run:**
```
1.  GET  /health
2.  GET  /eda/summary
3.  GET  /eda/analysis
4.  GET  /data-eng/splits
5.  POST /model/train?model_name=random_forest
6.  POST /model/train?model_name=gradient_boosting      (optional)
7.  POST /model/train?model_name=logistic_regression    (optional)
8.  GET  /model/list
9.  POST /model/predict?model_name=random_forest
10. GET  /metrics/evaluate?model_name=random_forest
11. GET  /metrics/evaluate?model_name=gradient_boosting (compare)
```

---

## Running the Test Suite

| Tier | Folder | Speed | Needs CSV |
|---|---|---|---|
| Unit | `tests/unit/` | < 5 seconds | No — dependencies mocked |
| API | `tests/api/` | < 5 seconds | No — services mocked |
| Integration | `tests/integration/` | 2–5 minutes | Yes — real data, real training |

```bash
# Install dependencies (includes pytest)
pip install -r requirements.txt

# Fast tests only — recommended during development
pytest -m "not integration" -v

# Integration tests only
pytest -m integration -v

# All tests
pytest -v
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
- Classifier: one of `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`

The entire fitted pipeline (scaler + encoder + model) is saved as one `.joblib` file. Loading it for prediction gives back the same transformation chain — no train/serve mismatch.

### `app/services/model_service.py` — Training & Inference
- `train_model()` — fits the pipeline, scores on both splits, saves artifact
- `predict()` — loads artifact, wraps input in DataFrame, returns class + probabilities
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

Evaluated on the stratified 10,000-record held-out test split.

| Model | Test Accuracy | Training Time (approx) |
|---|---|---|
| Random Forest | **85.25%** | ~45 seconds |
| Gradient Boosting | Train then `GET /metrics/evaluate?model_name=gradient_boosting` | ~3 minutes |
| Logistic Regression | Train then `GET /metrics/evaluate?model_name=logistic_regression` | ~10 seconds |

Random Forest train accuracy is ~98.9% vs 85.25% test — a gap indicating overfitting typical of deep trees. This can be reduced by tuning `max_depth` or `min_samples_leaf`. Logistic Regression trains fastest and shows the smallest train/test gap. Gradient Boosting sits in between.
