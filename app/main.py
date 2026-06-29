from fastapi import FastAPI
from app.routers import eda, data_engineering, modeling, metrics

app = FastAPI(
    title="Global EV Adoption MLOps API",
    description="Productionized ML service for predicting EV adoption likelihood.",
    version="1.0.0",
)

app.include_router(eda.router)
app.include_router(data_engineering.router)
app.include_router(modeling.router)
app.include_router(metrics.router)


@app.get("/health")
def health():
    return {"status": "ok"}
