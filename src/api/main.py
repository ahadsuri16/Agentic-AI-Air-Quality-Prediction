import os
import sys
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import datetime

# Ensure model and agents are importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.predictive_model import AQIPredictor
from agents.multi_agent_system import MultiAgentOrchestrator

app = FastAPI(
    title="Agentic AI Air Quality Prediction API",
    description="Production REST API for real-time environmental IoT telemetry, XGBoost AQI forecasting, and autonomous multi-agent response.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = AQIPredictor()
orchestrator = MultiAgentOrchestrator(predictor)

# In-memory shared state for live dashboard streaming
latest_state = {
    "telemetry": {
        "pm25": 35.0, "pm10": 60.0, "no2": 25.0, "so2": 8.0, "co": 0.8,
        "o3": 30.0, "temperature": 25.0, "humidity": 50.0, "wind_speed": 4.5, "pressure": 1013.0
    },
    "predicted_aqi": 52.4,
    "category": "Moderate",
    "color": "#FFFF00",
    "severity": "Moderate",
    "dominant_pollutant": "PM2.5",
    "timestamp": datetime.datetime.now().isoformat(),
    "advisory": {
        "public_health_guideline": "Air quality is acceptable.",
        "recommended_actions": ["Maintain normal operations."],
        "municipal_response": "Standard environmental protocols active."
    }
}

history_log = []

class SensorTelemetry(BaseModel):
    pm25: float = Field(..., example=32.4)
    pm10: float = Field(..., example=58.1)
    no2: float = Field(..., example=22.0)
    so2: float = Field(..., example=9.5)
    co: float = Field(..., example=0.8)
    o3: float = Field(..., example=35.0)
    temperature: float = Field(..., example=27.2)
    humidity: float = Field(..., example=55.0)
    wind_speed: float = Field(..., example=3.2)
    pressure: float = Field(..., example=1012.0)

class ChatQuery(BaseModel):
    query: str = Field(..., example="Is it safe for an outdoor run right now?")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Agentic AI Air Quality Prediction Engine",
        "model_metrics": predictor.metrics
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/api/sensor_data")
def ingest_sensor_data(data: SensorTelemetry, background_tasks: BackgroundTasks):
    """
    Sub-10ms sensor ingestion endpoint. Delegates multi-agent reasoning to background tasks.
    """
    telemetry_dict = data.dict()

    def run_agent_pipeline(telemetry):
        global latest_state, history_log
        processed = orchestrator.process_telemetry(telemetry)
        processed["timestamp"] = datetime.datetime.now().isoformat()
        latest_state = processed

        history_log.append({
            "timestamp": processed["timestamp"],
            "aqi": processed["predicted_aqi"],
            "pm25": telemetry["pm25"],
            "pm10": telemetry["pm10"],
            "category": processed["category"]
        })
        if len(history_log) > 100:
            history_log.pop(0)

    background_tasks.add_task(run_agent_pipeline, telemetry_dict)
    return {"status": "INGESTED", "received_at": datetime.datetime.now().isoformat()}

@app.get("/api/dashboard_data")
def get_dashboard_data():
    """
    Retrieves the latest state, telemetry, and autonomous advisories for the frontend dashboard.
    """
    return {
        "latest": latest_state,
        "history": history_log[-25:],
        "metrics": predictor.metrics
    }

@app.post("/api/predict")
def predict_aqi_direct(data: SensorTelemetry):
    """
    Synchronous direct prediction endpoint.
    """
    result = orchestrator.process_telemetry(data.dict())
    return result

@app.post("/api/chat")
def chat_with_agent(payload: ChatQuery):
    """
    Conversational interface with the Reporting Agent.
    """
    reply = orchestrator.reporting_agent.respond_to_query(
        payload.query,
        {
            "aqi": latest_state.get("predicted_aqi", 50),
            "category": latest_state.get("category", "Good"),
            "dominant_pollutant": latest_state.get("dominant_pollutant", "PM2.5")
        }
    )
    return {"response": reply, "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
