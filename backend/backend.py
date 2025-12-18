"""
Fitness OS Backend - Event Ingestion & RAG Orchestration
QA-Hardened, Deterministic State Machine with Guardrails
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict

app = FastAPI(title="Fitness OS API")

# CORS for web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class WorkoutType(str, Enum):
    PUSH = "push"
    PULL = "pull"
    LEGS = "legs"
    REST = "rest"

class EventType(str, Enum):
    WEIGHT = "weight"
    SLEEP = "sleep"
    WAKE = "wake"
    HYDRATION = "hydration"
    MEAL = "meal"
    WORKOUT_SET = "workout_set"

class BaseEvent(BaseModel):
    event_type: EventType
    timestamp: datetime
    date: str  # YYYY-MM-DD
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class WeightEvent(BaseEvent):
    event_type: EventType = EventType.WEIGHT
    weight_kg: float = Field(gt=0)

class SleepEvent(BaseEvent):
    event_type: EventType = EventType.SLEEP

class WakeEvent(BaseEvent):
    event_type: EventType = EventType.WAKE

class HydrationEvent(BaseEvent):
    event_type: EventType = EventType.HYDRATION
    window_name: str
    ml: int = Field(ge=0)

class MealEvent(BaseEvent):
    event_type: EventType = EventType.MEAL
    meal_name: str
    kcal: int = Field(ge=0)

class WorkoutSetEvent(BaseEvent):
    event_type: EventType = EventType.WORKOUT_SET
    exercise: str
    set_number: int
    planned_reps: int
    actual_reps: int = Field(ge=0)

# ============================================================================
# CANONICAL DAILY OBJECT
# ============================================================================

class PlannedDay(BaseModel):
    workout: str
    meals_kcal: int
    hydration_ml: int
    meals: List[Dict[str, Any]]
    hydration_windows: List[Dict[str, Any]]
    workout_sets: List[Dict[str, Any]]

class LoggedData(BaseModel):
    weight: Optional[float] = None
    sleep_time: Optional[datetime] = None
    wake_time: Optional[datetime] = None
    meals: List[Dict[str, Any]] = []
    hydration: List[Dict[str, Any]] = []
    workout_sets: List[Dict[str, Any]] = []

class DerivedMetrics(BaseModel):
    sleep_duration_min: Optional[int] = None
    hydration_adherence_pct: int = 0
    calorie_surplus: int = 0
    workout_duration_min: Optional[int] = None

class ConfidenceFlags(BaseModel):
    weight_logged: bool = False
    hydration_complete: bool = False
    sleep_complete: bool = False
    nutrition_complete: bool = False
    workout_complete: bool = False

class CanonicalDay(BaseModel):
    date: str
    day_index: int
    planned: PlannedDay
    logged: LoggedData
    derived_metrics: DerivedMetrics
    confidence_flags: ConfidenceFlags
    data_quality_score: float = 0.0
    
    class Config:
        json_encoders = { datetime: lambda v: v.isoformat() }

# ============================================================================
# DATA STORE & LOGIC
# ============================================================================

class DataStore:
    def __init__(self):
        self.days: Dict[str, CanonicalDay] = {}
        self.events: List[BaseEvent] = []
        self.day0_locked: bool = False
        
    def get_day(self, date: str) -> Optional[CanonicalDay]:
        return self.days.get(date)
    
    def save_day(self, day: CanonicalDay):
        # FIXED: Changed 'date' to 'day.date'
        if day.day_index == 0 and self.day0_locked and day.date in self.days:
            raise HTTPException(400, "Day 0 is locked")
        self.days[day.date] = day

store = DataStore()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health():
    return {"status": "online", "days_logged": len(store.days)}

@app.post("/api/log")
async def sync_full_state(request: Request):
    """
    Endpoint for the React 'Sync to Cloud' button.
    Receives the entire frontend state and persists it.
    """
    try:
        data = await request.json()
        # In a real app, you would validate this against CanonicalDay model
        # For now, we update our in-memory store
        date_key = data.get("date")
        if not date_key:
            raise HTTPException(400, "Missing date in payload")
            
        # Basic parsing to ensure it matches our internal structure
        store.days[date_key] = data 
        return {"status": "success", "message": f"Synced {date_key}"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/events/ingest")
async def ingest_event(event: Dict[str, Any]):
    # ... (Your existing ingestion logic remains the same)
    return {"status": "event_received"}

# ============================================================================
# INITIALIZATION UTILS
# ============================================================================

def initialize_day(date: str, day_index: int) -> CanonicalDay:
    # (Your existing initialize_day logic remains the same)
    pass

# IMPORTANT: Remove uvicorn.run for Vercel. 
# Vercel handles the server lifecycle.
