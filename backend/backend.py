"""
Fitness OS Backend - Event Ingestion & RAG Orchestration
QA-Hardened, Deterministic State Machine with Guardrails
"""

from fastapi import FastAPI, HTTPException
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
# DATA MODELS - Strict Schema Validation
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
    weight_kg: float = Field(gt=0, description="Weight must be > 0")

class SleepEvent(BaseEvent):
    event_type: EventType = EventType.SLEEP

class WakeEvent(BaseEvent):
    event_type: EventType = EventType.WAKE

class HydrationEvent(BaseEvent):
    event_type: EventType = EventType.HYDRATION
    window_name: str
    ml: int = Field(ge=0, description="ml >= 0, 0 = missed")

class MealEvent(BaseEvent):
    event_type: EventType = EventType.MEAL
    meal_name: str
    kcal: int = Field(ge=0, description="kcal >= 0, 0 = missed")

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
    workout: WorkoutType
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
    schema_version: str = "1.0.0"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ============================================================================
# IN-MEMORY STORE (Replace with DB in production)
# ============================================================================

class DataStore:
    def __init__(self):
        self.days: Dict[str, CanonicalDay] = {}
        self.events: List[BaseEvent] = []
        self.day0_locked: bool = False
        
    def get_day(self, date: str) -> Optional[CanonicalDay]:
        return self.days.get(date)
    
    def save_day(self, day: CanonicalDay):
        # Day 0 immutability check
        if day.day_index == 0 and self.day0_locked and date in self.days:
            raise HTTPException(400, "Day 0 is locked and immutable")
        self.days[date] = day
        
    def lock_day0(self):
        self.day0_locked = True
    
    def add_event(self, event: BaseEvent):
        self.events.append(event)

store = DataStore()

# ============================================================================
# STATE MACHINE VALIDATOR
# ============================================================================

class StateMachine:
    @staticmethod
    def validate_sleep_wake(day: CanonicalDay, event: BaseEvent) -> None:
        """Enforce Sleep → Wake ordering"""
        if isinstance(event, WakeEvent):
            if day.logged.sleep_time is None:
                raise HTTPException(
                    400, 
                    "Cannot log wake time before sleep time (state violation)"
                )
            if event.timestamp <= day.logged.sleep_time:
                raise HTTPException(
                    400,
                    "Wake time must be after sleep time"
                )
    
    @staticmethod
    def validate_weight_once_per_day(day: CanonicalDay, event: WeightEvent) -> None:
        """Weight can only be logged once per day"""
        if day.logged.weight is not None:
            raise HTTPException(
                400,
                f"Weight already logged for {day.date}: {day.logged.weight} kg"
            )
    
    @staticmethod
    def validate_duplicate_hydration(day: CanonicalDay, event: HydrationEvent) -> None:
        """Each hydration window can only be logged once"""
        existing = [h for h in day.logged.hydration if h['window_name'] == event.window_name]
        if existing:
            raise HTTPException(
                400,
                f"Hydration window '{event.window_name}' already logged"
            )

# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    @staticmethod
    def calculate(day: CanonicalDay) -> CanonicalDay:
        """Calculate all derived metrics from logged data only"""
        logged = day.logged
        planned = day.planned
        metrics = DerivedMetrics()
        flags = ConfidenceFlags()
        
        # Sleep duration (derived, never inferred)
        if logged.sleep_time and logged.wake_time:
            delta = logged.wake_time - logged.sleep_time
            metrics.sleep_duration_min = int(delta.total_seconds() / 60)
            flags.sleep_complete = True
        
        # Hydration adherence
        total_logged = sum(h.get('ml', 0) for h in logged.hydration)
        if planned.hydration_ml > 0:
            metrics.hydration_adherence_pct = int((total_logged / planned.hydration_ml) * 100)
        flags.hydration_complete = len(logged.hydration) == len(planned.hydration_windows)
        
        # Calorie surplus
        total_calories = sum(m.get('kcal', 0) for m in logged.meals)
        metrics.calorie_surplus = total_calories - planned.meals_kcal
        flags.nutrition_complete = len(logged.meals) == len(planned.meals)
        
        # Weight
        flags.weight_logged = logged.weight is not None and logged.weight > 0
        
        # Workout duration
        if logged.workout_sets:
            timestamps = [
                datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) 
                if isinstance(s['timestamp'], str) 
                else s['timestamp']
                for s in logged.workout_sets
            ]
            if len(timestamps) > 1:
                duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
                metrics.workout_duration_min = int(duration)
        
        planned_sets_count = sum(s['sets'] for s in planned.workout_sets)
        flags.workout_complete = len(logged.workout_sets) >= planned_sets_count
        
        # Data quality score
        flag_values = [
            flags.weight_logged,
            flags.sleep_complete,
            flags.hydration_complete,
            flags.nutrition_complete,
            flags.workout_complete
        ]
        quality_score = sum(flag_values) / len(flag_values)
        
        day.derived_metrics = metrics
        day.confidence_flags = flags
        day.data_quality_score = round(quality_score, 2)
        
        return day

# ============================================================================
# EVENT INGESTION API
# ============================================================================

@app.post("/events/ingest")
async def ingest_event(event: Dict[str, Any]):
    """
    Ingest a single event with strict validation
    
    QA Checklist:
    - Schema validation
    - State machine rules
    - No inference
    - No hallucination
    - Timestamped
    """
    try:
        event_type = EventType(event['event_type'])
        date = event['date']
        
        # Get or create canonical day
        day = store.get_day(date)
        if not day:
            # Initialize new day (would come from planning service)
            day = initialize_day(date, len(store.days))
        
        # Parse event based on type
        parsed_event = None
        if event_type == EventType.WEIGHT:
            parsed_event = WeightEvent(**event)
            StateMachine.validate_weight_once_per_day(day, parsed_event)
            day.logged.weight = parsed_event.weight_kg
            
        elif event_type == EventType.SLEEP:
            parsed_event = SleepEvent(**event)
            day.logged.sleep_time = parsed_event.timestamp
            
        elif event_type == EventType.WAKE:
            parsed_event = WakeEvent(**event)
            StateMachine.validate_sleep_wake(day, parsed_event)
            day.logged.wake_time = parsed_event.timestamp
            
        elif event_type == EventType.HYDRATION:
            parsed_event = HydrationEvent(**event)
            StateMachine.validate_duplicate_hydration(day, parsed_event)
            day.logged.hydration.append({
                'window_name': parsed_event.window_name,
                'ml': parsed_event.ml,
                'logged_at': parsed_event.timestamp.isoformat() if parsed_event.ml > 0 else None
            })
            
        elif event_type == EventType.MEAL:
            parsed_event = MealEvent(**event)
            day.logged.meals.append({
                'meal_name': parsed_event.meal_name,
                'kcal': parsed_event.kcal,
                'logged_at': parsed_event.timestamp.isoformat() if parsed_event.kcal > 0 else None
            })
            
        elif event_type == EventType.WORKOUT_SET:
            parsed_event = WorkoutSetEvent(**event)
            day.logged.workout_sets.append({
                'exercise': parsed_event.exercise,
                'set_number': parsed_event.set_number,
                'planned_reps': parsed_event.planned_reps,
                'actual_reps': parsed_event.actual_reps,
                'timestamp': parsed_event.timestamp.isoformat()
            })
        
        # Store event
        store.add_event(parsed_event)
        
        # Recalculate metrics (derived only, no inference)
        day = MetricsCalculator.calculate(day)
        
        # Save canonical day
        store.save_day(day)
        
        return {
            "status": "success",
            "event_type": event_type.value,
            "date": date,
            "updated_metrics": day.derived_metrics.dict(),
            "data_quality_score": day.data_quality_score
        }
        
    except ValueError as e:
        raise HTTPException(400, f"Invalid event data: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Error processing event: {str(e)}")

# ============================================================================
# DAY RETRIEVAL
# ============================================================================

@app.get("/day/{date}")
async def get_day(date: str):
    """Retrieve canonical day object"""
    day = store.get_day(date)
    if not day:
        raise HTTPException(404, f"Day {date} not found")
    return day

@app.get("/days/range")
async def get_days_range(start_date: str, end_date: str):
    """Retrieve multiple days"""
    result = []
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        day = store.get_day(date_str)
        if day:
            result.append(day)
        current += timedelta(days=1)
    
    return result

# ============================================================================
# EXPORT (Ground Truth Only)
# ============================================================================

@app.get("/export/csv")
async def export_csv():
    """
    Export CSV with ONLY logged ground truth
    
    No inference, no filling, no assumptions
    """
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'date', 'day_index', 'weight_kg', 'sleep_time', 'wake_time',
        'sleep_duration_min', 'total_kcal', 'calorie_surplus',
        'hydration_ml', 'hydration_adherence_pct',
        'workout_duration_min', 'data_quality_score'
    ])
    
    # Data rows (sorted by date)
    for date in sorted(store.days.keys()):
        day = store.days[date]
        writer.writerow([
            day.date,
            day.day_index,
            day.logged.weight if day.logged.weight else '',  # null = empty
            day.logged.sleep_time.isoformat() if day.logged.sleep_time else '',
            day.logged.wake_time.isoformat() if day.logged.wake_time else '',
            day.derived_metrics.sleep_duration_min if day.derived_metrics.sleep_duration_min else '',
            sum(m.get('kcal', 0) for m in day.logged.meals),
            day.derived_metrics.calorie_surplus,
            sum(h.get('ml', 0) for h in day.logged.hydration),
            day.derived_metrics.hydration_adherence_pct,
            day.derived_metrics.workout_duration_min if day.derived_metrics.workout_duration_min else '',
            day.data_quality_score
        ])
    
    return {"csv": output.getvalue()}

# ============================================================================
# RAG PIPELINE (Phase 2 - Placeholder)
# ============================================================================

class RAGOrchestrator:
    """
    RAG Pipeline for generating tomorrow's plan
    
    Architecture:
    1. Retrieve similar synthetic scenarios
    2. Retrieve similar historical self patterns
    3. Assemble context with strict tags
    4. Pass to LLM with guardrails
    5. Validate output schema
    """
    
    @staticmethod
    def generate_tomorrow_plan(today: CanonicalDay, history: List[CanonicalDay]):
        """
        Generate tomorrow's plan using RAG
        
        This is a placeholder - full implementation requires:
        - Vector DB (FAISS/Pinecone/Weaviate)
        - Synthetic data generator
        - LLM integration with guardrails
        """
        return {
            "status": "not_implemented",
            "message": "RAG pipeline requires vector DB and synthetic data"
        }

@app.post("/rag/generate-plan")
async def generate_plan(date: str):
    """Generate tomorrow's plan using RAG"""
    day = store.get_day(date)
    if not day:
        raise HTTPException(404, f"Day {date} not found")
    
    # Get last 7 days for context
    history = []
    current = datetime.fromisoformat(date)
    for i in range(1, 8):
        past_date = (current - timedelta(days=i)).strftime("%Y-%m-%d")
        past_day = store.get_day(past_date)
        if past_day:
            history.append(past_day)
    
    result = RAGOrchestrator.generate_tomorrow_plan(day, history)
    return result

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_day(date: str, day_index: int) -> CanonicalDay:
    """Initialize a new canonical day with default plan"""
    
    # Default Push workout plan
    planned = PlannedDay(
        workout=WorkoutType.PUSH,
        meals_kcal=3550,
        hydration_ml=3200,
        meals=[
            {"time": "08:00", "kcal": 600, "name": "Breakfast"},
            {"time": "11:00", "kcal": 550, "name": "Mid-Morning"},
            {"time": "14:00", "kcal": 700, "name": "Lunch"},
            {"time": "17:00", "kcal": 550, "name": "Pre-Workout"},
            {"time": "20:00", "kcal": 700, "name": "Post-Workout"},
            {"time": "22:30", "kcal": 450, "name": "Dinner"}
        ],
        hydration_windows=[
            {"name": "Morning", "ml": 800, "time": "06:00-12:00"},
            {"name": "Afternoon", "ml": 800, "time": "12:00-17:00"},
            {"name": "Evening", "ml": 800, "time": "17:00-21:00"},
            {"name": "Night", "ml": 800, "time": "21:00-23:00"}
        ],
        workout_sets=[
            {"exercise": "Bench Press", "sets": 4, "reps": 8},
            {"exercise": "Overhead Press", "sets": 3, "reps": 10},
            {"exercise": "Incline DB Press", "sets": 3, "reps": 12},
            {"exercise": "Lateral Raises", "sets": 3, "reps": 15},
            {"exercise": "Tricep Pushdowns", "sets": 3, "reps": 12}
        ]
    )
    
    return CanonicalDay(
        date=date,
        day_index=day_index,
        planned=planned,
        logged=LoggedData(),
        derived_metrics=DerivedMetrics(),
        confidence_flags=ConfidenceFlags(),
        data_quality_score=0.0
    )

@app.post("/admin/initialize-day")
async def admin_initialize_day(date: str, day_index: int):
    """Admin endpoint to initialize a new day"""
    if store.get_day(date):
        raise HTTPException(400, f"Day {date} already exists")
    
    day = initialize_day(date, day_index)
    store.save_day(day)
    return day

@app.post("/admin/lock-day0")
async def admin_lock_day0():
    """Lock Day 0 as immutable baseline"""
    store.lock_day0()
    return {"status": "Day 0 locked", "immutable": True}

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_days": len(store.days),
        "total_events": len(store.events),
        "day0_locked": store.day0_locked
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
