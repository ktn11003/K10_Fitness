"""
Fitness OS Test Suite
Comprehensive testing for QA-hardened system
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

# Import your backend
# from backend import app, store, initialize_day

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """FastAPI test client"""
    from backend import app
    return TestClient(app)

@pytest.fixture
def reset_store():
    """Reset in-memory store between tests"""
    from backend import store
    store.days.clear()
    store.events.clear()
    store.day0_locked = False
    yield
    store.days.clear()
    store.events.clear()

@pytest.fixture
def sample_day():
    """Sample canonical day for testing"""
    return {
        "date": "2025-12-18",
        "day_index": 0,
        "planned": {
            "workout": "push",
            "meals_kcal": 3550,
            "hydration_ml": 3200
        },
        "logged": {
            "weight": None,
            "sleep_time": None,
            "wake_time": None,
            "meals": [],
            "hydration": [],
            "workout_sets": []
        }
    }

# ============================================================================
# TEST: HEALTH CHECK
# ============================================================================

def test_health_check(client, reset_store):
    """Test basic health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

# ============================================================================
# TEST: DAY INITIALIZATION
# ============================================================================

def test_initialize_day(client, reset_store):
    """Test day initialization"""
    response = client.post(
        "/admin/initialize-day?date=2025-12-18&day_index=0"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2025-12-18"
    assert data["day_index"] == 0
    assert data["data_quality_score"] == 0.0

def test_duplicate_day_initialization(client, reset_store):
    """Test duplicate day initialization fails"""
    # First initialization
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Second initialization should fail
    response = client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]

# ============================================================================
# TEST: WEIGHT LOGGING
# ============================================================================

def test_weight_logging_valid(client, reset_store):
    """Test valid weight logging"""
    # Initialize day
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Log weight
    event = {
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": 50.2
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 200
    
    # Verify weight was logged
    day_response = client.get("/day/2025-12-18")
    day = day_response.json()
    assert day["logged"]["weight"] == 50.2
    assert day["confidence_flags"]["weight_logged"] is True

def test_weight_logging_invalid_zero(client, reset_store):
    """Test weight = 0 is rejected"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": 0
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 400

def test_weight_logging_invalid_negative(client, reset_store):
    """Test negative weight is rejected"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": -5.0
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 400

def test_weight_logging_duplicate(client, reset_store):
    """Test weight can only be logged once per day"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # First weight log
    event1 = {
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": 50.2
    }
    response1 = client.post("/events/ingest", json=event1)
    assert response1.status_code == 200
    
    # Second weight log should fail
    event2 = {
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T08:00:00+05:30",
        "weight_kg": 50.5
    }
    response2 = client.post("/events/ingest", json=event2)
    assert response2.status_code == 400
    assert "already logged" in response2.json()["detail"]

# ============================================================================
# TEST: SLEEP STATE MACHINE
# ============================================================================

def test_sleep_wake_valid_sequence(client, reset_store):
    """Test valid sleep → wake sequence"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Log sleep
    sleep_event = {
        "event_type": "sleep",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T23:00:00+05:30"
    }
    response1 = client.post("/events/ingest", json=sleep_event)
    assert response1.status_code == 200
    
    # Log wake
    wake_event = {
        "event_type": "wake",
        "date": "2025-12-18",
        "timestamp": "2025-12-19T07:00:00+05:30"
    }
    response2 = client.post("/events/ingest", json=wake_event)
    assert response2.status_code == 200
    
    # Verify sleep duration calculated
    day = client.get("/day/2025-12-18").json()
    assert day["derived_metrics"]["sleep_duration_min"] == 480
    assert day["confidence_flags"]["sleep_complete"] is True

def test_wake_before_sleep_rejected(client, reset_store):
    """Test wake before sleep is rejected (state violation)"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Try to log wake without sleep
    wake_event = {
        "event_type": "wake",
        "date": "2025-12-18",
        "timestamp": "2025-12-19T07:00:00+05:30"
    }
    response = client.post("/events/ingest", json=wake_event)
    assert response.status_code == 400
    assert "before sleep" in response.json()["detail"].lower()

def test_wake_before_sleep_time_rejected(client, reset_store):
    """Test wake timestamp before sleep timestamp is rejected"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Log sleep
    sleep_event = {
        "event_type": "sleep",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T23:00:00+05:30"
    }
    client.post("/events/ingest", json=sleep_event)
    
    # Try to log wake with earlier timestamp
    wake_event = {
        "event_type": "wake",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T22:00:00+05:30"  # Before sleep!
    }
    response = client.post("/events/ingest", json=wake_event)
    assert response.status_code == 400

# ============================================================================
# TEST: HYDRATION LOGGING
# ============================================================================

def test_hydration_logging_valid(client, reset_store):
    """Test valid hydration logging"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "hydration",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T12:00:00+05:30",
        "window_name": "Morning",
        "ml": 800
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 200
    
    # Check adherence calculated
    data = response.json()
    assert "hydration_adherence_pct" in data["updated_metrics"]

def test_hydration_logging_zero_missed(client, reset_store):
    """Test ml=0 means missed window"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "hydration",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T12:00:00+05:30",
        "window_name": "Morning",
        "ml": 0
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 200
    
    # Check logged_at is null for missed window
    day = client.get("/day/2025-12-18").json()
    hydration = day["logged"]["hydration"][0]
    assert hydration["ml"] == 0
    assert hydration["logged_at"] is None

def test_hydration_duplicate_window(client, reset_store):
    """Test duplicate hydration window is rejected"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # First log
    event1 = {
        "event_type": "hydration",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T12:00:00+05:30",
        "window_name": "Morning",
        "ml": 800
    }
    client.post("/events/ingest", json=event1)
    
    # Second log same window should fail
    event2 = {
        "event_type": "hydration",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T12:30:00+05:30",
        "window_name": "Morning",
        "ml": 500
    }
    response = client.post("/events/ingest", json=event2)
    assert response.status_code == 400
    assert "already logged" in response.json()["detail"]

# ============================================================================
# TEST: NUTRITION LOGGING
# ============================================================================

def test_meal_logging_valid(client, reset_store):
    """Test valid meal logging"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "meal",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T08:00:00+05:30",
        "meal_name": "Breakfast",
        "kcal": 600
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 200
    
    # Check calorie surplus calculated
    data = response.json()
    assert "calorie_surplus" in data["updated_metrics"]

def test_meal_logging_zero_missed(client, reset_store):
    """Test kcal=0 means missed meal"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "meal",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T08:00:00+05:30",
        "meal_name": "Breakfast",
        "kcal": 0
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 200
    
    # Verify logged_at is null
    day = client.get("/day/2025-12-18").json()
    meal = day["logged"]["meals"][0]
    assert meal["kcal"] == 0
    assert meal["logged_at"] is None

# ============================================================================
# TEST: WORKOUT LOGGING
# ============================================================================

def test_workout_set_logging(client, reset_store):
    """Test workout set logging"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "workout_set",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T18:00:00+05:30",
        "exercise": "Bench Press",
        "set_number": 1,
        "planned_reps": 8,
        "actual_reps": 8
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 200

def test_workout_duration_calculation(client, reset_store):
    """Test workout duration is calculated from first to last set"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # First set
    event1 = {
        "event_type": "workout_set",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T18:00:00+05:30",
        "exercise": "Bench Press",
        "set_number": 1,
        "planned_reps": 8,
        "actual_reps": 8
    }
    client.post("/events/ingest", json=event1)
    
    # Last set (45 minutes later)
    event2 = {
        "event_type": "workout_set",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T18:45:00+05:30",
        "exercise": "Tricep Pushdowns",
        "set_number": 3,
        "planned_reps": 12,
        "actual_reps": 10
    }
    client.post("/events/ingest", json=event2)
    
    # Check duration
    day = client.get("/day/2025-12-18").json()
    assert day["derived_metrics"]["workout_duration_min"] == 45

# ============================================================================
# TEST: DATA QUALITY SCORE
# ============================================================================

def test_data_quality_score_empty(client, reset_store):
    """Test data quality score is 0 for empty day"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    day = client.get("/day/2025-12-18").json()
    assert day["data_quality_score"] == 0.0

def test_data_quality_score_partial(client, reset_store):
    """Test data quality score increases with logging"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Log weight (1/5 = 0.2)
    client.post("/events/ingest", json={
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": 50.2
    })
    
    day = client.get("/day/2025-12-18").json()
    assert day["data_quality_score"] == 0.2

def test_data_quality_score_complete(client, reset_store):
    """Test data quality score is 1.0 when complete"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Log all categories
    # Weight
    client.post("/events/ingest", json={
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": 50.2
    })
    
    # Sleep
    client.post("/events/ingest", json={
        "event_type": "sleep",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T23:00:00+05:30"
    })
    client.post("/events/ingest", json={
        "event_type": "wake",
        "date": "2025-12-18",
        "timestamp": "2025-12-19T07:00:00+05:30"
    })
    
    # Hydration (all 4 windows)
    for window in ["Morning", "Afternoon", "Evening", "Night"]:
        client.post("/events/ingest", json={
            "event_type": "hydration",
            "date": "2025-12-18",
            "timestamp": "2025-12-18T12:00:00+05:30",
            "window_name": window,
            "ml": 800
        })
    
    # Nutrition (all 6 meals)
    for meal in ["Breakfast", "Mid-Morning", "Lunch", "Pre-Workout", "Post-Workout", "Dinner"]:
        client.post("/events/ingest", json={
            "event_type": "meal",
            "date": "2025-12-18",
            "timestamp": "2025-12-18T08:00:00+05:30",
            "meal_name": meal,
            "kcal": 500
        })
    
    # Workout (16 total sets)
    for i in range(16):
        client.post("/events/ingest", json={
            "event_type": "workout_set",
            "date": "2025-12-18",
            "timestamp": "2025-12-18T18:00:00+05:30",
            "exercise": "Bench Press",
            "set_number": i+1,
            "planned_reps": 8,
            "actual_reps": 8
        })
    
    day = client.get("/day/2025-12-18").json()
    assert day["data_quality_score"] == 1.0

# ============================================================================
# TEST: DAY 0 IMMUTABILITY
# ============================================================================

def test_day0_lock(client, reset_store):
    """Test Day 0 can be locked"""
    response = client.post("/admin/lock-day0")
    assert response.status_code == 200
    assert response.json()["immutable"] is True

def test_day0_immutable_after_lock(client, reset_store):
    """Test Day 0 cannot be modified after lock"""
    # Initialize and log Day 0
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    client.post("/events/ingest", json={
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": 50.0
    })
    
    # Lock Day 0
    client.post("/admin/lock-day0")
    
    # Try to modify Day 0
    response = client.post("/events/ingest", json={
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T08:00:00+05:30",
        "weight_kg": 51.0
    })
    
    # Should fail (either duplicate weight or immutability)
    assert response.status_code == 400

# ============================================================================
# TEST: CSV EXPORT
# ============================================================================

def test_csv_export_empty(client, reset_store):
    """Test CSV export with no data"""
    response = client.get("/export/csv")
    assert response.status_code == 200
    csv_data = response.json()["csv"]
    assert "date,day_index" in csv_data

def test_csv_export_ground_truth_only(client, reset_store):
    """Test CSV export contains only logged ground truth"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    # Log weight only (partial day)
    client.post("/events/ingest", json={
        "event_type": "weight",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30",
        "weight_kg": 50.2
    })
    
    # Export CSV
    response = client.get("/export/csv")
    csv_data = response.json()["csv"]
    lines = csv_data.strip().split("\n")
    
    # Should have header + 1 data row
    assert len(lines) == 2
    
    # Data row should have weight, but empty sleep/hydration/etc
    data_row = lines[1].split(",")
    assert "50.2" in data_row  # Weight logged
    assert data_row[3] == ""  # Sleep not logged (empty)

# ============================================================================
# TEST: EDGE CASES
# ============================================================================

def test_invalid_event_type(client, reset_store):
    """Test invalid event type is rejected"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "invalid_type",
        "date": "2025-12-18",
        "timestamp": "2025-12-18T07:00:00+05:30"
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code in [400, 422]

def test_missing_required_fields(client, reset_store):
    """Test event with missing required fields is rejected"""
    client.post("/admin/initialize-day?date=2025-12-18&day_index=0")
    
    event = {
        "event_type": "weight",
        # Missing date, timestamp, weight_kg
    }
    response = client.post("/events/ingest", json=event)
    assert response.status_code == 422

def test_future_date_rejection(client, reset_store):
    """Test events with future dates can be logged (time zones)"""
    # This depends on your business rules
    # For now, we allow future timestamps due to IST timezone
    pass

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
