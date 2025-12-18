"""
Synthetic Fitness Data Generator
Generates realistic training journeys with known outcomes for RAG system

Purpose: Teach the system PATTERNS, not FACTS
"""

import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Literal
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# SYNTHETIC PROFILE DIMENSIONS
# ============================================================================

class Metabolism(Enum):
    FAST = "fast"      # Gains weight slowly, needs 3800+ kcal
    MEDIUM = "medium"  # Normal response, 3500 kcal
    SLOW = "slow"      # Gains easily, 3200 kcal

class Adherence(Enum):
    STRICT = "strict"    # 95%+ adherence
    PARTIAL = "partial"  # 70-90% adherence
    POOR = "poor"        # <70% adherence

class Recovery(Enum):
    HIGH = "high"      # Can train 6 days/week
    MEDIUM = "medium"  # Needs rest days
    LOW = "low"        # Needs extra recovery

class SleepDebt(Enum):
    NONE = "none"      # 7-9h consistently
    MILD = "mild"      # 6-7h
    SEVERE = "severe"  # <6h

class Hydration(Enum):
    OPTIMAL = "optimal"     # 90%+ adherence
    SUBOPTIMAL = "suboptimal"  # <80% adherence

class TrainingAge(Enum):
    BEGINNER = "beginner"        # <6 months
    INTERMEDIATE = "intermediate"  # 6mo-2yr

# ============================================================================
# SYNTHETIC PROFILE
# ============================================================================

@dataclass
class SyntheticProfile:
    """Represents a synthetic person's characteristics"""
    profile_id: str
    metabolism: Metabolism
    adherence: Adherence
    recovery: Recovery
    sleep_debt: SleepDebt
    hydration: Hydration
    training_age: TrainingAge
    starting_weight: float
    target_weight: float
    weeks: int = 12
    
    def get_label(self) -> str:
        return f"synthetic_{self.profile_id}"

# ============================================================================
# OUTCOME PREDICTOR (Ground Truth for Synthetic Data)
# ============================================================================

class OutcomePredictor:
    """
    Deterministic rules for synthetic outcomes
    
    These are KNOWN outcomes used to teach the RAG system
    """
    
    @staticmethod
    def predict_weekly_weight_gain(profile: SyntheticProfile, week: int) -> float:
        """
        Predict weekly weight gain based on profile
        
        Rules:
        - Optimal conditions: 0.5-0.8 kg/week
        - Suboptimal: 0.2-0.5 kg/week
        - Poor: 0-0.2 kg/week or stagnation
        """
        base_gain = 0.5  # kg/week baseline
        
        # Metabolism adjustment
        if profile.metabolism == Metabolism.FAST:
            base_gain *= 0.7
        elif profile.metabolism == Metabolism.SLOW:
            base_gain *= 1.3
        
        # Adherence impact (critical)
        if profile.adherence == Adherence.STRICT:
            adherence_mult = 1.0
        elif profile.adherence == Adherence.PARTIAL:
            adherence_mult = 0.6
        else:  # POOR
            adherence_mult = 0.2
        
        # Sleep debt impact
        if profile.sleep_debt == SleepDebt.SEVERE:
            adherence_mult *= 0.5
        elif profile.sleep_debt == SleepDebt.MILD:
            adherence_mult *= 0.8
        
        # Hydration impact
        if profile.hydration == Hydration.SUBOPTIMAL:
            adherence_mult *= 0.9
        
        # Recovery impact
        if profile.recovery == Recovery.LOW:
            adherence_mult *= 0.85
        
        # Diminishing returns over time
        time_decay = 1.0 - (week * 0.02)  # Slight decrease each week
        
        return base_gain * adherence_mult * time_decay
    
    @staticmethod
    def recommend_volume_adjustment(
        profile: SyntheticProfile, 
        week: int,
        weight_gain_last_week: float
    ) -> Dict[str, Any]:
        """
        Recommend training volume adjustments
        
        Returns recommendation with reasoning
        """
        expected_gain = OutcomePredictor.predict_weekly_weight_gain(profile, week)
        
        # If weight gain is much lower than expected
        if weight_gain_last_week < expected_gain * 0.5:
            # Check why
            if profile.sleep_debt == SleepDebt.SEVERE:
                return {
                    "action": "reduce_volume",
                    "amount": "10-15%",
                    "reason": "Severe sleep debt detected + low weight gain",
                    "confidence": "high"
                }
            elif profile.adherence == Adherence.POOR:
                return {
                    "action": "focus_on_adherence",
                    "amount": "0%",
                    "reason": "Low adherence is primary bottleneck",
                    "confidence": "high"
                }
            elif profile.hydration == Hydration.SUBOPTIMAL:
                return {
                    "action": "maintain_volume",
                    "amount": "0%",
                    "reason": "Low hydration affecting performance",
                    "confidence": "medium"
                }
        
        # If weight gain is on track
        elif expected_gain * 0.8 <= weight_gain_last_week <= expected_gain * 1.2:
            return {
                "action": "maintain_volume",
                "amount": "0%",
                "reason": "Weight gain on track, continue current plan",
                "confidence": "high"
            }
        
        # If weight gain is too fast (rare)
        elif weight_gain_last_week > expected_gain * 1.5:
            return {
                "action": "reduce_calories",
                "amount": "200-300 kcal",
                "reason": "Weight gain faster than target (minimize fat gain)",
                "confidence": "medium"
            }
        
        return {
            "action": "maintain_volume",
            "amount": "0%",
            "reason": "Default recommendation",
            "confidence": "low"
        }

# ============================================================================
# SYNTHETIC DAY GENERATOR
# ============================================================================

class SyntheticDayGenerator:
    """Generate realistic daily logs for a synthetic profile"""
    
    @staticmethod
    def generate_day(
        profile: SyntheticProfile,
        day_index: int,
        current_weight: float
    ) -> Dict[str, Any]:
        """
        Generate a complete synthetic day
        
        Includes realistic variance and mistakes
        """
        date = (datetime.now() - timedelta(days=365-day_index)).strftime("%Y-%m-%d")
        
        # Workout type (4-day split)
        workout_types = ["push", "pull", "legs", "rest"]
        workout = workout_types[day_index % 4]
        
        # Generate adherence-based logs
        adherence_roll = random.random()
        
        # Sleep (affected by sleep_debt)
        if profile.sleep_debt == SleepDebt.NONE:
            sleep_hours = random.uniform(7.5, 9)
        elif profile.sleep_debt == SleepDebt.MILD:
            sleep_hours = random.uniform(6, 7)
        else:  # SEVERE
            sleep_hours = random.uniform(4.5, 6)
        
        # Nutrition adherence
        target_kcal = 3550
        if profile.adherence == Adherence.STRICT:
            if adherence_roll > 0.05:  # 95% adherence
                actual_kcal = target_kcal + random.randint(-100, 100)
            else:
                actual_kcal = target_kcal - random.randint(400, 800)
        elif profile.adherence == Adherence.PARTIAL:
            if adherence_roll > 0.25:  # 75% adherence
                actual_kcal = target_kcal + random.randint(-200, 50)
            else:
                actual_kcal = target_kcal - random.randint(600, 1000)
        else:  # POOR
            if adherence_roll > 0.50:  # 50% adherence
                actual_kcal = target_kcal - random.randint(300, 700)
            else:
                actual_kcal = target_kcal - random.randint(800, 1500)
        
        # Hydration adherence
        target_ml = 3200
        if profile.hydration == Hydration.OPTIMAL:
            actual_ml = int(target_ml * random.uniform(0.9, 1.1))
        else:
            actual_ml = int(target_ml * random.uniform(0.5, 0.8))
        
        # Workout performance (affected by recovery, sleep, hydration)
        if workout != "rest":
            performance_factor = 1.0
            
            if profile.recovery == Recovery.LOW:
                performance_factor *= 0.9
            if sleep_hours < 6:
                performance_factor *= 0.85
            if actual_ml < target_ml * 0.7:
                performance_factor *= 0.9
            
            # Random variance
            performance_factor *= random.uniform(0.95, 1.05)
        else:
            performance_factor = None
        
        # Construct synthetic day
        synthetic_day = {
            "date": date,
            "day_index": day_index,
            "profile_id": profile.profile_id,
            "label": profile.get_label(),
            
            # Characteristics (for context)
            "characteristics": {
                "metabolism": profile.metabolism.value,
                "adherence": profile.adherence.value,
                "recovery": profile.recovery.value,
                "sleep_debt": profile.sleep_debt.value,
                "hydration": profile.hydration.value,
                "training_age": profile.training_age.value
            },
            
            # Logged metrics (what would be recorded)
            "logged": {
                "weight": round(current_weight, 1),
                "sleep_duration_min": int(sleep_hours * 60),
                "hydration_ml": actual_ml,
                "hydration_adherence_pct": int((actual_ml / target_ml) * 100),
                "calories_kcal": actual_kcal,
                "calorie_surplus": actual_kcal - target_kcal,
                "workout_type": workout,
                "performance_factor": performance_factor
            },
            
            # Derived metrics
            "derived": {
                "data_quality_score": 1.0,  # Synthetic = complete
                "sleep_adequate": sleep_hours >= 7,
                "nutrition_on_track": abs(actual_kcal - target_kcal) < 300,
                "hydration_adequate": actual_ml >= target_ml * 0.8
            }
        }
        
        return synthetic_day

# ============================================================================
# SYNTHETIC JOURNEY GENERATOR
# ============================================================================

class SyntheticJourneyGenerator:
    """Generate complete 12-week synthetic journeys"""
    
    @staticmethod
    def generate_journey(profile: SyntheticProfile) -> Dict[str, Any]:
        """
        Generate a complete synthetic journey with known outcomes
        """
        journey = {
            "journey_id": f"journey_{profile.profile_id}",
            "profile": asdict(profile),
            "days": [],
            "weekly_summaries": [],
            "outcome": {}
        }
        
        current_weight = profile.starting_weight
        
        for week in range(profile.weeks):
            week_days = []
            week_start_weight = current_weight
            
            # Generate 7 days
            for day in range(7):
                day_index = week * 7 + day
                
                synthetic_day = SyntheticDayGenerator.generate_day(
                    profile, day_index, current_weight
                )
                week_days.append(synthetic_day)
                journey["days"].append(synthetic_day)
            
            # Calculate weekly weight gain
            weekly_gain = OutcomePredictor.predict_weekly_weight_gain(profile, week)
            current_weight += weekly_gain
            
            # Generate weekly summary
            avg_sleep = sum(d["logged"]["sleep_duration_min"] for d in week_days) / 7 / 60
            avg_adherence = sum(
                1 if d["derived"]["nutrition_on_track"] else 0 
                for d in week_days
            ) / 7
            
            weekly_summary = {
                "week": week,
                "start_weight": week_start_weight,
                "end_weight": current_weight,
                "weight_gain": weekly_gain,
                "avg_sleep_hours": round(avg_sleep, 1),
                "nutrition_adherence": round(avg_adherence * 100, 1),
                "recommendation": OutcomePredictor.recommend_volume_adjustment(
                    profile, week, weekly_gain
                )
            }
            journey["weekly_summaries"].append(weekly_summary)
        
        # Final outcome
        journey["outcome"] = {
            "final_weight": current_weight,
            "total_gain": current_weight - profile.starting_weight,
            "target_gain": profile.target_weight - profile.starting_weight,
            "success": abs(current_weight - profile.target_weight) < 2.0,
            "avg_weekly_gain": (current_weight - profile.starting_weight) / profile.weeks
        }
        
        return journey

# ============================================================================
# BATCH GENERATOR
# ============================================================================

def generate_synthetic_dataset(num_profiles: int = 50) -> List[Dict[str, Any]]:
    """
    Generate a diverse set of synthetic journeys
    
    Returns list of journeys ready for vector embedding
    """
    profiles = []
    
    for i in range(num_profiles):
        # Random profile generation
        profile = SyntheticProfile(
            profile_id=f"S{i:03d}",
            metabolism=random.choice(list(Metabolism)),
            adherence=random.choice(list(Adherence)),
            recovery=random.choice(list(Recovery)),
            sleep_debt=random.choice(list(SleepDebt)),
            hydration=random.choice(list(Hydration)),
            training_age=random.choice(list(TrainingAge)),
            starting_weight=random.uniform(45, 55),
            target_weight=random.uniform(55, 65),
            weeks=12
        )
        profiles.append(profile)
    
    journeys = []
    for profile in profiles:
        journey = SyntheticJourneyGenerator.generate_journey(profile)
        journeys.append(journey)
        print(f"Generated journey {journey['journey_id']}: "
              f"{journey['outcome']['total_gain']:.1f} kg gained "
              f"(target: {journey['outcome']['target_gain']:.1f} kg)")
    
    return journeys

# ============================================================================
# EXPORT FOR VECTOR DB
# ============================================================================

def export_for_embedding(journeys: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert journeys to embedding-ready format
    
    Only includes derived metrics, NOT raw logs
    """
    embeddings = []
    
    for journey in journeys:
        # Daily embeddings
        for day in journey["days"]:
            embedding_payload = {
                "day_index": day["day_index"],
                "workout_type": day["logged"]["workout_type"],
                "sleep_duration_min": day["logged"]["sleep_duration_min"],
                "hydration_adherence_pct": day["logged"]["hydration_adherence_pct"],
                "calorie_surplus": day["logged"]["calorie_surplus"],
                "data_quality_score": day["derived"]["data_quality_score"],
                "label": "synthetic",
                "profile_id": day["profile_id"],
                
                # Context (for retrieval)
                "characteristics": day["characteristics"]
            }
            embeddings.append(embedding_payload)
        
        # Weekly embeddings
        for week_summary in journey["weekly_summaries"]:
            week_embedding = {
                "type": "weekly_summary",
                "week": week_summary["week"],
                "weight_gain": week_summary["weight_gain"],
                "avg_sleep_hours": week_summary["avg_sleep_hours"],
                "nutrition_adherence": week_summary["nutrition_adherence"],
                "recommendation": week_summary["recommendation"],
                "label": "synthetic",
                "profile_id": journey["profile"]["profile_id"]
            }
            embeddings.append(week_embedding)
    
    return embeddings

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Generating synthetic fitness journeys...")
    print("=" * 60)
    
    # Generate 50 diverse profiles
    journeys = generate_synthetic_dataset(num_profiles=50)
    
    # Export for vector embedding
    embeddings = export_for_embedding(journeys)
    
    # Save to JSON
    with open("synthetic_journeys.json", "w") as f:
        json.dump(journeys, f, indent=2)
    
    with open("synthetic_embeddings.json", "w") as f:
        json.dump(embeddings, f, indent=2)
    
    print("=" * 60)
    print(f"Generated {len(journeys)} synthetic journeys")
    print(f"Generated {len(embeddings)} embedding payloads")
    print(f"Saved to synthetic_journeys.json and synthetic_embeddings.json")
    
    # Print summary statistics
    successful = sum(1 for j in journeys if j["outcome"]["success"])
    print(f"\nSuccess rate: {successful}/{len(journeys)} "
          f"({successful/len(journeys)*100:.1f}%)")
    
    avg_gain = sum(j["outcome"]["total_gain"] for j in journeys) / len(journeys)
    print(f"Average weight gain: {avg_gain:.2f} kg")
