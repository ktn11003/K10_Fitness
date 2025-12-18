"""
Vector Database & RAG Orchestration
Implements retrieval-augmented generation with strict guardrails
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime, timedelta

# ============================================================================
# EMBEDDING MODEL
# ============================================================================

class EmbeddingService:
    """Generate embeddings for fitness data"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Using sentence-transformers for semantic similarity
        Smaller model (384 dim) for efficiency
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
    
    def embed_day(self, day_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert day metrics to embedding
        
        Creates a text representation of the day for semantic search
        """
        # Create text representation (NO raw logs, only derived metrics)
        text_parts = [
            f"Day {day_data.get('day_index', 0)}",
            f"workout: {day_data.get('workout_type', 'unknown')}",
            f"sleep: {day_data.get('sleep_duration_min', 0)} minutes",
            f"hydration: {day_data.get('hydration_adherence_pct', 0)}%",
            f"calorie surplus: {day_data.get('calorie_surplus', 0)} kcal",
            f"quality: {day_data.get('data_quality_score', 0)}"
        ]
        
        # Add characteristics if available (synthetic data)
        if 'characteristics' in day_data:
            chars = day_data['characteristics']
            text_parts.extend([
                f"metabolism: {chars.get('metabolism', 'unknown')}",
                f"adherence: {chars.get('adherence', 'unknown')}",
                f"recovery: {chars.get('recovery', 'unknown')}"
            ])
        
        text = ", ".join(text_parts)
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_weekly_summary(self, summary: Dict[str, Any]) -> np.ndarray:
        """Embed weekly summary data"""
        text_parts = [
            f"Week {summary.get('week', 0)}",
            f"weight gain: {summary.get('weight_gain', 0):.2f} kg",
            f"sleep: {summary.get('avg_sleep_hours', 0):.1f} hours",
            f"adherence: {summary.get('nutrition_adherence', 0):.1f}%"
        ]
        
        # Add recommendation context
        if 'recommendation' in summary:
            rec = summary['recommendation']
            text_parts.append(f"recommendation: {rec.get('action', 'unknown')}")
        
        text = ", ".join(text_parts)
        return self.model.encode(text, convert_to_numpy=True)

# ============================================================================
# VECTOR DATABASE
# ============================================================================

class VectorDatabase:
    """
    FAISS-based vector database with namespace separation
    
    Enforces strict separation between synthetic and user data
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        
        # Separate indices for synthetic vs user data
        self.synthetic_index = faiss.IndexFlatL2(dimension)
        self.user_index = faiss.IndexFlatL2(dimension)
        
        # Metadata stores (parallel to FAISS indices)
        self.synthetic_metadata: List[Dict[str, Any]] = []
        self.user_metadata: List[Dict[str, Any]] = []
        
        print(f"Initialized VectorDB with dimension {dimension}")
    
    def add_synthetic_data(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ):
        """
        Add synthetic data to separate namespace
        
        Args:
            embeddings: (N, dimension) array
            metadata: List of N metadata dicts
        """
        assert embeddings.shape[0] == len(metadata), "Mismatch in embeddings/metadata"
        assert embeddings.shape[1] == self.dimension, "Dimension mismatch"
        
        # Add to synthetic index
        self.synthetic_index.add(embeddings.astype('float32'))
        self.synthetic_metadata.extend(metadata)
        
        print(f"Added {len(metadata)} synthetic vectors")
    
    def add_user_data(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ):
        """Add user historical data to separate namespace"""
        assert embeddings.shape[0] == len(metadata), "Mismatch in embeddings/metadata"
        assert embeddings.shape[1] == self.dimension, "Dimension mismatch"
        
        # Add to user index
        self.user_index.add(embeddings.astype('float32'))
        self.user_metadata.extend(metadata)
        
        print(f"Added {len(metadata)} user vectors")
    
    def search_synthetic(
        self, 
        query_vector: np.ndarray, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search synthetic data only
        
        Returns k most similar synthetic scenarios
        """
        if self.synthetic_index.ntotal == 0:
            return []
        
        query = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.synthetic_index.search(query, min(k, self.synthetic_index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.synthetic_metadata):
                result = self.synthetic_metadata[idx].copy()
                result['distance'] = float(dist)
                result['similarity'] = 1.0 / (1.0 + float(dist))  # Convert to similarity
                results.append(result)
        
        return results
    
    def search_user_history(
        self, 
        query_vector: np.ndarray, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search user historical data only
        
        Returns k most similar historical patterns
        """
        if self.user_index.ntotal == 0:
            return []
        
        query = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.user_index.search(query, min(k, self.user_index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.user_metadata):
                result = self.user_metadata[idx].copy()
                result['distance'] = float(dist)
                result['similarity'] = 1.0 / (1.0 + float(dist))
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        return {
            "synthetic_vectors": self.synthetic_index.ntotal,
            "user_vectors": self.user_index.ntotal,
            "total_vectors": self.synthetic_index.ntotal + self.user_index.ntotal
        }

# ============================================================================
# CONTEXT ASSEMBLER
# ============================================================================

@dataclass
class RetrievalContext:
    """Structured context for LLM"""
    current_day: Dict[str, Any]
    last_7_days: List[Dict[str, Any]]
    synthetic_matches: List[Dict[str, Any]]
    historical_matches: List[Dict[str, Any]]
    data_quality_flags: Dict[str, bool]

class ContextAssembler:
    """
    Assemble retrieval context with strict tagging
    
    Ensures LLM knows what's synthetic vs real
    """
    
    @staticmethod
    def assemble(
        current_day: Dict[str, Any],
        last_7_days: List[Dict[str, Any]],
        synthetic_matches: List[Dict[str, Any]],
        historical_matches: List[Dict[str, Any]]
    ) -> RetrievalContext:
        """Assemble all context components"""
        
        # Extract data quality flags
        flags = {
            "weight_logged": current_day.get("confidence_flags", {}).get("weight_logged", False),
            "sleep_complete": current_day.get("confidence_flags", {}).get("sleep_complete", False),
            "hydration_complete": current_day.get("confidence_flags", {}).get("hydration_complete", False),
            "nutrition_complete": current_day.get("confidence_flags", {}).get("nutrition_complete", False),
            "workout_complete": current_day.get("confidence_flags", {}).get("workout_complete", False)
        }
        
        return RetrievalContext(
            current_day=current_day,
            last_7_days=last_7_days,
            synthetic_matches=synthetic_matches,
            historical_matches=historical_matches,
            data_quality_flags=flags
        )
    
    @staticmethod
    def format_for_llm(context: RetrievalContext) -> str:
        """
        Format context with explicit tags for LLM
        
        LLM MUST distinguish synthetic from real data
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are a fitness advisor that uses ONLY retrieved context to make recommendations.

STRICT RULES:
1. You CANNOT invent data
2. You CANNOT override logged values
3. You CANNOT suggest violating constraints (skip sleep/meals)
4. All reasoning must reference retrieved context
5. Declare confidence level for all recommendations

""")
        
        # Current day state
        prompt_parts.append("[CURRENT_DAY_STATE]")
        prompt_parts.append(f"Date: {context.current_day.get('date')}")
        prompt_parts.append(f"Day Index: {context.current_day.get('day_index')}")
        prompt_parts.append(f"Data Quality: {context.current_day.get('data_quality_score', 0):.2f}")
        
        metrics = context.current_day.get('derived_metrics', {})
        prompt_parts.append(f"Sleep: {metrics.get('sleep_duration_min')} min")
        prompt_parts.append(f"Hydration: {metrics.get('hydration_adherence_pct')}%")
        prompt_parts.append(f"Calorie Surplus: {metrics.get('calorie_surplus')} kcal")
        prompt_parts.append(f"Workout Duration: {metrics.get('workout_duration_min')} min")
        prompt_parts.append("")
        
        # 7-day trend
        if context.last_7_days:
            prompt_parts.append("[LAST_7_DAYS_TREND]")
            for day in context.last_7_days[-7:]:
                metrics = day.get('derived_metrics', {})
                prompt_parts.append(
                    f"Day {day.get('day_index')}: "
                    f"Sleep={metrics.get('sleep_duration_min')}min, "
                    f"Hydration={metrics.get('hydration_adherence_pct')}%, "
                    f"Surplus={metrics.get('calorie_surplus')}kcal"
                )
            prompt_parts.append("")
        
        # Synthetic context
        if context.synthetic_matches:
            prompt_parts.append("[SYNTHETIC_CONTEXT]")
            prompt_parts.append("These are simulated training scenarios with known outcomes:")
            for i, match in enumerate(context.synthetic_matches[:3], 1):
                prompt_parts.append(f"\nSynthetic Match {i} (similarity: {match.get('similarity', 0):.3f}):")
                chars = match.get('characteristics', {})
                prompt_parts.append(f"  Profile: {chars}")
                prompt_parts.append(f"  Metrics: sleep={match.get('sleep_duration_min')}min, "
                                  f"hydration={match.get('hydration_adherence_pct')}%, "
                                  f"surplus={match.get('calorie_surplus')}kcal")
                if 'recommendation' in match:
                    rec = match['recommendation']
                    prompt_parts.append(f"  Outcome: {rec.get('action')} - {rec.get('reason')}")
            prompt_parts.append("")
        
        # Historical self patterns
        if context.historical_matches:
            prompt_parts.append("[HISTORICAL_SELF_CONTEXT]")
            prompt_parts.append("Your past similar days:")
            for i, match in enumerate(context.historical_matches[:3], 1):
                prompt_parts.append(f"\nHistorical Match {i} (similarity: {match.get('similarity', 0):.3f}):")
                prompt_parts.append(f"  Day {match.get('day_index')}: "
                                  f"sleep={match.get('sleep_duration_min')}min, "
                                  f"hydration={match.get('hydration_adherence_pct')}%, "
                                  f"surplus={match.get('calorie_surplus')}kcal")
            prompt_parts.append("")
        
        # Request
        prompt_parts.append("""[TASK]
Based ONLY on the above context, provide recommendations for tomorrow's training.

Output JSON format:
{
  "recommendations": [
    {
      "type": "workout|nutrition|recovery",
      "action": "specific action",
      "reason": "reasoning referencing retrieved context",
      "confidence": "low|medium|high"
    }
  ],
  "overall_confidence": 0.0-1.0
}
""")
        
        return "\n".join(prompt_parts)

# ============================================================================
# OUTPUT VALIDATOR
# ============================================================================

class OutputValidator:
    """
    Validate LLM outputs against guardrails
    
    Rejects outputs that violate constraints
    """
    
    @staticmethod
    def validate(
        llm_output: Dict[str, Any],
        context: RetrievalContext
    ) -> Tuple[bool, List[str]]:
        """
        Validate LLM output
        
        Returns: (is_valid, list_of_violations)
        """
        violations = []
        
        # Check required schema
        if 'recommendations' not in llm_output:
            violations.append("Missing 'recommendations' field")
            return False, violations
        
        if 'overall_confidence' not in llm_output:
            violations.append("Missing 'overall_confidence' field")
        
        # Check each recommendation
        for i, rec in enumerate(llm_output.get('recommendations', [])):
            # Required fields
            required = ['type', 'action', 'reason', 'confidence']
            for field in required:
                if field not in rec:
                    violations.append(f"Recommendation {i}: missing '{field}'")
            
            # Check for forbidden suggestions
            action = rec.get('action', '').lower()
            forbidden = ['skip sleep', 'skip meal', 'skip rest', 'ignore recovery']
            if any(word in action for word in forbidden):
                violations.append(
                    f"Recommendation {i}: suggests violating constraint: '{action}'"
                )
            
            # Check for invented metrics
            reason = rec.get('reason', '').lower()
            # This is a simple check - in production, use NER to extract metrics
            # and verify they exist in context
            
        # Check confidence bounds
        conf = llm_output.get('overall_confidence', 0)
        if not (0 <= conf <= 1):
            violations.append(f"overall_confidence out of bounds: {conf}")
        
        is_valid = len(violations) == 0
        return is_valid, violations

# ============================================================================
# RAG ORCHESTRATOR
# ============================================================================

class RAGOrchestrator:
    """
    Main RAG pipeline orchestrator
    
    Coordinates: Retrieval → Context Assembly → LLM → Validation
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_service: EmbeddingService
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
    
    def generate_recommendations(
        self,
        current_day: Dict[str, Any],
        last_7_days: List[Dict[str, Any]],
        k_synthetic: int = 5,
        k_historical: int = 3
    ) -> Dict[str, Any]:
        """
        Generate recommendations using RAG pipeline
        
        Steps:
        1. Embed current state
        2. Retrieve similar synthetic scenarios
        3. Retrieve similar historical patterns
        4. Assemble context
        5. Query LLM (with guardrails)
        6. Validate output
        """
        
        # Step 1: Embed current state
        current_embedding = self.embedding_service.embed_day(current_day)
        
        # Step 2: Retrieve synthetic matches
        synthetic_matches = self.vector_db.search_synthetic(
            current_embedding, 
            k=k_synthetic
        )
        
        # Step 3: Retrieve historical matches
        historical_matches = self.vector_db.search_user_history(
            current_embedding,
            k=k_historical
        )
        
        # Step 4: Assemble context
        context = ContextAssembler.assemble(
            current_day=current_day,
            last_7_days=last_7_days,
            synthetic_matches=synthetic_matches,
            historical_matches=historical_matches
        )
        
        # Step 5: Format for LLM
        llm_prompt = ContextAssembler.format_for_llm(context)
        
        # Step 6: Call LLM (placeholder - integrate with Anthropic API)
        llm_output = self._call_llm(llm_prompt)
        
        # Step 7: Validate output
        is_valid, violations = OutputValidator.validate(llm_output, context)
        
        if not is_valid:
            return {
                "status": "validation_failed",
                "violations": violations,
                "raw_output": llm_output
            }
        
        # Step 8: Return validated recommendations
        return {
            "status": "success",
            "recommendations": llm_output['recommendations'],
            "overall_confidence": llm_output['overall_confidence'],
            "retrieval_stats": {
                "synthetic_matches": len(synthetic_matches),
                "historical_matches": len(historical_matches),
                "top_synthetic_similarity": synthetic_matches[0]['similarity'] if synthetic_matches else 0,
                "top_historical_similarity": historical_matches[0]['similarity'] if historical_matches else 0
            }
        }
    
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM with strict guardrails
        
        In production: integrate with Anthropic API
        For now: return structured placeholder
        """
        # Placeholder response
        return {
            "recommendations": [
                {
                    "type": "recovery",
                    "action": "Prioritize 8+ hours sleep tonight",
                    "reason": "Current sleep deficit detected (6.2h avg last 3 days). Synthetic match #1 showed recovery improvement with adequate sleep.",
                    "confidence": "high"
                },
                {
                    "type": "nutrition",
                    "action": "Maintain current calorie target",
                    "reason": "Calorie surplus on track (+180 kcal avg). Historical pattern shows this leads to steady gains.",
                    "confidence": "high"
                },
                {
                    "type": "workout",
                    "action": "Reduce volume by 10% if fatigue persists",
                    "reason": "Low sleep + suboptimal hydration detected. Synthetic match #2 showed better outcomes with volume reduction under similar conditions.",
                    "confidence": "medium"
                }
            ],
            "overall_confidence": 0.78
        }

# ============================================================================
# INITIALIZATION HELPER
# ============================================================================

def initialize_rag_system(synthetic_data_path: str) -> RAGOrchestrator:
    """
    Initialize complete RAG system
    
    Args:
        synthetic_data_path: Path to synthetic_embeddings.json
    
    Returns:
        Configured RAG orchestrator
    """
    print("Initializing RAG system...")
    
    # Initialize components
    embedding_service = EmbeddingService()
    vector_db = VectorDatabase(dimension=embedding_service.dimension)
    
    # Load and index synthetic data
    print(f"Loading synthetic data from {synthetic_data_path}")
    with open(synthetic_data_path, 'r') as f:
        synthetic_data = json.load(f)
    
    # Separate daily and weekly embeddings
    daily_data = [d for d in synthetic_data if d.get('type') != 'weekly_summary']
    weekly_data = [d for d in synthetic_data if d.get('type') == 'weekly_summary']
    
    # Generate embeddings
    print("Generating embeddings for synthetic data...")
    daily_embeddings = []
    daily_metadata = []
    
    for data in daily_data:
        emb = embedding_service.embed_day(data)
        daily_embeddings.append(emb)
        daily_metadata.append(data)
    
    weekly_embeddings = []
    weekly_metadata = []
    
    for data in weekly_data:
        emb = embedding_service.embed_weekly_summary(data)
        weekly_embeddings.append(emb)
        weekly_metadata.append(data)
    
    # Add to vector DB
    if daily_embeddings:
        vector_db.add_synthetic_data(
            np.array(daily_embeddings),
            daily_metadata
        )
    
    if weekly_embeddings:
        vector_db.add_synthetic_data(
            np.array(weekly_embeddings),
            weekly_metadata
        )
    
    print(f"RAG system initialized: {vector_db.get_stats()}")
    
    # Create orchestrator
    orchestrator = RAGOrchestrator(vector_db, embedding_service)
    
    return orchestrator

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # This would be called by the FastAPI backend
    
    # Initialize RAG system
    rag = initialize_rag_system("synthetic_embeddings.json")
    
    # Example: Generate recommendations for today
    current_day = {
        "date": "2025-12-18",
        "day_index": 5,
        "workout_type": "push",
        "sleep_duration_min": 370,
        "hydration_adherence_pct": 65,
        "calorie_surplus": 150,
        "data_quality_score": 0.8,
        "derived_metrics": {
            "sleep_duration_min": 370,
            "hydration_adherence_pct": 65,
            "calorie_surplus": 150
        },
        "confidence_flags": {
            "weight_logged": True,
            "sleep_complete": True,
            "hydration_complete": False,
            "nutrition_complete": True,
            "workout_complete": True
        }
    }
    
    last_7_days = []  # Would come from database
    
    # Generate recommendations
    result = rag.generate_recommendations(current_day, last_7_days)
    
    print("\n" + "="*60)
    print("RAG RECOMMENDATIONS")
    print("="*60)
    print(json.dumps(result, indent=2))  
