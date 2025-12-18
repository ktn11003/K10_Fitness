import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

# Load a lightweight model for Vercel's memory limits
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_synthetic_scenarios():
    scenarios = [
        "User slept 8 hours, ate 3500kcal, successfully completed Push workout with 4x8 bench press.",
        "User slept 5 hours, increased caffeine, hit targets but felt high fatigue.",
        "User missed Lunch, compensated with high-calorie Dinner, maintained weight stability.",
        "User hit 100% hydration, reported improved recovery and lower muscle soreness."
    ]
    # In a real run, we would generate 500-1000 of these
    return scenarios

def create_index():
    scenarios = generate_synthetic_scenarios()
    embeddings = model.encode(scenarios)
    
    # Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Save index and metadata
    faiss.write_index(index, "fitness_os.index")
    with open("metadata.json", "w") as f:
        json.dump(scenarios, f)
    
    print(f"RAG Index initialized with {len(scenarios)} scenarios.")

if __name__ == "__main__":
    create_index()
