from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime

app = FastAPI()

# Allow your website to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI Brain (MiniLM is lightweight for Vercel)
# This model converts text into numbers for the search engine
model = SentenceTransformer('all-MiniLM-L6-v2')

# Temporary Memory (Resets when the app is inactive)
storage = {}

@app.get("/api/health")
async def health():
    return {"status": "Running", "model_loaded": True}

@app.post("/api/log")
async def log_data(request: Request):
    """Saves the data sent from your React App"""
    data = await request.json()
    date_key = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    storage[date_key] = data
    return {"status": "success", "message": f"Saved data for {date_key}"}

@app.post("/api/rag/generate-plan")
async def generate_plan(date: str):
    """The RAG Engine: Finds advice based on your logged data"""
    user_day = storage.get(date)
    
    if not user_day:
        return {"next_workout": "Push", "ai_advice": "No data logged today. Please sync your data first!"}

    # Default advice if search fails
    advice = "Keep up the good work!"
    
    # --- START RAG SEARCH ---
    try:
        if os.path.exists("fitness_os.index"):
            # 1. Load the search index and the text descriptions
            index = faiss.read_index("fitness_os.index")
            with open("metadata.json", "r") as f:
                metadata = json.load(f)
            
            # 2. Convert today's stats into a search query
            stats = user_day.get('derived_metrics', {})
            search_query = f"Sleep: {stats.get('sleep_duration_min', 480)} mins, Calories: {stats.get('calorie_surplus', 0)}"
            
            # 3. Find the most similar scenario in the index
            query_vector = model.encode([search_query]).astype('float32')
            distances, indices = index.search(query_vector, k=1)
            
            # 4. Get the advice from that scenario
            advice = metadata[indices[0][0]]
    except Exception as e:
        advice = f"Standard Plan: Focus on recovery. (Search error: {str(e)})"
    # --- END RAG SEARCH ---

    return {
        "status": "generated",
        "next_workout": "Next Session in Cycle",
        "ai_advice": advice
    }
