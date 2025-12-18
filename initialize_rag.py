import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def setup_knowledge():
    # These are your "Lessons" for the AI
    scenarios = [
        "Scenario: Slept < 6 hours. Advice: Reduce weight by 10% today to prevent injury.",
        "Scenario: Hit all calorie goals. Advice: High energy detected! Try to add 2kg to your main lift.",
        "Scenario: Missed hydration target. Advice: Drink 500ml water before starting your workout.",
        "Scenario: Perfect sleep and nutrition. Advice: Great day! Follow your standard progression plan."
    ]
    
    # Convert text to vectors (numbers)
    embeddings = model.encode(scenarios)
    
    # Create the FAISS Index (The search engine file)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Save the files
    faiss.write_index(index, "fitness_os.index")
    with open("metadata.json", "w") as f:
        json.dump(scenarios, f)
    
    print("Files created: fitness_os.index and metadata.json. Now upload these to GitHub!")

if __name__ == "__main__":
    setup_knowledge()
