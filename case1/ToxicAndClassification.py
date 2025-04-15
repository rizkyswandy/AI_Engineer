import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import torch
from huggingface_hub import login
import uvicorn

HF_TOKEN = os.environ.get("HF_TOKEN")

# Log in to Hugging Face if token is provided
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Successfully logged in to Hugging Face Hub")
else:
    print("Warning: No Hugging Face token provided. Only public models will be accessible.")
    
class PromptRequest(BaseModel):
    prompt: str

def get_model(task: str, model_name: Optional[str] = None):
    """Get model from cache or load it"""
    key = f"{task}_{model_name}" if model_name else task
    
    if key not in MODEL_CACHE:
        try:
            if not model_name:
                if task == "toxic-detection":
                    model_name = "unitary/toxic-bert"
                elif task == "classification":
                    model_name = "facebook/bart-large-mnli"  
            
            # Initialize the pipeline
            MODEL_CACHE[key] = pipeline(task, model=model_name, device=0 if torch.mps.is_available() else -1)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return MODEL_CACHE[key]

MODEL_CACHE = {}

def toxic_detection(text: str) -> List[Dict[str, Any]]:
    """Detect toxic comments"""
    model = get_model("text-classification", "unitary/toxic-bert")
    
    result = model(text)
    
    return [{"label": r["label"], "score": r["score"]} for r in result]

def text_classification(text: str) -> Dict[str, float]:
    """Classify text"""
    model = get_model("zero-shot-classification", "facebook/bart-large-mnli")
    
    candidate_labels = ["politics", "sports", "technology", "entertainment", "health"]
    
    result = model(text, candidate_labels)
    
    # Return dictionary of labels and scores
    return {label: score for label, score in zip(result["labels"], result["scores"])}

app = FastAPI()

app.add_middleware( CORSMiddleware, allow_origins=["*"],  allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )

@app.post("/")
def health_check():
    return {"status": "ok"}

@app.post("/moderate")
async def run_model_workflow(request: PromptRequest):
    """Run the model workflow"""
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Detect toxic comments
    toxic_results = toxic_detection(prompt)
    
    toxic_result = toxic_results[0]
    print(f"Toxicity result: {toxic_result}")
    
    if toxic_result["label"] == "toxic":
        is_toxic =  True
    
    # Classify text
    classification_results = text_classification(prompt)
    classification_result = {k: v for k, v in classification_results.items() if v > 0.5}
    print(f"Classification results: {classification_results}")
    
    return {
        "text" : prompt,
        "toxicity_score": toxic_result["score"],
        "is_toxic": is_toxic,
        "classification_results": classification_result
    }
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run("ToxicAndClassification:app", host="0.0.0.0", port=port, reload=True)
    
    
# Test the API
# curl -X POST http://127.0.0.1:8000/moderate -H "Content-Type: application/json" -d '{"prompt": "That award was rigged. The winner is an overrated hack."}'