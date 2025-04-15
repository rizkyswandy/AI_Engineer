import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from transformers import pipeline
from typing import Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
import torch
import uvicorn
from ocr import detect_image

HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Successfully logged in to Hugging Face Hub")
else:
    print("Warning: No Hugging Face token provided. Only public models will be accessible.")

class Payload():
    image_path: str
    
def get_model(task: str, model_name: Optional[str] = None):
    """Get model from cache or load it"""
    key = f"{task}_{model_name}" if model_name else task
    
    if key not in MODEL_CACHE:
        try:
            model_name = "facebook/bart-large-mnli"  
            
            MODEL_CACHE[key] = pipeline(task, model=model_name, device=0 if torch.mps.is_available() else -1)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return MODEL_CACHE[key]

MODEL_CACHE = {}

def extract_invoice_fields(text: str) -> Dict[str, Any]:
    """Extract fields specific to invoices"""
    import re
    
    fields = {}
    
    # Extract vendor/issuer name
    vendor_match = re.search(r'([\w\s]+)\s+Contractor', text)
    if vendor_match:
        fields["vendor"] = vendor_match.group(1).strip()
    
    # Extract customer name
    customer_match = re.search(r'ISSUED TO\s+(.+?)(?:\s+INVOICE|\s+\d{4}|$)', text, re.DOTALL)
    if customer_match:
        fields["customer"] = customer_match.group(1).strip()
    
    # Extract date
    date_match = re.search(r'DATE ISSUED\s+(\d{2}\.\d{2}\.\d{4})', text)
    if date_match:
        fields["date"] = date_match.group(1)
    
    # Extract customer address
    address_match = re.search(r'(\d+\s+Anywhere\s+St.*?Any\s+City,\s+ST\s+\d{5})', text)
    if address_match:
        fields["customer_address"] = address_match.group(1).strip()
    
    # Extract customer email
    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
    if email_match:
        fields["customer_email"] = email_match.group(1)
    
    # Extract line items
    items = []
    item_pattern = r'([A-Za-z\s]+)\s+\$?([\d,.]+\.\d{2})'
    for match in re.finditer(item_pattern, text):
        description = match.group(1).strip()
        if not any(keyword in description.upper() for keyword in ["SUBTOTAL", "TAX", "TOTAL"]):
            items.append({
                "description": description,
                "amount": match.group(2).replace(',', '')
            })
    
    if items:
        fields["line_items"] = items
    
    # Extract subtotal
    subtotal_match = re.search(r'SUBTOTAL\s+\$?([\d,.]+)', text)
    if subtotal_match:
        fields["subtotal"] = subtotal_match.group(1).replace(',', '')
    
    # Extract tax
    tax_match = re.search(r'TAX\s+\$?([\d,.]+)', text)
    if tax_match:
        fields["tax"] = tax_match.group(1).replace(',', '')
    
    # Extract total
    total_match = re.search(r'TOTAL\s+\$?([\d,.]+)', text)
    if total_match:
        fields["total"] = total_match.group(1).replace(',', '')
    
    # Extract payment details
    payment_match = re.search(r'PAYMENT DETAILS(.*?)(?:$)', text, re.DOTALL)
    if payment_match:
        fields["payment_details"] = payment_match.group(1).strip()
    
    return fields

def extract_receipt_fields(text: str) -> Dict[str, Any]:
    """Extract fields specific to restaurant receipts"""
    import re
    
    fields = {}
    
    # Extract merchant name (usually first line)
    merchant_match = re.search(r'^([A-Z\s]+)', text)
    if merchant_match:
        fields["merchant"] = merchant_match.group(1).strip()
    
    # Extract address
    address_match = re.search(r'(\d+\s+[^,;\n]+[,;]\s*[A-Z]{2})', text)
    if address_match:
        fields["address"] = address_match.group(1).strip()
    
    # Extract date
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
    if date_match:
        fields["date"] = date_match.group(1)
    
    # Extract time
    time_match = re.search(r'(\d{1,2}\*\d{2}\s*[AP]M)', text)
    if time_match:
        fields["time"] = time_match.group(1).replace('*', ':')
    
    # Extract payment method
    payment_match = re.search(r'(VISA|MASTERCARD|AMEX|DISCOVER)\s+(\d{4})', text)
    if payment_match:
        fields["payment_method"] = payment_match.group(1)
        fields["card_last_four"] = payment_match.group(2)
    
    # Extract items
    items = []
    item_matches = re.finditer(r'([^\n\d$]+)\s+\$?(\d+\.\d{2})', text)
    for match in item_matches:
        description = match.group(1).strip()
        if not description.upper() in ["SUBTOTAL:", "TAX:", "TOTAL:", "TIP:"]:
            items.append({
                "description": description,
                "price": match.group(2)
            })
    
    if items:
        fields["items"] = items
    
    # Extract subtotal
    subtotal_match = re.search(r'SUBTOTAL:\s*\$?(\d+\.\d{2})', text)
    if subtotal_match:
        fields["subtotal"] = subtotal_match.group(1)
    
    # Extract tax
    tax_match = re.search(r'Tax:\s*\$?(\d+\.\d{2})', text)
    if tax_match:
        fields["tax"] = tax_match.group(1)
    
    # Extract total
    subtotal_match = re.search(r'SUBTOTAL:\s*\$?(\d+\.\d{2})', text, re.IGNORECASE)
    if subtotal_match:
        fields["subtotal"] = subtotal_match.group(1)
    
    total_match = re.search(r'TOTAL:\s*\$?(\d+\.\d{2})', text, re.IGNORECASE)
    if total_match:
        fields["total"] = total_match.group(1)
    
    return fields

def extract_fields(text: str, doc_type: str) -> Dict[str, Any]:
    """Extract key fields based on document type"""
    if doc_type == "invoice":
        return extract_invoice_fields(text)
    elif doc_type == "receipt":
        return extract_receipt_fields(text)
    else:
        return {}
    
def text_classification(text: str) -> Dict[str, Any]:
    """
    Classify text to identify document type and extract key fields
    
    Args:
        text: The document text to analyze
        
    Returns:
        Dict containing document type and extracted fields
    """
    # Get document classification model
    model = get_model("zero-shot-classification", "facebook/bart-large-mnli")
    
    candidate_labels = ["invoice", "receipt"]
    
    # Classify document type
    classification_result = model(text, candidate_labels)
    
    # Handle different possible output formats
    if isinstance(classification_result, dict) and 'labels' in classification_result:
        doc_type = classification_result['labels'][0]
    elif isinstance(classification_result, list) and len(classification_result) > 0:
        doc_type = classification_result[0]['label']
    else:
        # Default fallback
        doc_type = "unknown"
        
    print(f"document type: {doc_type}")
    
    # Extract key fields based on document type
    fields = extract_fields(text, doc_type)
    
    return {
        "document_type": doc_type,
        "fields": fields
    }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Text Classification API"}

@app.post("/analyze")
async def classify_text(image: UploadFile = File(...)):
    """Classify text"""
    try:
        temp_path = f"temp/{image.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await image.read())
        
        text_from_image = detect_image(temp_path)
        full_text = ""
    
        for detection in text_from_image:
            text = detection[1]
            full_text += text + " "
        print("Text from image:", full_text)
        
        if not text_from_image:
            raise HTTPException(status_code=400, detail="No text detected in the image")
        
        result = text_classification(full_text)
        
        return {
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run("DocsIntelligenceSystem:app", host="0.0.0.0", port=port, reload=True)