# BFI Live Code Case Study
##### Property of Rizky Azmi Swandy

## [Case 1: Real-Time Content Moderator](./case1/README.md)

An API service that performs real-time content moderation with:

- **Toxicity Detection**: Identifies harmful or inappropriate content
- **Topic Classification**: Automatically categorizes text into relevant topics
- **REST API**: Simple `/moderate` endpoint for easy integration

```bash
# Example usage
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your text here"}'
```

## [Case 2: Document Intelligence System](./case2/README.md)

A document processing service that:

- **Extracts Text**: Uses OCR to convert images to text
- **Classifies Documents**: Identifies document types (invoices, receipts)
- **Extracts Key Information**: Automatically pulls out dates, amounts, and other important data
- **REST API**: Simple `/analyze` endpoint for document processing

```bash
# Example usage
curl -X POST http://localhost:8000/analyze \
  -F "image=@path/to/document.png"
```

## Implementation Features

- **Modern AI Models**: Leverages Hugging Face transformers for NLP tasks
- **FastAPI Backend**: High-performance, easy-to-use REST API
- **Extensible Design**: Easily add new document types or classification categories
- **GPU Acceleration**: Supports GPU for faster processing when available

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the services:
   ```bash
   # For content moderation service
   cd case1
   python ToxicAndClassification.py
   
   # For document processing service
   cd case2
   python DocsIntelligenceSystem.py
   ```

## Technologies Used

- **FastAPI**: Modern, high-performance web framework
- **Hugging Face Transformers**: State-of-the-art NLP models
- **EasyOCR**: Powerful OCR engine for text extraction
- **PyTorch**: Deep learning framework

## Use Cases

- **Content Platforms**: Filter user-generated content
- **Finance**: Process invoices and receipts automatically
- **Customer Service**: Classify and route customer communications
- **Document Management**: Extract and organize information from scanned documents