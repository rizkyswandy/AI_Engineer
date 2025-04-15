# Real-Time Content Moderator with Toxicity Detection + Auto-Categorization

An API service that moderates and classifies user-submitted text using Hugging Face models.

## Features

- **Toxicity Detection**: Identifies toxic or inappropriate language in text
- **Topic Classification**: Categorizes text into topics like politics, sports, entertainment, etc.
- **Simple REST API**: Easy-to-use API endpoint for content moderation

## API Documentation

### Moderate Text

Analyzes text for toxicity and classifies its topic.

**Endpoint**: `POST /moderate`

**Request Format**:
```json
{
  "prompt": "Your example sentence here"
}
```

**Response Format**:
```json
{
  "text": "That award was rigged. The winner is an overrated hack.",
  "toxicity_score": 0.92,
  "is_toxic": true,
  "classification_results": {
    "entertainment": 0.75
  }
}
```

**Response Parameters**:
- `text` (string): The original input text
- `toxicity_score` (float): A score between 0 and 1 indicating toxicity level
- `is_toxic` (boolean): Whether the text is considered toxic
- `classification_results` (object): Topic categories with confidence scores above 0.5

## Technical Implementation

### Models Used

- **Toxicity Detection**: `unitary/toxic-bert`
- **Topic Classification**: `facebook/bart-large-mnli` with zero-shot classification

### Topic Categories

The service classifies text into the following categories:
- Politics
- Sports
- Technology
- Entertainment
- Health

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn transformers torch huggingface_hub pydantic
   ```
3. Run the server:
   ```bash
   python ToxicAndClassification.py
   ```

## Example Usage

```bash
curl -X POST http://127.0.0.1:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "That award was rigged. The winner is an overrated hack."}'
```