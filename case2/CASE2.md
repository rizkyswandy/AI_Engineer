# Document Intelligence System with OCR + NLP

A lightweight AI service that extracts text from images, classifies document types, and extracts key information from documents like invoices and receipts.

## Features

- **OCR Processing**: Extracts text from uploaded images using EasyOCR
- **Document Classification**: Identifies document type (invoice, receipt, etc.)
- **Field Extraction**: Extracts key information like dates, amounts, and company names
- **REST API**: Simple API for document processing

## API Documentation

### Analyze Document

Processes an uploaded document image, extracts text, and identifies key information.

**Endpoint**: `POST /analyze`

**Request Format**:
- Form data with an image file upload

**Response Format**:
```json
{
  "result": {
    "document_type": "invoice",
    "fields": {
      "vendor": "XYZ Company",
      "customer": "ABC Corp",
      "date": "20.04.2025",
      "total": "2,450.00",
      "line_items": [
        {
          "description": "Professional Services",
          "amount": "2000.00"
        },
        {
          "description": "Support Fees",
          "amount": "450.00"
        }
      ]
    }
  }
}
```

**Response Parameters**:
- `document_type` (string): Type of document detected (invoice, receipt)
- `fields` (object): Extracted information specific to the document type

### Field Extraction Details

#### Invoice Fields
- `vendor`: Company issuing the invoice
- `customer`: Customer name
- `date`: Invoice date
- `customer_address`: Customer's address
- `customer_email`: Customer's email
- `line_items`: List of products/services with amounts
- `subtotal`: Subtotal amount
- `tax`: Tax amount
- `total`: Total amount
- `payment_details`: Payment information

#### Receipt Fields
- `merchant`: Store or restaurant name
- `address`: Merchant address
- `date`: Transaction date
- `time`: Transaction time
- `payment_method`: Type of payment
- `card_last_four`: Last four digits of card (if applicable)
- `items`: List of purchased items with prices
- `subtotal`: Subtotal amount
- `tax`: Tax amount
- `total`: Total amount

## Technical Implementation

### Technologies Used

- **OCR Engine**: EasyOCR
- **Document Classification**: Facebook's BART model with zero-shot classification
- **Backend Framework**: FastAPI
- **Text Processing**: Regular expressions for field extraction

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn transformers torch huggingface_hub easyocr python-multipart
   ```
3. Create a temp directory for uploaded files:
   ```bash
   mkdir temp
   ```
4. Run the server:
   ```bash
   python DocsIntelligenceSystem.py
   ```

## Example Usage

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/invoice.png"
```