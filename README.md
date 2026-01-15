# Dual-RAG Presentation Chatbot

A sophisticated RAG (Retrieval-Augmented Generation) system that uses both text and visual embeddings to answer questions about PDF presentations.

## Features

- **Dual Vector Stores**: Separate indexes for text content and visual slide images
- **Amazon Nova Multimodal Embeddings**: Advanced image understanding
- **Amazon Titan Text Embeddings**: High-quality text embeddings
- **Claude Sonnet 4**: State-of-the-art response generation
- **DeepEval Metrics**: Comprehensive evaluation (Faithfulness, Answer Relevancy, Contextual Relevancy)
- **Two-Phase Extraction**: Extracts text from images before synthesis for better accuracy
- **Streamlit UI**: Interactive chat interface with comparison mode

## Prerequisites

### 1. AWS Credentials
- AWS account with Bedrock access
- AWS CLI configured with profile named `devan2` (or update `PROFILE_NAME` in code)
- Access to these Bedrock models:
  - `us.anthropic.claude-sonnet-4-20250514-v1:0`
  - `amazon.titan-embed-text-v2:0`
  - `amazon.nova-2-multimodal-embeddings-v1:0`

### 2. Tesseract OCR
**Windows:**
```bash
# Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

# Add to PATH (usually):
C:\Program Files\Tesseract-OCR
```

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 3. Python
- Python 3.10 or higher
- Virtual environment recommended

## Installation

### Step 1: Clone/Navigate to Project
```bash
cd "C:\Users\dxp4392\OneDrive - International Flavors & Fragrances Inc\Desktop\kh-chatbot"
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
**Windows:**
```bash
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Upgrade pip
```bash
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

**If you encounter pandas build errors on Windows:**
```bash
# Install pre-built pandas wheel first
pip install pandas --only-binary=:all:

# Then install remaining dependencies
pip install -r requirements.txt
```

## Project Structure

```
kh-chatbot/
├── Amazon Nova Multimodal Embeddings/  # Main application
│   ├── indexing_image_contents.py      # Streamlit app (main UI)
│   ├── llm_evaluation_api.py           # FastAPI evaluation service
│   ├── test_evaluation.py              # Automated testing
│   ├── test_dataset.json               # Test cases
│   ├── models_test_image_embedding.py  # Model listing utility
│   └── requirements.txt                # Dependencies
├── v1/                                  # Earlier prototype version
└── requirements.txt                     # Combined dependencies
```

## Usage

### Running the Evaluation API (Required for Metrics)

The evaluation API must be running **before** starting the Streamlit app if you want evaluation metrics.

**Terminal 1:**
```bash
# Navigate to the folder
cd "Amazon Nova Multimodal Embeddings"

# Activate virtual environment
..\venv\Scripts\activate

# Run the API
python llm_evaluation_api.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Test the API:**
```bash
# In another terminal
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

### Running the Streamlit App

**Terminal 2:**
```bash
# Navigate to the folder
cd "Amazon Nova Multimodal Embeddings"

# Activate virtual environment
..\venv\Scripts\activate

# Run Streamlit
streamlit run indexing_image_contents.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The app will automatically open in your browser at `http://localhost:8501`

## Using the Application

### 1. Upload PDFs
- Go to the **"Manage Documents"** tab
- Click **"Choose PDF files"** and select your presentations
- Click **"🔨 Build/Rebuild Search Index"**
- Wait for processing (OCR + embedding generation)

### 2. Chat with Your Documents
- Switch to the **"Chat"** tab
- Ask questions about your presentations
- View retrieved slides and evaluation metrics

### 3. Comparison Mode
- Toggle **"Comparison Mode"** in the sidebar
- See side-by-side results from text vs. image retrieval
- Compare which retrieval method works better for different queries

### 4. Two-Phase Extraction
- Toggle **"Two-Phase Extraction"** in the sidebar
- Recommended for slides with embedded screenshots or tables
- Extracts text from images first, then synthesizes (avoids compression issues)

## Running Automated Tests

Test your RAG system with ground truth data:

```bash
# Make sure evaluation API is running first!

# Run tests
python test_evaluation.py
```

**Output:**
- Console: Real-time test progress and metrics
- File: `test_results_YYYYMMDD_HHMMSS.json` with detailed results

**Edit test cases:**
```json
// test_dataset.json
[
    {
        "query": "What's the launch location?",
        "expected_output": "Dallas to Houston route"
    }
]
```

## Configuration

### AWS Profile
Edit in `indexing_image_contents.py` and `llm_evaluation_api.py`:
```python
PROFILE_NAME = "devan2"  # Change to your AWS profile
REGION_NAME = "us-east-1"
```

### Models
Change models in the code:
```python
# Text embeddings
model_id="amazon.titan-embed-text-v2:0"

# Image embeddings
modelId="amazon.nova-2-multimodal-embeddings-v1:0"

# LLM
modelId="us.anthropic.claude-sonnet-4-20250514-v1:0"
```

### Evaluation Thresholds
In `llm_evaluation_api.py`:
```python
threshold: float = 0.7  # Minimum score to pass (0.0-1.0)
```

## Troubleshooting

### Pandas Installation Error
```bash
# Error: Could not find vswhere.exe
# Solution: Install pre-built wheel
pip install pandas --only-binary=:all:
```

### Tesseract Not Found
```bash
# Error: TesseractNotFoundError
# Solution: Add Tesseract to PATH or specify path in code:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### AWS Credentials Error
```bash
# Error: Unable to locate credentials
# Solution: Configure AWS CLI
aws configure --profile devan2
```

### Evaluation API Connection Error
```bash
# Error: Cannot connect to evaluation API
# Solution: Make sure API is running on port 8000
# Check: curl http://localhost:8000/health
```

### Port Already in Use
```bash
# Error: Address already in use
# Solution: Kill process or use different port

# For API (change in llm_evaluation_api.py):
uvicorn.run(app, host="0.0.0.0", port=8001)

# For Streamlit:
streamlit run indexing_image_contents.py --server.port 8502
```

### Dependency Conflicts
```bash
# Solution: Use clean virtual environment
deactivate
rm -rf venv  # or: rmdir /s venv (Windows)
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## API Endpoints

### Evaluation API (Port 8000)

**Health Check:**
```bash
GET http://localhost:8000/health
```

**Evaluate Response:**
```bash
POST http://localhost:8000/evaluate
Content-Type: application/json

{
    "query": "What is the revenue?",
    "context": "Q1 2024 revenue was $10 million",
    "response": "The revenue in Q1 2024 was $10 million",
    "eval_type": "text",
    "threshold": 0.7
}
```

**Clear Cache:**
```bash
DELETE http://localhost:8000/cache
```

## Performance Tips

1. **Use Two-Phase Extraction** for slides with embedded screenshots
2. **Adjust DPI** in `convert_pdf_pages_to_text_and_images()` (default: 300)
3. **Tune k parameter** for retrieval (default: 3 for images, 3 for text)
4. **Cache results** - API automatically caches evaluation results
5. **Batch processing** - Upload multiple PDFs at once

## Evaluation Metrics Explained

- **Faithfulness**: Does the response stick to the retrieved context? (No hallucinations)
- **Answer Relevancy**: Does the response actually answer the question?
- **Contextual Relevancy**: Is the retrieved context relevant to the query?
- **Contextual Recall** (optional): Did we retrieve enough context? (Requires ground truth)

## Cost Considerations

- **Bedrock API calls**: Charged per token/image
- **Embeddings**: ~$0.0001 per 1K tokens (Titan Text)
- **Image embeddings**: ~$0.06 per 1K images (Nova Multimodal)
- **LLM calls**: ~$0.003 per 1K input tokens (Claude Sonnet 4)

## Support

For issues:
1. Check the troubleshooting section above
2. Review console output for error messages
3. Verify AWS credentials and Bedrock model access
4. Ensure Tesseract is installed and in PATH

## License

Internal use only - IFF proprietary
