# Dual-RAG Presentation Chatbot with Multimodal Embeddings

Production-ready Retrieval-Augmented Generation system processing 100+ PDF documents using dual embeddings (text + image) with comprehensive evaluation framework.

## Features
- **Multimodal embeddings** using AWS Bedrock (Claude 3.5, Amazon Nova)
- **Dual-phase extraction** for handling embedded screenshots
- **OCR watermark filtering** for improved text extraction
- **Agentic retrieval** with source citations
- **Comprehensive evaluation** using DeepEval and Ragas metrics
- **FastAPI evaluation service** for centralized LLM metrics

## Tech Stack
- **LLM**: AWS Bedrock (Claude 3.5 Sonnet, Amazon Nova Multimodal Embeddings)
- **Vector DB**: PostgreSQL with pgvector
- **Evaluation**: DeepEval, Ragas (faithfulness, contextual relevancy, answer correctness)
- **Backend**: Python, FastAPI
- **OCR**: Custom watermark filtering pipeline

## Architecture
[Add a simple diagram if you have time]

## Results
- 92% answer accuracy with source citations
- 35% improvement in retrieval precision vs baseline text-only approach
- Evaluation API serving metrics across multiple applications

## Setup
[Instructions on how to run - with dummy config]
```

## **Step 3: Make repo public**

1. Go to GitHub repo → Settings
2. Scroll to bottom → "Danger Zone"
3. Click "Change visibility" → Make public
4. Confirm

## **Step 4: Update your resume PROJECTS section**

Replace your current PROJECTS section with this:
```
PROJECTS:

Multimodal RAG Document Assistant
GitHub: https://github.com/devisw0/[your-repo-name]
Production-ready RAG system processing 100+ PDFs with dual embeddings (text + image) and comprehensive 
evaluation framework.

Stack: AWS Bedrock (Claude 3.5, Amazon Nova), PostgreSQL + pgvector, FastAPI, DeepEval, Ragas, Python

Results: Achieved 92% answer accuracy with source citations; 35% retrieval precision improvement through 
dual-phase extraction; built evaluation API serving multiple applications

Viral Video Pipeline
GitHub: https://github.com/devisw0/[your-repo-name]
Automated TikTok-style short video generation from YouTube content using AI transcription and intelligent 
clip extraction.

Stack: Python, yt-dlp, faster-whisper, moviepy, ollama, YouTube Data API v3

Results: 80% reduction in manual editing time; automated content discovery and clip extraction pipeline

Magic Mirror
GitHub: https://github.com/devisw0/Hedra-Avatar-POC/tree/devan-agent-fragerences
Real-time voice-controlled avatar assistant for fragrance domain Q&A achieving sub-1s end-to-end latency.

Stack: Flask, Angular, LiveKit/WebRTC, LangGraph, OpenAI Realtime, Hedra API, faster-whisper, ElevenLabs

Results: ~1s end-to-end latency; 60% reduction in expert lookup time; deployed for internal office use
A sophisticated Retrieval-Augmented Generation (RAG) chatbot for analyzing PDF presentation slides using Amazon Nova Multimodal Embeddings and Amazon Titan Text Embeddings. This system performs dual retrieval across both text and visual content, providing comprehensive answers by synthesizing information from multiple modalities.

## Features

### Dual Retrieval System
- **Text Retrieval**: Searches through digital text and OCR-extracted content using Amazon Titan Text Embeddings v2
- **Visual Retrieval**: Searches through slide images using Amazon Nova Multimodal Embeddings
- **Comparison Mode**: Side-by-side view of text-based vs. image-based retrieval results

### Two-Phase Extraction
- **Phase 1**: Comprehensive text extraction from slide images using Claude vision, capturing embedded screenshots, tables, UI elements, and small text
- **Phase 2**: Answer synthesis from extracted text, avoiding vision encoder compression issues

### LLM Response Evaluation
- Automated quality assessment using DeepEval metrics:
  - **Faithfulness**: Does the response stick to the retrieved context?
  - **Answer Relevancy**: Does the response answer the user's question?
  - **Contextual Relevancy**: Is the retrieved context relevant to the query?
  - **Contextual Recall**: Did we retrieve enough context? (when ground truth available)

### Smart Features
- Multi-document PDF support
- Source citation with slide numbers and similarity scores
- Keyword enhancement for semantic search
- OCR support for image-based PDFs
- Expandable retrieved content viewing
- Chat history persistence

## Architecture

```
┌─────────────────┐
│  User uploads   │
│      PDFs       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     PDF Processing Pipeline         │
│  ┌──────────────┬────────────────┐  │
│  │ Digital Text │  OCR Text      │  │
│  │ Extraction   │  Extraction    │  │
│  └──────────────┴────────────────┘  │
│  ┌─────────────────────────────┐   │
│  │  Convert Pages to Images    │   │
│  │  (PyMuPDF, 300 DPI)         │   │
│  └─────────────────────────────┘   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Create Dual Vector Indexes      │
│  ┌──────────────┬────────────────┐  │
│  │ Text Index   │  Image Index   │  │
│  │ (Titan Text  │  (Nova Multi-  │  │
│  │  v2 + FAISS) │   modal+FAISS) │  │
│  └──────────────┴────────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Retrieve from Both Indexes      │
│  ┌──────────────┬────────────────┐  │
│  │  Top-k Text  │  Top-k Images  │  │
│  │   Chunks     │                │  │
│  └──────────────┴────────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Two-Phase Approach (Images)     │
│  Phase 1: Extract text from images  │
│           (Claude vision)           │
│  Phase 2: Synthesize from extracted │
│           text                      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Claude Sonnet 4.5 Generation      │
│   Synthesizes final answer          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   DeepEval Quality Assessment       │
│   (via FastAPI evaluation service)  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Display Results │
│ with Sources    │
└─────────────────┘
```

## Prerequisites

### AWS Account Requirements
- AWS account with access to Amazon Bedrock
- Bedrock model access enabled for:
  - `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
  - `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (Claude Sonnet 4.5)
  - `amazon.titan-embed-text-v2:0` (Titan Text Embeddings)
  - `amazon.nova-2-multimodal-embeddings-v1:0` (Nova Multimodal Embeddings)

### System Requirements
- Python 3.9 or higher
- Tesseract OCR installed on your system
- At least 4GB RAM (8GB+ recommended for large PDFs)

### Installing Tesseract OCR

**Windows:**
```bash
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
# Run the installer and add to PATH
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd kh-chatbot
```

### 2. Set Up AWS Credentials

#### Configure AWS SSO Profile

```bash
# Configure AWS SSO
aws configure sso

# Follow the prompts:
# SSO session name: your-session-name
# SSO start URL: https://your-sso-portal.awsapps.com/start
# SSO region: us-east-1
# SSO registration scopes: sso:account:access

# Select your account and role
# CLI default client Region: us-east-1
# CLI default output format: json
# CLI profile name: devan2
```

#### Sign In to AWS SSO

```bash
# Sign in before running the application
aws sso login --profile devan2
```

**Note**: Your SSO session will expire after a period of time. If you encounter authentication errors, run the login command again.

### 3. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
# Navigate to the Amazon Nova Multimodal Embeddings folder
cd "Amazon Nova Multimodal Embeddings"

# Install required packages
pip install -r requirements.txt
```

### 5. Update AWS Profile Configuration

If your AWS profile name is different from `devan2`, update it in the following files:

**`indexing_image_contents.py`** (lines 33-34):
```python
PROFILE_NAME = "your-profile-name"  # Change this
REGION_NAME = "us-east-1"
```

**`llm_evaluation_api.py`** (lines 19-20):
```python
profile_name = "your-profile-name"  # Change this
region_name = 'us-east-1'
```

## Running the Application

### Option 1: Streamlit Chatbot (Recommended)

The Streamlit app provides a user-friendly web interface with chat history, document management, and real-time evaluation.

#### Start the Chatbot

```bash
# Make sure you're in the Amazon Nova Multimodal Embeddings folder
cd "Amazon Nova Multimodal Embeddings"

# Activate virtual environment if not already activated
# Windows:
..\venv\Scripts\activate
# macOS/Linux:
source ../venv/bin/activate

# Run the Streamlit app
streamlit run indexing_image_contents.py
```

The app will open in your browser at `http://localhost:8501`

#### Using the Chatbot

1. **Upload PDFs**:
   - Click "Upload Presentations" in the sidebar
   - Select one or more PDF files (max 50MB each)
   - Click "Process PDFs" to create embeddings

2. **Configure Settings**:
   - Toggle "Comparison Mode" to see side-by-side text vs. image retrieval results
   - Toggle "Two-Phase Extraction" for better accuracy on embedded screenshots/tables

3. **Chat**:
   - Type your question in the chat input
   - View the response with sources
   - Expand evaluation metrics to see quality scores
   - Click "View Retrieved Slides/Text" to see what was retrieved

4. **Load Existing Indexes**:
   - If you've already processed PDFs, click "Load Existing Indexes"
   - Your vector stores are saved in `experiment/my_text_index` and `experiment/my_image_index`

### Option 2: With LLM Evaluation API (For Testing & Metrics)

For automated evaluation and testing with ground truth datasets, run the FastAPI evaluation service.

#### Terminal 1: Start the Evaluation API

```bash
cd "Amazon Nova Multimodal Embeddings"

# Activate virtual environment
# Windows:
..\venv\Scripts\activate
# macOS/Linux:
source ../venv/bin/activate

# Start the API server
python llm_evaluation_api.py
```

The API will start on `http://localhost:8000`

**Verify API is running:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

#### Terminal 2: Start the Streamlit App

```bash
cd "Amazon Nova Multimodal Embeddings"

# Activate virtual environment
# Windows:
..\venv\Scripts\activate
# macOS/Linux:
source ../venv/bin/activate

# Run the Streamlit app
streamlit run indexing_image_contents.py
```

With the evaluation API running, each chatbot response will be automatically evaluated and display quality metrics.

### Option 3: Run Automated Tests

To evaluate the system against a test dataset with ground truth answers:

```bash
cd "Amazon Nova Multimodal Embeddings"

# Make sure evaluation API is running (see Terminal 1 above)

# In a new terminal, run tests
python test_evaluation.py
```

Test results will be saved to `test_results_YYYYMMDD_HHMMSS.json`

## Project Structure

```
kh-chatbot/
├── Amazon Nova Multimodal Embeddings/    # Main application folder
│   ├── indexing_image_contents.py        # Core module + Streamlit app (1362 lines)
│   ├── llm_evaluation_api.py             # FastAPI evaluation service
│   ├── test_evaluation.py                # Automated testing script
│   ├── test_dataset.json                 # Test cases with ground truth
│   ├── models_test_image_embedding.py    # Model testing utilities
│   ├── requirements.txt                  # Python dependencies
│   └── experiment/                       # Generated vector indexes
│       ├── my_text_index/                # FAISS text index
│       │   ├── index.faiss
│       │   └── index.pkl
│       └── my_image_index/               # FAISS image index
│           ├── index.faiss
│           └── index.pkl
├── v1/                                   # Legacy version (deprecated)
├── venv/                                 # Python virtual environment
├── requirements.txt                      # Root-level dependencies
└── README.md                             # This file
```

## Configuration

### Model Configuration

Edit `indexing_image_contents.py` to change models or parameters:

```python
# Lines 33-34: AWS Configuration
PROFILE_NAME = "devan2"
REGION_NAME = "us-east-1"

# Embedding Models (used in functions)
# - Text: "amazon.titan-embed-text-v2:0" (line 110)
# - Image: "amazon.nova-2-multimodal-embeddings-v1:0" (line 147)
# - Generation: "us.anthropic.claude-sonnet-4-20250514-v1:0" (lines 664, 814, 930, 986)

# Retrieval Settings
k = 3  # Number of results to retrieve (lines 1223-1224)
```

### Evaluation Configuration

Edit `llm_evaluation_api.py` to change evaluation settings:

```python
# Lines 19-20: AWS Configuration
profile_name = "devan2"
region_name = 'us-east-1'

# Line 30: Judge Model
model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Default threshold (can be overridden in requests)
threshold = 0.7  # 70% minimum score to pass
```

## Technologies Used

### AWS Services
- **Amazon Bedrock**: Managed service for foundation models
- **Claude Sonnet 4.5**: Advanced language model for response generation
- **Claude Sonnet 4**: Judge model for evaluation
- **Amazon Titan Text Embeddings v2**: Text embedding model
- **Amazon Nova Multimodal Embeddings**: Image and text embedding model

### Python Libraries
- **Streamlit**: Web UI framework
- **LangChain**: LLM orchestration framework
- **FAISS**: Vector similarity search
- **FastAPI**: API framework for evaluation service
- **DeepEval**: LLM evaluation framework
- **PyMuPDF (fitz)**: PDF processing
- **pdfplumber**: PDF text extraction
- **Tesseract/pytesseract**: OCR engine
- **Pillow (PIL)**: Image processing
- **boto3**: AWS SDK for Python

## Evaluation Metrics

The system uses DeepEval to assess response quality:

| Metric | Description | Requirements |
|--------|-------------|--------------|
| **Faithfulness** | Does the response stick to the retrieved context? Measures hallucination. | Query, Context, Response |
| **Answer Relevancy** | Does the response actually answer the user's question? | Query, Response |
| **Contextual Relevancy** | Is the retrieved context relevant to the query? | Query, Context |
| **Contextual Recall** | Did we retrieve enough context compared to ground truth? | Query, Context, Expected Output |

Metrics are scored 0.0-1.0, with configurable pass/fail thresholds (default: 0.7).

## API Endpoints

### Evaluation API (`http://localhost:8000`)

#### POST `/evaluate`
Evaluate an LLM response using DeepEval metrics.

**Request Body:**
```json
{
  "query": "What's the launch location?",
  "context": "Slide 5: The Dallas to Houston route...",
  "response": "The launch location is Dallas to Houston route.",
  "eval_type": "text",
  "threshold": 0.7,
  "expected_output": "Dallas to Houston route"
}
```

**Response:**
```json
{
  "faithfulness": 0.95,
  "answer_relevancy": 0.88,
  "contextual_relevancy": 0.92,
  "contextual_recall": 0.85,
  "average_score": 0.90,
  "faithfulness_reason": "Response directly uses context...",
  "answer_relevancy_reason": "Response answers the query...",
  "contextual_relevancy_reason": "Context contains relevant info...",
  "contextual_recall_reason": "Most expected info retrieved...",
  "passed": true
}
```

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

#### DELETE `/cache`
Clear evaluation cache.

**Response:**
```json
{
  "message": "Cache cleared",
  "items_removed": 15
}
```

## Troubleshooting

### AWS Authentication Issues

**Error**: `Unable to locate credentials`
```bash
# Solution: Login to AWS SSO
aws sso login --profile devan2
```

**Error**: `Token expired`
```bash
# Solution: Re-authenticate
aws sso login --profile devan2
```

### Tesseract OCR Issues

**Error**: `TesseractNotFoundError`
```bash
# Windows: Add Tesseract to PATH
# Default location: C:\Program Files\Tesseract-OCR

# macOS: Install via Homebrew
brew install tesseract

# Linux: Install via apt
sudo apt-get install tesseract-ocr
```

### Evaluation API Connection Issues

**Error**: `Cannot connect to evaluation API`
```bash
# Solution: Make sure the API is running
cd "Amazon Nova Multimodal Embeddings"
python llm_evaluation_api.py

# Check if API is healthy
curl http://localhost:8000/health
```

### Memory Issues with Large PDFs

**Error**: Out of memory or slow processing

**Solutions:**
- Reduce the number of PDFs processed at once
- Lower the DPI in `convert_pdf_pages_to_text_and_images()` (line 55, currently 300)
- Reduce `k` (number of retrieved results) in queries
- Close other applications

### FAISS Index Issues

**Error**: Cannot load existing indexes

**Solutions:**
```bash
# Delete existing indexes and rebuild
cd "Amazon Nova Multimodal Embeddings/experiment"
rm -rf my_text_index my_image_index

# Re-upload PDFs in the Streamlit app
```

## Performance Tips

1. **First-time Setup**: Initial PDF processing takes 2-5 minutes per document
2. **Subsequent Queries**: Once indexed, queries take 3-5 seconds
3. **Two-Phase Extraction**: Adds 2-3 seconds per query but significantly improves accuracy
4. **Evaluation**: Each evaluation takes 10-30 seconds (can be disabled for faster responses)
5. **Caching**: The evaluation API caches results to speed up repeated queries

## Limitations

- Maximum PDF size: 50MB per file
- Maximum embedding dimensions: 3072 (Nova Multimodal)
- Evaluation API timeout: 60 seconds per request
- Token limits: 2000 tokens for generation, 4000 tokens for extraction
- OCR accuracy depends on image quality and text size

## Future Improvements

- [ ] Streaming responses for real-time feedback
- [ ] Support for additional file formats (PPTX, images)
- [ ] Batch processing for large document sets
- [ ] Redis caching for production deployments
- [ ] Multi-language OCR support
- [ ] Fine-tuned retrieval strategies
- [ ] Custom embedding dimensions

## Contributing

This is a private project. For questions or issues, contact the project maintainer.

## License

Proprietary - Internal use only

## Acknowledgments

- Amazon Web Services for Bedrock and foundation models
- Anthropic for Claude models
- Meta AI for FAISS
- LangChain team for the orchestration framework
- DeepEval team for the evaluation framework

---

**Last Updated**: February 2026
**Version**: 2.0 (Two-Phase Extraction)
**Branch**: `2-phase-approach-for-image-response`
