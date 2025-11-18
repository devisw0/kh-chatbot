import streamlit as st
import boto3
import pandas as pd
import os
import io
import tempfile
import pytesseract
from PIL import Image
import warnings

from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
INDEX_PATH = "my_slide_index"
CSV_OUTPUT = "extracted_content.csv"  # Optional: Save extraction to CSV
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
AWS_PROFILE = "devan2"
AWS_REGION = "us-east-1"
TOP_K = 7  # Retrieve top 7 slides

# --- HELPER FUNCTIONS ---

def get_session():
    """Returns an authenticated boto3 session."""
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)

@st.cache_resource
def get_embedding_model(_session):
    """Initializes the Titan Embedding model."""
    client = BedrockEmbeddings(
        client=_session.client('bedrock-runtime'), 
        model_id=TITAN_EMBEDDING_MODEL_ID
    )
    return client

@st.cache_resource
def get_llm(_session):
    """Initializes the Claude Chat Model."""
    client = ChatBedrock(
        client=_session.client('bedrock-runtime'),
        model_id=CLAUDE_MODEL_ID,
        model_kwargs={"temperature": 0.1}
    )
    return client

# --- TEXT EXTRACTION ---

def extract_text_from_pdf_ocr(pdf_path: str):
    """Extracts text (digital + OCR) from a single PDF."""
    import pdfplumber
    
    slide_data = []
    source_filename = os.path.basename(pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                digital_text = page.extract_text()
                text = ""

                if digital_text and digital_text.strip():
                    text = digital_text.strip()
                else:
                    # Fallback to OCR for image-based slides
                    img = page.to_image(resolution=300).original
                    text = pytesseract.image_to_string(img, lang='eng').strip()

                if text:
                    slide_data.append({
                        "DocumentSource": source_filename,
                        "SlideNumber": i,
                        "Content": text
                    })
    except Exception as e:
        st.error(f"Error during OCR for {source_filename}: {e}")
        return []
        
    return slide_data

# --- KEYWORD EXTRACTION FOR BETTER RETRIEVAL ---

def extract_keywords(text: str):
    """Extract important keywords to improve semantic search."""
    text_lower = text.lower()
    keywords = []
    
    # Define patterns to detect important topics
    patterns = {
        'allergen': ['allergen', 'allergy', 'allergic', 'non-allergen', 'non allergen'],
        'sensitive_skin': ['sensitive skin', 'dermatological', 'skin safe', 'skin suitable'],
        'safety': ['safety', 'safe', 'caution', 'warning', 'hazard'],
        'regulatory': ['regulatory', 'compliance', 'documentation', 'approval'],
        'fragrance': ['fragrance', 'scent', 'perfume', 'aroma', 'odor'],
        'floral': ['floral', 'flower', 'rose', 'jasmine', 'lily', 'violet'],
        'woody': ['woody', 'wood', 'cedar', 'sandalwood', 'pine'],
        'citrus': ['citrus', 'lemon', 'orange', 'bergamot', 'grapefruit'],
        'fresh': ['fresh', 'clean', 'aquatic', 'marine'],
        'spicy': ['spicy', 'spice', 'pepper', 'cinnamon', 'ginger'],
    }
    
    for category, terms in patterns.items():
        for term in terms:
            if term in text_lower:
                keywords.append(category)
                break
    
    return keywords

# --- INDEXING PIPELINE ---

def build_index_from_uploaded_files(uploaded_files, embedding_model):
    """
    Complete indexing pipeline:
    1. Extract text from PDFs
    2. Add keyword enhancement
    3. Create LangChain Documents
    4. Build FAISS index
    5. Save to disk
    6. Optionally save to CSV
    """
    all_extracted_data = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # --- STEP 1: Extract text from all PDFs ---
        for uploaded_file in uploaded_files:
            st.info(f"📄 Processing: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text/OCR
            extracted_text = extract_text_from_pdf_ocr(temp_path)
            st.success(f"✅ Extracted {len(extracted_text)} slides from {uploaded_file.name}")
            
            all_extracted_data.extend(extracted_text)

        # --- Check if extraction worked ---
        if not all_extracted_data:
            st.error("❌ No content extracted from PDFs. Check your files.")
            return None
        
        st.success(f"📊 Total slides extracted: {len(all_extracted_data)}")

        # --- STEP 2: Save to CSV (optional, for inspection) ---
        df = pd.DataFrame(all_extracted_data)
        df.to_csv(CSV_OUTPUT, index=False)
        st.success(f"💾 Saved extraction to {CSV_OUTPUT}")

        # --- STEP 3: Convert to LangChain Documents with keyword enhancement ---
        st.info("🔧 Creating embeddings and building search index...")
        documents = []
        
        for item in all_extracted_data:
            content = item['Content']
            keywords = extract_keywords(content)
            
            # Enhance content with keywords for better semantic matching
            enhanced_content = content
            if keywords:
                enhanced_content = f"{content}\n\nKey Topics: {', '.join(keywords)}"
            
            # Create document with metadata
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "document_source": item['DocumentSource'],
                    "slide_number": item['SlideNumber'],
                    "keywords": ", ".join(keywords) if keywords else "none"
                }
            )
            documents.append(doc)
        
        # --- STEP 4: Build FAISS vector store ---
        vector_store = FAISS.from_documents(documents, embedding_model)
        
        # --- STEP 5: Save to disk ---
        vector_store.save_local(INDEX_PATH)
        st.success(f"✅ Index built and saved to {INDEX_PATH}")
        
        return vector_store

# --- AGENT CREATION ---

def create_rag_agent(llm, vector_store):
    """Creates the RAG agent with improved search capabilities."""
    
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """
        Retrieve relevant slides from the vector store to answer questions.
        Use this tool to search the document knowledge base.
        """
        retrieved_docs = retriever.invoke(query)
        
        # Format content for the LLM
        serialized_content = "\n\n".join(
            f"📄 Source: {doc.metadata.get('document_source', 'N/A')} | Slide: {doc.metadata.get('slide_number', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        
        return serialized_content, retrieved_docs

    tools = [retrieve_context]
    
    # Improved system prompt
    system_prompt = (
        "You are a helpful assistant for answering questions about uploaded documents. "
        "You have access to the 'retrieve_context' tool to search the knowledge base.\n\n"
        
        "RULES:\n"
        "1. For casual conversation (greetings, how are you, etc.), respond directly WITHOUT using the tool.\n"
        "2. For ANY factual question about the documents, you MUST use the 'retrieve_context' tool FIRST.\n"
        "3. ALWAYS cite which document and slide number(s) your answer comes from.\n"
        "4. If the retrieved content doesn't contain the answer, say: 'I'm sorry, that information is not in the uploaded documents.'\n"
        "5. NEVER make up information. Only use what's in the retrieved context.\n"
    )

    return create_agent(llm, tools, system_prompt=system_prompt)

# --- STREAMLIT UI ---

st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Document RAG Chatbot")
st.markdown("Upload PDFs, build a searchable index, and ask questions!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load models
session = get_session()
llm = get_llm(session)
embeddings = get_embedding_model(session)

# Create tabs
chat_tab, manage_tab = st.tabs(["💬 Chat", "⚙️ Manage Documents"])

# --- DOCUMENT MANAGEMENT TAB ---
with manage_tab:
    st.header("📤 Upload and Index Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("🔨 Build/Rebuild Search Index", type="primary"):
        if not uploaded_files:
            st.error("⚠️ Please upload at least one PDF file.")
        else:
            with st.spinner("🔄 Building index... This may take a minute for large files."):
                
                # Build the index
                vector_store = build_index_from_uploaded_files(uploaded_files, embeddings)
                
                if vector_store:
                    # Create new agent with the updated index
                    st.session_state.vector_store = vector_store
                    st.session_state.agent = create_rag_agent(llm, vector_store)
                    
                    st.balloons()
                    st.success(f"🎉 Success! Indexed {len(vector_store.docstore._dict)} slides.")
                else:
                    st.error("❌ Indexing failed. Check the error messages above.")
    
    # Show current index status
    if os.path.exists(INDEX_PATH):
        st.info(f"📂 Existing index found at `{INDEX_PATH}`")
    else:
        st.warning("⚠️ No index found. Please upload files and build the index.")

# --- CHAT TAB ---
with chat_tab:
    
    # Check if index exists
    if not os.path.exists(INDEX_PATH):
        st.warning("⚠️ No search index found. Go to 'Manage Documents' tab to upload and index your PDFs.")
        st.stop()
    
    # Load agent if not already loaded
    if "agent" not in st.session_state:
        with st.spinner("Loading search index..."):
            st.session_state.vector_store = FAISS.load_local(
                INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.session_state.agent = create_rag_agent(llm, st.session_state.vector_store)
        st.success("✅ Ready to answer questions!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 View Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching documents..."):
                
                # Convert history to LangChain messages
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))
                
                messages_for_agent = chat_history + [HumanMessage(content=prompt)]
                
                # Stream agent response
                stream_events = st.session_state.agent.stream(
                    {"messages": messages_for_agent},
                    stream_mode="values",
                )
                
                final_answer = ""
                source_documents = []
                
                for event in stream_events:
                    last_message = event["messages"][-1]
                    
                    if last_message.type == "tool":
                        source_documents = last_message.artifact
                    
                    if last_message.type == "ai" and not last_message.tool_calls:
                        final_answer = last_message.content
            
            # Display answer
            st.markdown(final_answer)
           
            # Display sources
            source_list = None
            if source_documents:
                source_list = [
                f"{doc.metadata.get('document_source', 'N/A')} (Slide {doc.metadata.get('slide_number', 'N/A')})"
                for doc in source_documents
]
                source_list = sorted(list(set(source_list)))
                
                with st.expander("📄 View Sources"):
                    for source in source_list:
                        st.write(f"• {source}")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sources": source_list
            })

# --- SIDEBAR ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **AWS Bedrock** (Claude Sonnet 4.5 & Titan Embeddings)
    - **FAISS** vector search
    - **LangChain** agents
    - **OCR** for image-based PDFs
    """)
    
    st.divider()
   
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.metric("Messages", len(st.session_state.messages))