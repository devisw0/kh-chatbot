import pdfplumber
import os
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock 
from langchain_community.vectorstores import FAISS
import os
from langchain.tools import tool
from langchain.agents import create_agent 
from langchain_core.messages import HumanMessage, AIMessage
import pymupdf
import pytesseract
from PIL import Image
from io import BytesIO
import base64
import io
import json
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import pymupdf
import tempfile
import shutil
import streamlit as st
import numpy as np
import requests 

# REMOVE these DeepEval imports - not needed anymore!
# from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
# from deepeval.test_case import LLMTestCase
# from deepeval.models import AmazonBedrockModel

PROFILE_NAME = "devan2"
REGION_NAME = "us-east-1"


def convert_pdf_pages_to_text_and_images(path, slide_show_name):

    image_data_list = []
    text_data_list = []

    #opening the document
    document = pymupdf.open(path)

    for i, page_obj in enumerate(document):
        slide_num = i+1
        #getting digital text along with other information
        slide_text = page_obj.get_text('text')

        #will be using for image text
        loaded_page_pix_map = page_obj.get_pixmap(dpi=140)

        #converting our pixmap to bytes, pixmap not usable to aws, or pillow
        picture_in_bytes = loaded_page_pix_map.tobytes('png')

        #aws requires json format and we cant send raw binary 0s and 1s in JSON
        #base64 allows us to convert the binary into into a byte string
        #and then we encode with utf-8 in a format we can use for the AWS model, no loss
        encoded_bytes = base64.b64encode(picture_in_bytes).decode('utf-8')

        #now we also need to save the actual image file (not bytes) into RAM for streamlit
        saved_image = io.BytesIO(picture_in_bytes)

        #appending the image info to list of dicts
        #slide number, the base64 string in utf8 format for aws
        #the image in bytes for streamlit
        image_data_list.append({
            'slide_number':slide_num,
            'image_base64': encoded_bytes,
            'image_bytes': picture_in_bytes,
            'source':slide_show_name
        })

        #now saving text data (text and ocr)

        #opening as pil image object
        pil_image = Image.open(saved_image)

        #ocr extraction
        ocr_text = pytesseract.image_to_string(pil_image)

        #combining and specifying the ocr and digital text
        full_text_content = f"Digital Text:\n{slide_text}\n\nOCR Text:\n{ocr_text}"

        text_data_list.append({
            'slide_number':slide_num,
            'content':full_text_content,
            'source':slide_show_name
        })

    return image_data_list, text_data_list


#client for the amazon titan embed text v2 model
def get_text_embedding_model():
    """
    Creates the LangChain wrapper for Amazon Titan Text v2.
    Used by FAISS.from_documents()
    """
    session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    bedrock_client = session.client('bedrock-runtime')
    return BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")


#used to create boto3 client, will have to use this as a helper since no built in langchain support
def get_boto_client():
    """
    Creates a raw Boto3 client.
    Used for manual API calls to Titan Multimodal.
    """
    session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    return session.client('bedrock-runtime')


def get_multimodal_vector(bedrock_client, base64_string):
    """
    Sends a Base64 image string to Amazon Nova Multimodal Embeddings 
    and returns the vector (list of floats).
    """
    #dumps converts dict to string - Nova uses a different schema
    body = json.dumps({
        'schemaVersion': 'nova-multimodal-embed-v1',
        'taskType': 'SINGLE_EMBEDDING',
        'singleEmbeddingParams': {
            'embeddingPurpose': 'GENERIC_INDEX',  # For indexing slides
            'embeddingDimension': 3072,
            'image': {
                'format': 'png',
                'source': {'bytes': base64_string}
            }
        }
    })

    try:
        #invoke model is the api request
        response = bedrock_client.invoke_model(
            body=body,
            modelId="amazon.nova-2-multimodal-embeddings-v1:0",
            #headers of the request
            accept="application/json",
            contentType="application/json"
        )
        #loads converts response string to dictionary
        #boto3 is streaming body request, continuous stream
        #.read takes all the data out of the stream into computer memory as bytes
        response_body = json.loads(response.get('body').read())
        
        # Nova returns embeddings in a different structure
        embedding = response_body.get('embeddings', [{}])[0].get('embedding')
        return embedding
    except Exception as e:
        print(f"error getting image embedding {e}")
        return None


def get_multimodal_text_vector(bedrock_client, text_query):
    """
    Sends a text query to Amazon Nova Multimodal Embeddings 
    and returns the vector (list of floats) for querying image index.
    """
    #dumps converts dict to string - Nova uses a different schema
    body = json.dumps({
        'schemaVersion': 'nova-multimodal-embed-v1',
        'taskType': 'SINGLE_EMBEDDING',
        'singleEmbeddingParams': {
            'embeddingPurpose': 'IMAGE_RETRIEVAL',  # For searching image index
            'embeddingDimension': 3072,
            'text': {
                'truncationMode': 'END',
                'value': text_query
            }
        }
    })

    try:
        #invoke model is the api request
        response = bedrock_client.invoke_model(
            body=body,
            modelId="amazon.nova-2-multimodal-embeddings-v1:0",
            #headers of the request
            accept="application/json",
            contentType="application/json"
        )
        #loads converts response string to dictionary
        response_body = json.loads(response.get('body').read())
        
        # Nova returns embeddings in a different structure
        embedding = response_body.get('embeddings', [{}])[0].get('embedding')
        return embedding
    except Exception as e:
        print(f"error getting text embedding for image search {e}")
        return None


def create_dual_vector_stores(text_data_list, image_data_list):
    """
    Takes extracted text and image data, creates two separate vector stores,
    and saves them to disk.
    """
    print(f"Building Text Index with {len(text_data_list)} documents...")
    
    if text_data_list:
        text_documents = []

        #converting each of the objects in the text_data_list into document objects (later for FAISS)
        for item in text_data_list:
            doc = Document(
                #text_data_list is list of dictionaries with content, slide_number, source parameters. type is for metadata
                page_content=item['content'],
                metadata={
                    'slide_number': item['slide_number'],
                    'source': item['source'],
                    'type': 'text'
                }
            )
            #appending the document objects to new list
            text_documents.append(doc)

        # initializing instance of our text embedding client
        text_embedding_model = get_text_embedding_model()

        # making faiss object using document objects, specifically the list we made and giving it the client
        # FAISS.from_documents handles embeddings
        text_vector_store = FAISS.from_documents(documents=text_documents, embedding=text_embedding_model)

        #saving index locally here
        text_vector_store.save_local("experiment/my_text_index")
        print("Text Index saved to 'my_text_index'")

    #For Visual Embeddings:
    visual_embedding_boto = get_boto_client()

    #to hold the vectors
    image_vectors = []

    #to hold the metadata
    image_documents = []

    for image_data in image_data_list:
        #getting embeddings, passing in boto client and base64 string in list of dicts
        vector = get_multimodal_vector(bedrock_client=visual_embedding_boto, base64_string=image_data['image_base64'])

        if vector:
            image_vectors.append(vector)
       
            #making document object containing the actual information, not just the vectors
            #used for the actual content the vectors point to
            doc = Document(page_content=f"Slide {image_data['slide_number']} from {image_data['source']}",
                    metadata={
                        'slide_number': image_data['slide_number'],
                        'source': image_data['source'],
                        'image_bytes': image_data['image_bytes'] 
                    }
                )
            image_documents.append(doc)

    if image_data_list:
        if image_vectors:
            #making matrix for FAISS library
            dimensions_image_embeddings = 3072
            index = faiss.IndexFlatL2(dimensions_image_embeddings)

            #already have embeddings, index for it is used for size + method, docstore is to create empty storage in RAM, index_to_docstore_id is to keep the indexes empty
            image_vector_store = FAISS(embedding_function=None, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

            image_vector_store.add_embeddings(
                text_embeddings=list(zip([""] * len(image_vectors), image_vectors)),
                metadatas=[d.metadata for d in image_documents],
                ids=[str(i) for i in range(len(image_documents))]
            )
            
            # Save
            image_vector_store.save_local("experiment/my_image_index")
            print("Visual Index saved to 'experiment/my_image_index'")


def load_vector_stores():
    """
    Loads existing FAISS indexes from disk.
    Returns text_store, image_store (can be None if not found)
    """
    text_store = None
    image_store = None
   
    try:
        if os.path.exists("experiment/my_text_index"):
            text_embedding_model = get_text_embedding_model()
            text_store = FAISS.load_local(
                "experiment/my_text_index", 
                embeddings=text_embedding_model,
                allow_dangerous_deserialization=True
            )
            print("Text index loaded successfully")
    except Exception as e:
        print(f"Error loading text index: {e}")
   
    try:
        if os.path.exists("experiment/my_image_index"):
            # For image index, we need to load without embedding function
            dimensions_image_embeddings = 3072
            #l2 matrix
            index = faiss.IndexFlatL2(dimensions_image_embeddings)
            
            image_store = FAISS.load_local(
                "experiment/my_image_index",
                embeddings=None,
                allow_dangerous_deserialization=True
            )
            print("Image index loaded successfully")
    except Exception as e:
        print(f"Error loading image index: {e}")
    
    return text_store, image_store


def query_text_index(text_store, query, k=2):
    """
    Query the text FAISS index and return results with similarity scores.
    Returns list of tuples: (Document, distance_score)
    """
    if text_store is None:
        return []
    
    # similarity_search_with_score returns (Document, score)
    results = text_store.similarity_search_with_score(query, k=k)
    return results


def query_image_index(image_store, query, k=2):
    """
    Query the image FAISS index using text query.
    Converts text to multimodal embedding first.
    Returns list of tuples: (Document, distance_score)
    """
    if image_store is None:
        return []
    
    # Get multimodal embedding for the text query
    bedrock_client = get_boto_client()
    query_vector = get_multimodal_text_vector(bedrock_client, query)
    
    if query_vector is None:
        return []
    
    # Convert to numpy array for FAISS
    query_vector_np = np.array([query_vector], dtype=np.float32)
    
    # Search the image index
    distances, indices = image_store.index.search(query_vector_np, k)
   
    # Get documents and combine with distances
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # Valid result
            doc_id = image_store.index_to_docstore_id.get(idx)
            if doc_id:
                doc = image_store.docstore.search(doc_id)
                results.append((doc, distances[0][i]))
    
    return results


def distance_to_similarity_percentage(distance):
    """
    Convert FAISS L2 distance to similarity percentage.
    Lower distance = higher similarity
    """
    # Using exponential decay: similarity = e^(-distance/scale)
    # Scale factor of 2.0 works well for normalized embeddings
    similarity = np.exp(-distance / 2.0) * 100
    return min(100, max(0, similarity))


def evaluate_response(query, context, response, retrieval_type):
    """
    Evaluate Claude's response by calling the FastAPI evaluation service.
    Instead of running DeepEval locally, we send the data to our API.
    
    Returns dict with scores or None if API call fails.
    """
    try:
        # API endpoint (make sure your FastAPI server is running on port 8000)
        api_url = "http://localhost:8000/evaluate"
        
        # Package data into the format the API expects
        # This matches the EvalRequest model we defined in the API
        payload = {
            "query": query,              # User's question
            "context": context,          # Retrieved context (text or image descriptions)
            "response": response,        # Claude's response we're evaluating
            "eval_type": retrieval_type, # "text", "image", or "combined"
            "threshold": 0.7             # Minimum passing score
        }
        
        # Send POST request to API
        # timeout=60 because evaluations can take 10-30 seconds
        api_response = requests.post(api_url, json=payload, timeout=60)
        
        # Check if request was successful (status code 200)
        api_response.raise_for_status()
        
        # Parse JSON response from API
        # API returns: {faithfulness: 0.92, answer_relevancy: 0.85, ...}
        metrics = api_response.json()
        
        # Convert API response to match our existing format
        # This ensures compatibility with display_evaluation_metrics()
        return {
            'faithfulness': metrics['faithfulness'],
            'answer_relevancy': metrics['answer_relevancy'],
            'contextual_relevancy': metrics['contextual_relevancy'],
            'faithfulness_reason': metrics['faithfulness_reason'],
            'answer_relevancy_reason': metrics['answer_relevancy_reason'],
            'contextual_relevancy_reason': metrics['contextual_relevancy_reason']
        }
        
    except requests.exceptions.Timeout:
        # API took too long (>60 seconds)
        print(f"Evaluation API timeout - request took longer than 60 seconds")
        return None
        
    except requests.exceptions.ConnectionError:
        # Can't reach API (is it running? Check http://localhost:8000/health)
        print(f"Cannot connect to evaluation API - is it running on port 8000?")
        return None
        
    except requests.exceptions.HTTPError as e:
        # API returned an error (4xx or 5xx status code)
        print(f"Evaluation API error: {e}")
        return None
        
    except Exception as e:
        # Any other unexpected error
        print(f"Unexpected evaluation error: {e}")
        return None


def display_evaluation_metrics(metrics, label=""):
    """Display evaluation metrics in an expandable section"""
    if not metrics:
        return
    
    # Count how many metrics we have (3 required + 2 optional)
    required_metrics = ['faithfulness', 'answer_relevancy', 'contextual_relevancy']
    optional_metrics = ['contextual_recall', 'factual_correctness']
    
    # Calculate average only from metrics that exist
    scores = [metrics[m] for m in required_metrics]
    
    # Add optional metrics to average if they exist and are not None
    if metrics.get('contextual_recall') is not None:
        scores.append(metrics['contextual_recall'])
    if metrics.get('factual_correctness') is not None:
        scores.append(metrics['factual_correctness'])
    
    avg_score = sum(scores) / len(scores)
    
    # Determine overall color
    if avg_score >= 0.8:
        color = "🟢"
    elif avg_score >= 0.6:
        color = "🟡"
    else:
        color = "🔴"
    
    with st.expander(f"{color} 📊 Evaluation Metrics {label}"):
        # Calculate number of columns needed (3 required + up to 2 optional)
        num_cols = 3
        if metrics.get('contextual_recall') is not None:
            num_cols += 1
        if metrics.get('factual_correctness') is not None:
            num_cols += 1
        
        # Create columns dynamically
        cols = st.columns(num_cols)
        col_idx = 0
        
        # Always show these 3
        with cols[col_idx]:
            st.metric("Faithfulness", f"{metrics['faithfulness']:.1%}")
            if metrics['faithfulness_reason']:
                st.caption(metrics['faithfulness_reason'][:200] + "...")
        col_idx += 1
        
        with cols[col_idx]:
            st.metric("Answer Relevancy", f"{metrics['answer_relevancy']:.1%}")
            if metrics['answer_relevancy_reason']:
                st.caption(metrics['answer_relevancy_reason'][:200] + "...")
        col_idx += 1
        
        with cols[col_idx]:
            st.metric("Contextual Relevancy", f"{metrics['contextual_relevancy']:.1%}")
            if metrics['contextual_relevancy_reason']:
                st.caption(metrics['contextual_relevancy_reason'][:200] + "...")
        col_idx += 1
        
        # Show contextual recall if it exists (only in test evaluation)
        if metrics.get('contextual_recall') is not None:
            with cols[col_idx]:
                st.metric("Contextual Recall", f"{metrics['contextual_recall']:.1%}")
                if metrics.get('contextual_recall_reason'):
                    st.caption(metrics['contextual_recall_reason'][:200] + "...")
            col_idx += 1
        
        # Show factual correctness if it exists (only in test evaluation)
        if metrics.get('factual_correctness') is not None:
            with cols[col_idx]:
                st.metric("Factual Correctness", f"{metrics['factual_correctness']:.1%}")
                # Ragas doesn't return reasons like DeepEval, so skip caption


def get_claude_response(query, context, retrieval_type, image_results=None):
    """
    Generate Claude response using retrieved context.
    retrieval_type: 'text' or 'image' to customize the prompt
    image_results: list of (Document, score) tuples with image_bytes in metadata
    """
    session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    bedrock_client = session.client('bedrock-runtime')
    
    # Customize prompt based on retrieval type
    if retrieval_type == 'text':
        system_prompt = """You are a helpful assistant analyzing presentation slides. 
You have been given text content extracted from slides (both digital text and OCR from images).
Answer the user's question based on this text content. Be specific and cite slide numbers when relevant."""
        
        prompt = f"""Context from slides:
{context}

User question: {query}

Please provide a helpful answer based on the context above."""
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        })
        
    else:  # image retrieval - send actual images
        system_prompt = """You are a helpful assistant analyzing presentation slides.
You have been given the actual slide images to analyze visually.
Answer the user's question based on what you see in these slides. Be specific and cite slide numbers when relevant."""
       
        # Build content array with images
        content = []
        
        # Add the query first
        content.append({"type": "text", "text": f"Please analyze these slides and answer: {query}"})
        
        # Add each retrieved slide image
        if image_results:
            for doc, score in image_results:
                content.append({"type": "text", "text": f"\nSlide {doc.metadata['slide_number']} from {doc.metadata['source']}:"})
                
                # Add the actual image
                if 'image_bytes' in doc.metadata:
                    image_base64 = base64.b64encode(doc.metadata['image_bytes']).decode('utf-8')
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    })
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}]
        })
    
    try:
        response = bedrock_client.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
        return answer
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def get_combined_claude_response(query, text_context, image_context, image_results=None):
    """
    Generate single Claude response using both text and image contexts combined.
    """
    session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    bedrock_client = session.client('bedrock-runtime')
    
    system_prompt = """You are a helpful assistant analyzing presentation slides.
You have been given both text content (digital text + OCR) AND the actual slide images.
Synthesize information from both sources to provide a comprehensive answer. Be specific and cite slide numbers when relevant."""
    
    # Build content array with both text and images
    content = []
    
    # Add the query
    content.append({
        "type": "text",
        "text": f"""Please analyze these slides using both the extracted text and visual content to answer: {query}

Text content from slides:
{text_context}"""
    })
    
    # Add the actual slide images
    if image_results:
        content.append({"type": "text", "text": "\nSlide images for visual analysis:"})
        
        for doc, score in image_results:
            content.append({"type": "text", "text": f"\nSlide {doc.metadata['slide_number']} from {doc.metadata['source']}:"})
            
            if 'image_bytes' in doc.metadata:
                image_base64 = base64.b64encode(doc.metadata['image_bytes']).decode('utf-8')
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64
                    }
                })
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}]
        })
        
        response = bedrock_client.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
            body=body
        )
      
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
        return answer
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Streamlit App

def main():
    st.set_page_config(page_title="Dual-RAG Presentation Chatbot", layout="wide")
    
    st.title("🎯 Dual-RAG Presentation Chatbot")
    st.markdown("Upload presentations and ask questions. Compare answers from text vs. visual embeddings!")
    
    # Initialize session state
    if 'text_store' not in st.session_state:
        st.session_state.text_store = None
    if 'image_store' not in st.session_state:
        st.session_state.image_store = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = True
    if 'indexes_loaded' not in st.session_state:
        st.session_state.indexes_loaded = False
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Settings")
        
        # Comparison mode toggle
        st.session_state.comparison_mode = st.toggle(
            "Comparison Mode", 
            value=st.session_state.comparison_mode,
            help="Show side-by-side comparison of text vs. image retrieval"
        )
        
        st.divider()
        
        st.header("Upload Presentations")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Max 50MB per file"
        )
        
        if uploaded_files:
            if st.button("🚀 Process PDFs", type="primary"):
                with st.spinner("Processing PDFs and creating embeddings..."):
                    all_text_data = []
                    all_image_data = []
                    
                    for uploaded_file in uploaded_files:
                        # Check file size (50MB = 52428800 bytes), just set some generic limit not based on any guidelines
                        if uploaded_file.size > 52428800:
                            st.error(f"{uploaded_file.name} exceeds 50MB limit")
                            continue
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        try:
                            st.info(f"Processing {uploaded_file.name}...")
                            image_data, text_data = convert_pdf_pages_to_text_and_images(tmp_path, uploaded_file.name)
                            all_image_data.extend(image_data)
                            all_text_data.extend(text_data)
                            st.success(f"{uploaded_file.name} processed")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            os.unlink(tmp_path)
                    
                    if all_text_data and all_image_data:
                        st.info("Creating vector indexes...")
                        create_dual_vector_stores(all_text_data, all_image_data)
                        st.session_state.text_store, st.session_state.image_store = load_vector_stores()
                        st.session_state.indexes_loaded = True
                        st.success("Indexes created and loaded!")
                        st.rerun()
        
        st.divider()
        
        # Load existing indexes button
        if st.button("📂 Load Existing Indexes"):
            with st.spinner("Loading indexes..."):
                st.session_state.text_store, st.session_state.image_store = load_vector_stores()
                if st.session_state.text_store or st.session_state.image_store:
                    st.session_state.indexes_loaded = True
                    st.success("Indexes loaded!")
                    st.rerun()
                else:
                    st.error("No indexes found. Please upload PDFs first.")
        
        # Clear indexes button
        if st.button("🗑️ Clear Indexes", type="secondary"):
            if st.session_state.text_store or st.session_state.image_store:
                # Clear from session
                st.session_state.text_store = None
                st.session_state.image_store = None
                st.session_state.indexes_loaded = False
                st.session_state.chat_history = []
                
                # Delete files
                if os.path.exists("experiment/my_text_index"):
                    shutil.rmtree("experiment/my_text_index")
                if os.path.exists("experiment/my_image_index"):
                    shutil.rmtree("experiment/my_image_index")
                
                st.success("Indexes cleared!")
                st.rerun()
        
        # Index status
        st.divider()
        st.header("Index Status")
        if st.session_state.text_store:
            st.success("Text Index Loaded")
        else:
            st.warning("No Text Index")
        
        if st.session_state.image_store:
            st.success("Image Index Loaded")
        else:
            st.warning("No Image Index")
    
    # Main chat area
    if not st.session_state.indexes_loaded:
        st.info("👈 Please upload PDFs or load existing indexes to start chatting!")
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                if message.get('comparison_mode'):
                    # Display comparison view
                    col1, col2 = st.columns(2)
                   
                    with col1:
                        st.markdown("**Text Retrieval**")
                        st.markdown(message['text_response'])
                        
                        if message.get('text_metrics'):
                            display_evaluation_metrics(message['text_metrics'], "(Text)")
                        
                        if message['text_results']:
                            with st.expander("View Retrieved Text Chunks"):
                                for i, (doc, score) in enumerate(message['text_results'], 1):
                                    similarity = distance_to_similarity_percentage(score)
                                    st.markdown(f"**Slide {doc.metadata['slide_number']}** from *{doc.metadata['source']}* (Similarity: {similarity:.1f}%)")
                                    st.text(doc.page_content[:300] + "...")
                                    st.divider()
                    
                    with col2:
                        st.markdown("**Image Retrieval**")
                        st.markdown(message['image_response'])
                        
                        if message.get('image_metrics'):
                            display_evaluation_metrics(message['image_metrics'], "(Image)")
                        
                        if message['image_results']:
                            with st.expander("View Retrieved Slides"):
                                for i, (doc, score) in enumerate(message['image_results'], 1):
                                    similarity = distance_to_similarity_percentage(score)
                                    st.markdown(f"**Slide {doc.metadata['slide_number']}** from *{doc.metadata['source']}* (Similarity: {similarity:.1f}%)")
                                    
                                    # Display slide image
                                    if 'image_bytes' in doc.metadata:
                                        image = Image.open(io.BytesIO(doc.metadata['image_bytes']))
                                        st.image(image, use_container_width=True)
                                    st.divider()
                else:
                    # Display simple combined view
                    st.markdown(message['content'])
                    
                    if message.get('combined_metrics'):
                        display_evaluation_metrics(message['combined_metrics'])
                    
                    # Re-render images/scores from history
                    
                    # Visual Matches (Images)
                    if message.get('image_results'):
                        with st.expander("🔍 View Retrieved Slides (Visual Match)"):
                            for i, (doc, score) in enumerate(message['image_results'], 1):
                                similarity = distance_to_similarity_percentage(score)
                                st.markdown(f"**Slide {doc.metadata.get('slide_number', 'N/A')}** from *{doc.metadata.get('source', 'Unknown')}* (Similarity: {similarity:.1f}%)")
                                
                                if 'image_bytes' in doc.metadata:
                                    image = Image.open(io.BytesIO(doc.metadata['image_bytes']))
                                    st.image(image, use_container_width=True)
                                st.divider()

                    # Text Matches (Snippets)
                    if message.get('text_results'):
                        with st.expander("📄 View Retrieved Text (Text Match)"):
                            for i, (doc, score) in enumerate(message['text_results'], 1):
                                similarity = distance_to_similarity_percentage(score)
                                st.markdown(f"**Slide {doc.metadata.get('slide_number', 'N/A')}** from *{doc.metadata.get('source', 'Unknown')}* (Similarity: {similarity:.1f}%)")
                                st.text(doc.page_content[:300] + "...")
                                st.divider()
    
    # Chat input
    user_query = st.chat_input("Ask a question about your presentations...")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({'role': 'user', 'content': user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query both indexes
                text_results = query_text_index(st.session_state.text_store, user_query, k=2)
                image_results = query_image_index(st.session_state.image_store, user_query, k=2)
                
                if st.session_state.comparison_mode:
                    # Comparison mode - two separate responses
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Text Retrieval**")
                        # Prepare context from text results
                        text_context = "\n\n".join([
                            f"Slide {doc.metadata['slide_number']} from {doc.metadata['source']}:\n{doc.page_content}"
                            for doc, _ in text_results
                        ])
                        text_response = get_claude_response(user_query, text_context, 'text')
                        st.markdown(text_response)
                        
                        # Evaluate text response
                        with st.spinner("Evaluating..."):
                            text_metrics = evaluate_response(user_query, text_context, text_response, 'text')
                            if text_metrics:
                                display_evaluation_metrics(text_metrics, "(Text)")
                       
                        if text_results:
                            with st.expander("View Retrieved Text Chunks"):
                                for i, (doc, score) in enumerate(text_results, 1):
                                    similarity = distance_to_similarity_percentage(score)
                                    st.markdown(f"**Slide {doc.metadata['slide_number']}** from *{doc.metadata['source']}* (Similarity: {similarity:.1f}%)")
                                    st.text(doc.page_content[:300] + "...")
                                    st.divider()
                    
                    with col2:
                        st.markdown("**Image Retrieval**")
                        # Prepare context from image results
                        image_context = "\n\n".join([
                            f"Slide {doc.metadata['slide_number']} from {doc.metadata['source']}"
                            for doc, _ in image_results
                        ])
                        image_response = get_claude_response(user_query, image_context, 'image', image_results=image_results)
                        st.markdown(image_response)
                        
                        # Evaluate image response
                        with st.spinner("Evaluating..."):
                            image_metrics = evaluate_response(user_query, image_context, image_response, 'image')
                            if image_metrics:
                                display_evaluation_metrics(image_metrics, "(Image)")
                        
                        if image_results:
                            with st.expander("🔍 View Retrieved Slides"):
                                for i, (doc, score) in enumerate(image_results, 1):
                                    similarity = distance_to_similarity_percentage(score)
                                    st.markdown(f"**Slide {doc.metadata['slide_number']}** from *{doc.metadata['source']}* (Similarity: {similarity:.1f}%)")
                                    
                                    # Display slide image
                                    if 'image_bytes' in doc.metadata:
                                        image = Image.open(io.BytesIO(doc.metadata['image_bytes']))
                                        st.image(image, use_container_width=True)
                                    st.divider()
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'comparison_mode': True,
                        'text_response': text_response,
                        'image_response': image_response,
                        'text_results': text_results,
                        'image_results': image_results,
                        'text_metrics': text_metrics,
                        'image_metrics': image_metrics
                    })
                
                else:
                    # Simple mode - combined response
                    text_context = "\n\n".join([
                        f"Slide {doc.metadata['slide_number']} from {doc.metadata['source']}:\n{doc.page_content}"
                        for doc, _ in text_results
                    ])
                    
                    image_context = "\n\n".join([
                        f"Slide {doc.metadata['slide_number']} from {doc.metadata['source']}"
                        for doc, _ in image_results
                    ])
                   
                    combined_response = get_combined_claude_response(user_query, text_context, image_context, image_results=image_results)
                    st.markdown(combined_response)
                    if image_results:
                        with st.expander("🔍 View Retrieved Slides (Visual Match)"):
                            for i, (doc, score) in enumerate(image_results, 1):
                                similarity = distance_to_similarity_percentage(score)
                                st.markdown(f"**Slide {doc.metadata['slide_number']}** from *{doc.metadata['source']}* (Similarity: {similarity:.1f}%)")
                                if 'image_bytes' in doc.metadata:
                                    image = Image.open(io.BytesIO(doc.metadata['image_bytes']))
                                    st.image(image, use_container_width=True)
                                st.divider()

                    if text_results:
                        with st.expander("📄 View Retrieved Text (Text Match)"):
                            for i, (doc, score) in enumerate(text_results, 1):
                                similarity = distance_to_similarity_percentage(score)
                                st.markdown(f"**Slide {doc.metadata['slide_number']}** from *{doc.metadata['source']}* (Similarity: {similarity:.1f}%)")
                                st.text(doc.page_content[:300] + "...")
                                st.divider()
                    # Evaluate combined response
                    with st.spinner("Evaluating..."):
                        combined_context = f"{text_context}\n\n{image_context}"
                        combined_metrics = evaluate_response(user_query, combined_context, combined_response, 'combined')
                        if combined_metrics:
                            display_evaluation_metrics(combined_metrics)
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'comparison_mode': False,
                        'content': combined_response,
                        'text_results': text_results,
                        'image_results': image_results,
                        'combined_metrics': combined_metrics
                    })


if __name__ == "__main__":
    main()