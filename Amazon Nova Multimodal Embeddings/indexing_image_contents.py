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
        # page_obj = document[page]
        slide_text = page_obj.get_text('text')

        #will be using for image text
        # loaded_page = document.load_page(page)
        loaded_page_pix_map = page_obj.get_pixmap(dpi=300)

        # loaded_page_pix_map.save(f"{output_folder}/{slide_show_name}_slide_{page+1}.png")
        #converting our pixmap to bytes, pixmap not usable to aws, or pillow
        picture_in_bytes = loaded_page_pix_map.tobytes('png')

        #aws requires json format and we cannot send raw binary 0s and 1s in JSON
        #base64 allows us to convert the binary into into a byte string
        #then we encode with utf-8 in a format we can use for our AWS model
        encoded_bytes = base64.b64encode(picture_in_bytes).decode('utf-8')

        #now we also need to save the actual image file (not bytes) into our RAM for streamlit
        saved_image = io.BytesIO(picture_in_bytes)

        #appending the image info to list of dicts
        #slide numebr, the base64 string in utf8 format for aws
        #the image in bytes for streamlit
        image_data_list.append({
            'slide_numer':slide_num,
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



def get_multimodal_vector(bedrock_client,base64_string):
    """
    Sends a Base64 image string to Amazon Titan Multimodal 
    and returns the vector (list of floats).
    """
    #dumps converts dict to string
    body = json.dumps({
        'inputImage': base64_string,
        'embeddingConfig': {
            'outputEmbeddingLength': 3072
        }
    })

    try:
        #invoke model is the api request
        response = bedrock_client.invoke_model(
            body = body,
            modelId="amazon.nova-2-multimodal-embeddings-v1:0",

            #headers of the request
            accept="application/json",
            contentType="application/json"
        )

        #loads converts response string to dictionary
        #boto3 is streaming body request, continuous stream
        #.read takes all the data out of the stream into computer memory as bytes
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get('embedding')

        return embedding
    
    except Exception as e:
        print(f"error getting image embedding {e}")
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
        # FAISS.from_documents handles the loop and embedding for us
        text_vector_store = FAISS.from_documents(documents=text_documents, embeddings =text_embedding_model)

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
           
                #making document object containing the actual information, not j the vectors
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
            #making matrix
            dimensions_image_embeddings = 3072
            index = faiss.IndexFlatL2(dimensions_image_embeddings)

            image_vector_store = FAISS(embedding_function=None, index = index, docstore=InMemoryDocstore(), index_to_docstore_id={})

            image_vector_store.add_embeddings(
                text_embeddings=list(zip([""] * len(image_vectors), image_vectors)),
                metadatas=[d.metadata for d in image_documents],
                ids=[str(i) for i in range(len(image_documents))]
            )
            
            # 4. Save
            image_vector_store.save_local("experiment/my_image_index")
            print("Visual Index saved to 'experiment/my_image_index'")
            
