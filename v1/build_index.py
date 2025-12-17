import pandas as pd
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document
from langchain_core.documents import Document
import os
# from langchain_core.vectorstores import FAISS


def get_embedding_model():
    """
        Sets up the AWS Bedrock client with the titan text embeddings v2 model
    """
    
    client = BedrockEmbeddings(
        region_name="us-east-1", 
        model_id="amazon.titan-embed-text-v2:0"
    )
    return client


def load_slide_info_from_excel(path:str):
    """
    Loads your extracted slide data from the Excel file.
    """
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return []
    
    print(f"Loading data from {path}")
    df = pd.read_excel(path)

    documents = []

    for index, row in df.iterrows():

        #string
        content = str(row['Content'])
        
        
        metadata = {
            "slide_number": int(row['SlideNumber'])
        }

        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    print(f"Loaded {len(documents)} documents.")
    return documents


def main():
    excel_file = "Extracted_content_OCR.xlsx"
    vector_store_path = "my_slide_index"

    documents = load_slide_info_from_excel(excel_file)
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    #setting up the embedding cliet
    print("Initializing embedding model")
    embeddings = get_embedding_model()

    #creating and saving the vector store
    print("Embedding documents and building vector store")

    #for all 12 documents (cells/slides) we send to titan model and get back vectors and built the faiss index (db)

    vector_store = FAISS.from_documents(documents, embeddings)

    #saving index
    vector_store.save_local(vector_store_path)
    print(f"Success! Your vector store is built and saved in the folder: '{vector_store_path}'")

if __name__ == '__main__':
    main()