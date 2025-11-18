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

def convert_pdf_to_images(pdf_path, output_folder="slide_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    print(f"Processing {pdf_path}...")

    # 2. Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through every page
        for i, page in enumerate(pdf.pages, start=1):
            
            #converting pages to image, setting resolution
            image_obj = page.to_image(resolution=300)
            
            #accessing pillow image attribute for our pageimage object
            # pil_image = image_obj.original

            #saving the file path for the image
            image_filename = f"slide_{i}.png"
            save_path = os.path.join(output_folder, image_filename)
            
            
            #convert to rgb to remove transparancy from pdfs
            # pil_image.convert("RGB").save(save_path, format="JPEG")
           
            image_obj.save(save_path, format='PNG')

            print(f"Saved: {save_path}")

    print(f"Done All images saved in '{output_folder}'")


    def get_embedding_model():
        """Sets up the AWS Bedrock client for the Titan V2 embedding model."""
    session = boto3.Session(
        profile_name='devan2',
        region_name='us-east-1'
    )
    bedrock_client = session.client('bedrock-runtime')
    client = BedrockEmbeddings(
        client=bedrock_client,  
        model_id="amazon.titan-embed-text-v2:0")
    return client

if __name__ == "__main__":
    pdf_file = "LILAIRE.pdf"
    convert_pdf_to_images(pdf_file)




   