import pdfplumber
import pandas as pd
import os
import pytesseract  
from PIL import Image

def extract_text_from_pdf_ocr(file_path:str):
    """
    Extracts text from each page of a PDF using OCR (Tesseract).
    This handles pages that are images.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    print(f"Extracting text from {file_path} using OCR...")
    print("This may take a moment...")
    
    slide_data = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start = 1):
            
            #original way (plumber)
            digital_text = page.extract_text()
            
            text = "" #empty text
            
            if digital_text and digital_text.strip():
                text = digital_text.strip()
                print(f"Page {i}: Extracted digital text.")

            else:
                #If no digital text, use OCR on an image of the page
                print(f"Page {i}: No digital text found. Using OCR...")
                
                # Convert the page to an image. (300 DPI for better OCR)
                # .to_image() returns a PIL.Image object
                img = page.to_image(resolution=300).original
                
                try:
                    #text from our images?
                    text = pytesseract.image_to_string(img, lang='eng') #image, and in english
                    text = text.strip()
                except Exception as e:
                    print(f"  Error during OCR on page {i}: {e}")

            #Add to our data
            if text:
                slide_data.append({
                    "SlideNumber": i,
                    "Content": text
                })
            else:
                print(f"  Page {i} appears to be blank.")

    print(f'Text Extraction done, found text on {len(slide_data)} slides')
    return slide_data

def save_to_excel(data, excel_path):
    """
    Saves the extracted data (list of dictionaries) to an Excel file.
    """
    if not data:
        print("No data to save.")
        return
    
    df = pd.DataFrame(data)
    try:

        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Successfully saved the data to {excel_path}")

    except Exception as e:

        print(f"Error saving to Excel: {e}")

def main():
    pdf_file = 'LILAIRE.pdf'
    excel_path = 'Extracted_content_OCR.xlsx'

    extracted_text = extract_text_from_pdf_ocr(pdf_file)

    if extracted_text:
        save_to_excel(extracted_text, excel_path)

if __name__ == "__main__":
    main()