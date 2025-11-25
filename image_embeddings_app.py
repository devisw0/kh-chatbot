import fitz  # PyMuPDF
from PIL import Image
import io
import pymupdf
import base64

def pdf_to_images(pdf_file, zoom_level=2):
    """
    Converts a PDF file into a list of high-quality images.
    
    Args:
        pdf_file: The file object (uploaded by user).
        zoom_level: How much to zoom in (2 = Double Resolution). 
                    Increase this to 3 if small text is being missed.
    
    Returns:
        list: A list of PIL Image objects.
    """
    # 1. Open the PDF from the memory stream
    # stream=pdf_file.read() reads the raw bytes, built into python
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    
    image_list = []
    
    # Loop through every page in the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # setting resolution
        # Matrix(2, 2) means zoom X by 2 and Y by 2.
        mat = pymupdf.Matrix(zoom_level, zoom_level)
        
        # Render the page to pixels (Pixmap) using that matrix
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to a standard Python Image format (PIL)
        # The AI needs PNG format usually, so we convert the raw pixels to PNG bytes
        img_data = pix.tobytes("png")
        
        # Create the actual Image object
        img = Image.open(io.BytesIO(img_data))
        
        image_list.append(img)
        
    return image_list

def image_to_base64(pil_image):
    """
    Wraps the raw image pixel data into a safe text string (Base64)
    so it can be mailed to Amazon Nova.
    """
    # 1. Create a virtual container in RAM
    buffered = io.BytesIO()
    
    # 2. Save the image into that container as a PNG
    # This renders the pixels into compressed PNG file bytes
    pil_image.save(buffered, format="PNG")
    
    # 3. Get the bytes out
    img_bytes = buffered.getvalue()
    
    # 4. Translate bytes -> Base64 String
    base64_string = base64.b64encode(img_bytes).decode("utf-8")
    
    return base64_string

