import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import easyocr
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import AutoTokenizer
import torch

# Load the Hugging Face model and processor
tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten", clean_up_tokenization_spaces=True)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load EasyOCR model for Hindi and English
reader = easyocr.Reader(['en', 'hi'], gpu=False)

# Function to extract text using Hugging Face TrOCR
def extract_text_huggingface(image):
    try:
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        return f"Error in Hugging Face extraction: {str(e)}"

# Function to extract text using EasyOCR
def extract_text_easyocr(image):
    try:
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=1)
        return results  # Return detailed results including bounding boxes
    except Exception as e:
        return f"Error in EasyOCR extraction: {str(e)}"

# Function to highlight keywords in the image
def highlight_keywords(image, results, keyword):
    draw = ImageDraw.Draw(image)
    found = False
    color_toggle = True  # To alternate between colors
    
    for bbox, text, _ in results:
        if keyword.lower() in text.lower():
            left = int(bbox[0][0])
            upper = int(bbox[0][1])
            right = int(bbox[2][0])
            lower = int(bbox[2][1])
            color = "blue" if color_toggle else "darkred"
            draw.rectangle([left, upper, right, lower], outline=color, width=5)  # Highlight
            color_toggle = not color_toggle  # Alternate colors
            found = True
    
    return image, found

# Streamlit application
def main():
    # Set page configuration at the very start
    st.set_page_config(page_title="OCR Application", layout="centered")
    
    st.title("📄 Optical Character Recognition (OCR) with Keyword Search")
    st.markdown("""
    **Welcome to the OCR Application!**  
    **_BY: JIYA KATHURIA_**  
    
    This application extracts text from uploaded images (both Hindi and English) and allows you to search for specific keywords. The matching words will be highlighted on the image.
    
    ### Instructions
    1. Upload an image with Hindi/English text.
    2. Enter a keyword to search within the extracted text.
    3. The application will highlight the matching words in red and blue boxes.
    """)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Extracting text with Hugging Face TrOCR..."):
            extracted_text_hf = extract_text_huggingface(np.array(image))
        
        st.write(f"**Extracted Text (Hugging Face):**\n\n{extracted_text_hf.strip() if extracted_text_hf.strip() else 'No text found.'}")
        
        with st.spinner("Extracting text with EasyOCR..."):
            results_easyocr = extract_text_easyocr(image)
        
        if isinstance(results_easyocr, str):
            st.error(results_easyocr)  # Display error message from EasyOCR
        else:
            st.write(f"**Extracted Text (EasyOCR):**\n\n{', '.join([text[1] for text in results_easyocr]) if results_easyocr else 'No text found.'}")
        
        keyword = st.text_input("Enter keyword to search:")
        if keyword:
            st.write("**Highlighting matching words...**")
            highlighted_image, found = highlight_keywords(image.copy(), results_easyocr, keyword)
            st.image(highlighted_image, caption="Highlighted Image", use_column_width=True)
            
            if found:
                st.success(f"'{keyword}' found in the image!")
            else:
                st.warning(f"'{keyword}' not found in the image.")
    else:
        st.info("Please upload an image to begin the OCR process.")

if __name__ == "__main__":
    main()
