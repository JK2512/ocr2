
Create a Virtual Environment
python -m venv ocr_env
source ocr_env/bin/activate  # On Windows use: ocr_env\Scripts\activate


Install Required Libraries: Use pip to install necessary libraries
pip install
torch
torchvision 
transformers
gradio 
streamlit 
pillow
numpy
easyocr

Run the Application:
first run the file solely then write this command and it will open web application
streamlit run ocr_app.py

