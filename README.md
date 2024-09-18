Image Classification and Text Extraction with MobileNetV2 and PyTesseract

This Streamlit web application presents a robust image processing pipeline that effectively combines the efficiency of MobileNetV2 for image classification with the accuracy of PyTesseract for text extraction. Users can upload an image, and the app will:

Image Classification: Employ MobileNetV2, a pre-trained convolutional neural network, to accurately categorize the image into predefined classes. The model leverages transfer learning to achieve high performance even with limited training data.

Text Extraction: Utilize PyTesseract, an optical character recognition (OCR) tool, to extract text from the same image. The extracted text is cleaned and presented to the user, enabling further analysis or processing.

Key Features:

User-Friendly Interface: The Streamlit app offers an intuitive interface for users to upload images and view classification results and extracted text.
MobileNetV2 Integration: Leverages the efficiency and accuracy of MobileNetV2 for image classification tasks.
PyTesseract OCR: Employs PyTesseract to accurately extract text from images, even in challenging scenarios.
Text Cleaning: Implements text cleaning techniques to enhance the quality of extracted text and improve readability.
Potential Applications:

Document Analysis: Extracting text from scanned documents for data entry or analysis.
Image Captioning: Generating descriptive captions for images based on both visual content and extracted text.
Product Search: Enabling visual search for products by classifying images and extracting relevant text.
Quality Control: Inspecting products for defects by analyzing images and comparing them to reference data.
Installation:

Clone the repository from GitHub.
Install the required dependencies using pip install -r requirements.txt.
Download the pre-trained MobileNetV2 model and PyTesseract library.
Usage:

Run the Streamlit app using streamlit run app.py.
Upload an image to the app.
View the classification results and extracted text.
Contributions:

Contributions to this project are welcome! Please feel free to fork the repository, make changes, and submit a pull request.
