Aadhar Card Data Extraction and Validation

This project involves extracting textual data from Aadhar card images using OCR (Optical Character Recognition) and validating the authenticity of these cards using a Convolutional Neural Network (CNN).

Features

OCR for Text Extraction: Utilizes pytesseract to extract text data from images, supporting multiple languages including Hindi and English.
Image Preprocessing: Converts images to grayscale and applies thresholding to enhance OCR accuracy.
CNN for Validation: A deep learning model designed to classify images as either real or fake Aadhar cards.
Installation

Clone the repository:
bash
Copy code
git clone https://github.com/your-username/aadhar-card-data-extraction.git
cd aadhar-card-data-extraction
Install dependencies:
Ensure you have Python installed. Then, install the required packages:
bash
Copy code
pip install -r requirements.txt
(The requirements.txt file should include all necessary libraries such as opencv-python, pytesseract, tensorflow, pandas, etc.)
Setup Tesseract OCR:
Download and install Tesseract OCR from Tesseract OCR. Update the path in your script:
python
Copy code
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Usage

Extracting Aadhar Data
To extract data from an Aadhar card image, use the extract_aadhar_data function:

python
Copy code
image_path = 'path_to_your_aadhar_image.jpg'
aadhar_data = extract_aadhar_data(image_path)
print(aadhar_data)
Training the CNN Model
Prepare your dataset: Organize your dataset into training and testing sets, with separate folders for real and fake Aadhar card images.
Train the model:
python
Copy code
improved_model = create_improved_cnn_model()
improved_model.fit(training_set, steps_per_epoch=120, epochs=15, validation_data=test_set, validation_steps=40)
improved_model.save('improved_aadhar_cnn_model.keras')
Predicting Aadhar Card Authenticity
Load the trained model and make predictions on new images:

python
Copy code
loaded_model = tf.keras.models.load_model('improved_aadhar_cnn_model.keras')

prediction = preprocess_and_predict('path_to_image.jpg')
print(f"The card is predicted as: {prediction}")
Results

The CNN model achieved a validation accuracy of approximately 77.43%. Further tuning and a larger dataset could potentially improve performance.


