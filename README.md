🧠 Brain Tumor MRI Classification Using Deep Learning

A deep learning-based medical image classification project that classifies brain MRI images into four categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The project uses a Convolutional Neural Network (CNN) with TensorFlow/Keras for image classification and a Streamlit web application for interactive predictions.

Disclaimer: This project is developed for educational and learning purposes. It is not intended to replace professional medical diagnosis or clinical decision-making.

📌 Project Overview

Brain MRI analysis can be time-consuming and requires specialized medical expertise. This project demonstrates how Deep Learning can be applied to MRI image classification.

The system takes an MRI image as input, preprocesses the image, passes it through the trained CNN model, and returns the predicted brain tumor category along with a confidence score.

The project covers an end-to-end Deep Learning workflow:

Dataset → Preprocessing → Data Augmentation → CNN Model → Training → Evaluation → Model Saving → Streamlit Deployment → Prediction

🎯 Objectives

Develop a Deep Learning model for Brain MRI image classification.

Classify MRI images into four categories.

Preprocess MRI images by resizing and normalization.

Train and evaluate a CNN model using TensorFlow and Keras.

Save the trained model for future predictions.

Build an interactive Streamlit web application.

Allow users to upload an unseen MRI image and receive a prediction with confidence.

🗂️ Dataset

The project uses a Brain Tumor MRI dataset containing MRI images organized into four classes.

Classes

Class

Description

Glioma

MRI images containing Glioma tumors

Meningioma

MRI images containing Meningioma tumors

Pituitary

MRI images containing Pituitary tumors

No Tumor

MRI images without a brain tumor

The project report identifies the dataset as the Brain Tumor MRI Dataset from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset is organized into:

dataset/
├── Training/
└── Testing/

🏗️ Project Structure

Medical_Image_Classification/
│
├── dataset/
│   ├── Training/
│   └── Testing/
│
├── models/
│   ├── brain_tumor_model.keras
│   └── final_brain_tumor_model.keras
│
├── notebooks/
│   └── Brain_Tumor_Classification.ipynb
│
├── sample_images/
│   └── Test.jpg
│
├── venv/
│   └── # Local Python virtual environment - should not be uploaded to GitHub
│
├── .gitignore
└── README.md

Note: The Streamlit application file is not visible in the folder structure shown with this project documentation. If your Streamlit file has a different name/location, use that filename in the run command below.

🛠️ Technologies Used

Python 3.10.11

TensorFlow 2.21.0

Keras

NumPy

Matplotlib

Pillow (PIL)

Streamlit

Jupyter Notebook

Visual Studio Code

Git & GitHub

🔄 Methodology

1. Dataset Collection

MRI images are collected and organized into training and testing directories.

2. Image Preprocessing

The images are:

Resized to 224 × 224 pixels

Converted into image arrays

Normalized by scaling pixel values to the range 0–1

Example preprocessing used in the project:

img = img.resize((224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

3. Data Augmentation

Data augmentation is applied during the training process to improve model generalization and reduce overfitting.

4. CNN Model

A Convolutional Neural Network automatically learns visual features from MRI images, including patterns, textures, and shapes.

The CNN performs:

Input MRI Image
       ↓
Image Preprocessing
       ↓
Convolutional Layers
       ↓
Pooling Layers
       ↓
Feature Extraction
       ↓
Dense / Classification Layers
       ↓
Four-Class Prediction

5. Model Training

The model was trained using TensorFlow/Keras.

The project report specifies:

Optimizer: Adam

Loss: Categorical Cross-Entropy

Metric: Accuracy

Epochs: 10

Example:

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

6. Model Evaluation

The project evaluates the model using training and validation performance, including:

Training Accuracy

Validation Accuracy

Training Loss

Validation Loss

The project report also discusses evaluation using unseen MRI images.

🤖 Prediction Workflow

When a user uploads an MRI image through the Streamlit application:

Upload MRI Image
       ↓
Resize to 224 × 224
       ↓
Normalize Pixel Values
       ↓
Load Trained CNN Model
       ↓
Generate Prediction
       ↓
Find Predicted Class
       ↓
Display Class + Confidence

The prediction logic uses the class with the highest model probability:

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

🌐 Streamlit Web Application

The trained model is integrated into a Streamlit web application.

The application allows users to:

Upload an MRI image.

View the uploaded image.

Automatically preprocess the image.

Generate a prediction using the trained CNN.

Display the predicted tumor category.

Display the confidence percentage.

Run the Streamlit Application

First activate your virtual environment.

Windows

venv\Scripts\activate

Then run your Streamlit application:

streamlit run <your_streamlit_file>.py

For example, if your Streamlit file is named app.py:

streamlit run app.py

The application will then open in your browser.

📦 Installation

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Medical_Image_Classification

2. Create a virtual environment

python -m venv venv

3. Activate the virtual environment

Windows:

venv\Scripts\activate

4. Install dependencies

If a requirements.txt file is available:

pip install -r requirements.txt

Otherwise, install the main project libraries:

pip install tensorflow numpy matplotlib pillow streamlit jupyter

5. Run the application

streamlit run <your_streamlit_file>.py

📓 Jupyter Notebook

The notebook used for the project is:

notebooks/Brain_Tumor_Classification.ipynb

It contains the Deep Learning workflow for working with the MRI dataset, preprocessing, model development/training, evaluation, and prediction.

💾 Trained Models

The repository contains trained Keras model files:

models/
├── brain_tumor_model.keras
└── final_brain_tumor_model.keras

The project report shows the final model being loaded with:

from tensorflow.keras.models import load_model

model = load_model("models/final_brain_tumor_model.keras")

📊 Sample Prediction Results

The project report presents sample predictions on unseen MRI images:

Test Image

Predicted Class

Confidence

MRI Image 1

Glioma

98.45%

MRI Image 2

Meningioma

97.82%

MRI Image 3

Pituitary

99.10%

MRI Image 4

No Tumor

99.52%

These values are the sample results documented in the project report and should not be interpreted as clinical accuracy or real-world diagnostic performance.

📁 Sample Image

A sample test image is included in:

sample_images/Test.jpg

You can use it to test the Streamlit application's upload and prediction workflow.

🚀 Future Scope

Possible future improvements include:

Training on a larger and more diverse MRI dataset.

Using advanced architectures such as ResNet, DenseNet, or EfficientNet.

Implementing tumor segmentation to identify the tumor location.

Adding Explainable AI (XAI), such as Grad-CAM.

Hyperparameter tuning and transfer learning.

Deploying the application to cloud platforms.

Developing a mobile application.

Integrating with hospital information systems and PACS.

⚠️ Medical Disclaimer

This application is an educational Deep Learning project and is not a medical diagnostic tool.

Predictions generated by the model should not be used as a substitute for examination, diagnosis, or treatment by a qualified medical professional.

👨‍💻 Author

Devanshu Wankhade

Brain Tumor MRI Classification Using Deep Learning

Technologies: Python | TensorFlow | Keras | CNN | Streamlit

📚 References

Brain Tumor MRI Dataset — Kaggle
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016.

François Chollet. Deep Learning with Python, 2nd Edition.

Aurélien Géron. Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition.

TensorFlow Documentation
https://www.tensorflow.org/

Streamlit Documentation
https://docs.streamlit.io/
