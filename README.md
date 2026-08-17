🧠 Brain Tumor MRI Classification Using Deep Learning

A deep learning-based medical image classification project that classifies brain MRI images into four categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras for image classification and a Streamlit web application for interactive predictions.

⚠️ Medical Disclaimer: This project is developed for educational and learning purposes. It is not intended to replace professional medical diagnosis, clinical examination, or treatment decisions.

📌 Project Overview

Brain MRI analysis can be time-consuming and requires specialized medical expertise. This project demonstrates how Deep Learning can be applied to MRI image classification.

The system takes an MRI image as input, preprocesses the image, passes it through the trained CNN model, and returns the predicted brain tumor category along with a confidence score.

End-to-End Workflow

Dataset → Image Preprocessing → Data Augmentation → CNN Model
→ Training → Evaluation → Model Saving → Streamlit Deployment → Prediction

🎯 Objectives

Develop a Deep Learning model for Brain MRI image classification.

Classify MRI images into four categories.

Resize and normalize MRI images before prediction.

Train and evaluate a CNN model using TensorFlow and Keras.

Save the trained model for future predictions.

Build an interactive Streamlit web application.

Allow users to upload an unseen MRI image.

Display the predicted category and model confidence.

🗂️ Dataset

The project uses a Brain Tumor MRI dataset containing images organized into four classes.

Class

Description

Glioma

MRI images containing glioma tumors

Meningioma

MRI images containing meningioma tumors

Pituitary

MRI images containing pituitary tumors

No Tumor

MRI images without a brain tumor

The project report identifies the dataset as the Brain Tumor MRI Dataset from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset Structure

dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

The repository includes the Testing dataset with approximately 1,600 MRI images organized into the four classification categories.

🏗️ Project Structure

Medical_Image_Classification/
│
├── app.py
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   │
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
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
├── .gitignore
├── requirements.txt
└── README.md

Note: The venv/ virtual environment is used locally and is intentionally excluded from GitHub through .gitignore.

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

MRI images are collected and organized into separate training and testing directories.

2. Image Preprocessing

The images are resized to 224 × 224 pixels, converted into image arrays, and normalized by scaling pixel values to the range 0–1.

Example:

img = img.resize((224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

3. Data Augmentation

Data augmentation is applied during the training process to improve model generalization and reduce overfitting.

4. CNN Model

A Convolutional Neural Network automatically learns visual features from MRI images, including patterns, textures, and shapes.

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

The project also demonstrates prediction on unseen MRI images.

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

Example prediction logic:

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

🌐 Streamlit Web Application

The project includes an interactive Streamlit application implemented in app.py.

Features

Upload a brain MRI image.

View the uploaded image.

Automatically preprocess the image.

Generate a prediction using the trained CNN.

Display the predicted tumor category.

Display the prediction confidence.

Run the Application

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run Streamlit:

streamlit run app.py

The application will normally open at:

http://localhost:8501

📦 Installation

1. Clone the Repository

git clone https://github.com/devanshu2003-tech/medical-image-classification-2.git
cd medical-image-classification-2

2. Create a Virtual Environment

python -m venv venv

3. Activate the Virtual Environment

Windows PowerShell:

venv\Scripts\activate

4. Install Dependencies

pip install -r requirements.txt

5. Run the Application

streamlit run app.py

📓 Jupyter Notebook

The complete Deep Learning workflow is available in:

notebooks/Brain_Tumor_Classification.ipynb

The notebook covers:

Dataset loading

Image preprocessing

Data augmentation

CNN model development

Model training

Model evaluation

Prediction

Model saving

Launch Jupyter Notebook with:

jupyter notebook

💾 Trained Models

The repository contains trained Keras model files:

models/
├── brain_tumor_model.keras
└── final_brain_tumor_model.keras

The final model can be loaded using:

from tensorflow.keras.models import load_model

model = load_model("models/final_brain_tumor_model.keras")

📊 Sample Prediction Results

The project report presents sample predictions on unseen MRI images.

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

These values represent sample prediction confidence documented in the project report. They should not be interpreted as clinical accuracy or real-world diagnostic performance.

📁 Sample Image

A sample test image is included in:

sample_images/Test.jpg

It can be used to test the Streamlit application's image-upload and prediction workflow.

🚀 Future Scope

Possible future improvements include:

Training on a larger and more diverse MRI dataset.

Using advanced architectures such as ResNet, DenseNet, or EfficientNet.

Implementing tumor segmentation to identify tumor location.

Adding Explainable AI (XAI), such as Grad-CAM.

Hyperparameter tuning.

Transfer learning.

Deploying the application to cloud platforms.

Developing a mobile application.

Integrating with hospital information systems and PACS.

⚠️ Medical Disclaimer

This application is an educational Deep Learning project and is not a medical diagnostic tool.

Predictions generated by the model should not be used as a substitute for examination, diagnosis, treatment, or advice from a qualified medical professional.

Always consult a qualified healthcare professional for medical decisions.

👨‍💻 Author

Devanshu Wankhade

Project: Brain Tumor MRI Classification Using Deep Learning

Technologies: Python | TensorFlow | Keras | CNN | Streamlit

📚 References

Dataset

Brain Tumor MRI Dataset — Kaggle
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Books

Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016.

François Chollet. Deep Learning with Python, 2nd Edition.

Aurélien Géron. Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition.

Documentation

TensorFlow Documentation:
https://www.tensorflow.org/

Streamlit Documentation:
https://docs.streamlit.io/

⭐ Project Repository

GitHub Repository:

https://github.com/devanshu2003-tech/medical-image-classification-2