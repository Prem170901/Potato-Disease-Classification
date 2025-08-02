# 🥔 **Potato Disease Classification**

## 📖 **About**

This project focuses on classifying potato plant diseases using deep learning. It involves training a custom image classification model on a dataset of potato plant leaves, and deploying it across a web interface and mobile app for real-time inference. The project showcases full-stack ML deployment and cloud hosting.

## 🔍 **Project Summary**

- **Developed** a deep learning model using *TensorFlow/Keras* to classify potato diseases from leaf images.  
- **Preprocessed** and trained the model on a curated dataset obtained from *Kaggle*, keeping only the relevant potato-related classes.  
- **Converted** the trained model into *TensorFlow Lite* format to enable efficient deployment on mobile devices.  
- **Built and tested** a *FastAPI*-based backend to serve predictions from both the standard and TF Lite models.  
- **Developed** a *ReactJS* web interface and a *React Native* mobile app for user interaction and disease prediction.  
- **Integrated** *TensorFlow Serving* with *Docker* and *FastAPI* for scalable model serving.  
- **Deployed** the trained models as *Google Cloud Functions* using both standard TensorFlow and TF Lite versions.  
- **Ensured** end-to-end pipeline from training, deployment, to user-facing interfaces.

## 🛠️ **Tech Stack**

- **Modeling & Training:** Python, TensorFlow, Jupyter Notebook  
- **API & Backend:** FastAPI, TensorFlow Serving, Docker  
- **Frontend:** ReactJS  
- **Mobile App:** React Native  
- **Cloud Deployment:** Google Cloud Platform (GCF, GCS)
