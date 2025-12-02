# AI Fitness Trainer – Smart Diet Type Recommendation System

## Overview
The **AI Fitness Trainer** is a machine learning–powered system that predicts the ideal diet type for users based on their physical and lifestyle characteristics.  
The system classifies users into one of the following diet categories:

- Balanced Diet  
- High-Protein Diet  
- Low-Carb Diet  

It helps users receive fast, accurate, and personalized nutritional advice without the need for a human dietitian.

---
---

## Motivation
Obesity rates are increasing worldwide, and people now seek quick and personalized nutritional recommendations.  
Our system provides:

- Automated diet suggestions based on user data  
- Affordable and scalable alternative to personal dietitians  
- Integration capabilities with mobile apps, gyms, and online fitness platforms  
- Science-based and data-driven recommendations  

---

## Dataset Overview
The dataset includes the following features:

- Age  
- Gender (0 = Female, 1 = Male)  
- Height (m)  
- Weight (kg)  
- BMI  
- Body Fat Percentage  
- Workout Frequency  
- Fitness Goal (0 = Maintain, 1 = Muscle Gain, 2 = Weight Loss)  
- **Diet_Type (Target): Balanced / High-Protein / Low-Carb**

Dataset Link:  
https://github.com/MOHAMED00974/Diet-dataset/blob/main/Diet_Type_Dataset_bad.csv

---

## Preprocessing
Several preprocessing steps were applied to improve data quality and model performance:

- Duplicate removal  
- Handling missing values  
- Label correction  
- Categorical feature encoding  
- Feature scaling  
- Handling class imbalance using SMOTE  
- Outlier checks and normalization  

These steps improved the stability and accuracy of the ML model.

---

## Model Building
Multiple classification models were trained, including:

- Support Vector Machine (SVM)  
- Random Forest  
- Logistic Regression  
- Decision Tree Classifier  

### Best Model: SVM  
**Final Accuracy: 85.7%**

Evaluation techniques:

- Confusion Matrix  
- Classification Report  
- Accuracy Score  
- Cross-Validation  

Model Training Notebook:  
https://colab.research.google.com/drive/1BqR129o_9U7DvkfrGzNUKl1ZKCpseNM3?usp=sharing

---

## UI / UX
The interface was designed to be simple, intuitive, and user-friendly.

Key features:

- Input fields for all user data  
- Instant prediction display  
- Nutrition dashboard showing calories and macronutrient distribution  
- Option to download a full PDF diet report  

The goal is to allow users with no technical background to easily access personalized diet recommendations.

---

## Features
- Accurate ML-based diet prediction  
- Clean and interactive user interface  
- Flask-based backend  
- PDF report generation  
- Scalable and easy to integrate into apps/platforms  
- Reliable science-based recommendations  

---

## Advantages
- Affordable alternative to human dietitians  
- Eliminates biases and improves decision-making  
- Learns and improves as more data is added  
- Lightweight and fast  
- Ideal for fitness centers, apps, and wellness platforms  

---

## Future Improvements
- Add more diet types (Keto, Mediterranean, Vegan, etc.)
- Build a full mobile app  
- Integration with smartwatches and fitness trackers  
- Daily meal plan generator  
- Food logging + calorie tracking  
- User progress tracking dashboard  

---

## Team Members
- Khaled Youssef Mohamed  
- Mahmoud Mohamed Kaoud  
- Abdulrahman Ashraf Albelasi  
- Mohamed Tamer Mohamed  

Supervisor: **Assoc. Prof. Mohamed Abd Elfattah**

---

## Acknowledgments
Special thanks to the AnalytIQ Team for their efforts in research, development, and design.
