---
title: Zomato-rating-predictor
emoji: 📊
colorFrom: red
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
license: mit
sdk_version: 1.52.2
---

# Zomato Rating Predictor 🍽️📊

This repository contains the code, application, and research paper for the conference project: **Exploratory and Predictive Analytics of Global Restaurant Markets with Zomato Price–Rating Data Using Ensemble Regression Models**.

## 📌 Project Overview
This project focuses on predicting Zomato restaurant ratings using various ensemble regression models based on factors like average cost, location, and cuisines. It includes a complete end-to-end machine learning pipeline, from exploratory data analysis (EDA) to a deployed interactive web application.

## 🚀 Live Application
The interactive application is built with **Streamlit** and is deployed live on Hugging Face Spaces:
👉 [**Zomato Rating Predictor - Hugging Face Space**](https://huggingface.co/spaces/saymyname-07/zomato-rating-predictor)

## 📁 Repository Contents
* `app.py`: The main Streamlit web application.
* `Exploratory_and_predictive_analytics_of_global_restaurant_markets_using_zomato_price_and_rating_data.ipynb`: Detailed Jupyter Notebook containing data preprocessing, EDA, and model training.
* `Exploratory and Predictive Analytics of Global Restaurant Markets with Zomato Price–Rating Data Using Ensemble Regression Models.doc`: The formal research paper for the conference.
* `dist/Zomato_Predictor.exe`: A standalone executable version of the application for Windows. No Python installation required!
* `best_model.pkl` & `scaler.pkl`: The serialized pre-trained model and data scaler used by the application.

## 💻 Running Locally
You can run this project locally in two ways:

### 1. Using the Standalone Executable (Windows)
Simply download and run `dist/Zomato_Predictor.exe`. The application will start locally without needing any dependencies.

### 2. Using Python
1. Clone the repository:
   ```bash
   git clone https://github.com/Saro-07/zomato-rating-predictor.git
   cd zomato-rating-predictor
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Technologies Used
* **Machine Learning:** Scikit-Learn (Ensemble Regression Models)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Web App:** Streamlit
* **Deployment:** Hugging Face Spaces, PyInstaller

---
*Created for Conference Presentation*
