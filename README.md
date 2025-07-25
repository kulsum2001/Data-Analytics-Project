# Data-Analytics-Project

# 📝 Classify Online Text Reviews

A **Flask-based Machine Learning Web App** that classifies **Amazon Alexa customer reviews** into **Positive, Negative, or Neutral sentiments**. The app uses **data preprocessing**, **feature extraction (TF-IDF)**, and **various ML models including an Artificial Neural Network (ANN)** to deliver sentiment insights.

---

## ✅ Overview
The goal of this project is to analyze customer feedback and automatically determine the sentiment expressed in the reviews. Understanding sentiment helps businesses make data-driven decisions and enhance their products or services.

---

## 🔍 Key Features
- **Text Preprocessing**: Lowercasing, stopword removal, tokenization.
- **Feature Extraction**: TF-IDF for converting text into numerical features.
- **Modeling**: Logistic Regression, Random Forest, and ANN.
- **Flask Web App**: Simple UI to input text and view predicted sentiment.
- **Visualization**: Plots for EDA and model performance.

---

## 🛠️ Tech Stack
- **Frontend**: HTML, CSS
- **Backend**: Flask (Python)
- **ML Libraries**:
  - `scikit-learn`
  - `pandas`, `numpy`
  - `tensorflow` / `keras` (for ANN)
- **Visualization**:
  - `matplotlib`, `seaborn`

---

## 📂 Project Structure
Classify-Online-Text-Reviews/
│
├── data/ # Dataset (CSV)
├── notebooks/ # Jupyter Notebooks for EDA & modeling
├── models/ # Saved ML models
├── requirements.txt # Dependencies

## Dataset
The dataset used in this project can be downloaded from:
[Amazon Alexa Reviews Dataset (Kaggle)](https://www.kaggle.com/)

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Classify-Online-Text-Reviews.git
   cd Classify-Online-Text-Reviews
2. Create a Virtual Environment:
                                  python -m venv venv
                                  source venv/bin/activate  # For Linux/Mac
                                  venv\Scripts\activate     # For Windows
                                  
3. Install Dependencies:   pip install -r requirements.txt
4. Run the Flask App:  python app.py
                       Access the app at [http](http://127.0.0.1:5000/)

📌 Future Enhancements
Deploy the app on Heroku or Render.
Add real-time sentiment prediction for live reviews.
Implement deep learning models like LSTM for better performance.

⭐ **If you like this project, dont forget to star the repo!**


Author:
Sakina Kulsum
📧 sakinakulsum82@gmail.com
🔗 https://github.com/kulsum2001


