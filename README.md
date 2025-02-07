# **Mental Health Prediction & AI Therapist Chatbot**  

A **Streamlit-based AI-powered web application** that predicts mental health conditions for **students and professionals** while offering an **AI therapist chatbot** for mental health support. This project integrates **TabNet**, **XGBoost**, and a **fine-tuned T5 transformer model** for accurate prediction and empathetic responses.  

---

## **📌 Project Overview**  

This project is designed to:  

- **Predict Mental Health Status:** Uses machine learning models (XGBoost for students, TabNet for professionals) to assess mental health risks based on user inputs.  
- **Provide AI Therapy:** A fine-tuned T5 model acts as a virtual therapist, responding to user queries with context-aware and empathetic support.  
- **Enable Self-Analysis:** Users can input their symptoms to assess their mental health condition.  

**Technologies Used:**  
- **ML Models:** XGBoost (Students), TabNet (Professionals)  
- **LLM Model:** Fine-tuned T5 transformer for AI therapy  
- **Libraries:** PyTorch, Hugging Face Transformers, scikit-learn, Streamlit  
- **Preprocessing:** Pipelines for both user types  

---

## **📂 Dataset Structure**  

### **1️⃣ Student Mental Health Dataset (`student_mental.csv`)**  
- Demographics: `Gender`, `Age`, `City`  
- Academic Factors: `Academic Pressure`, `CGPA`, `Study Satisfaction`  
- Lifestyle: `Sleep Duration`, `Dietary Habits`, `Financial Stress`, `Work/Study Hours`  
- Mental Health History: `Family History`, `Suicidal Thoughts`  
- Target Variable: `Depression (0 = No, 1 = Yes)`  

### **2️⃣ Professional Mental Health Dataset (`professional_mental.csv`)**  
- Demographics: `Gender`, `Age`, `City`, `Profession`, `Degree`  
- Workplace Factors: `Work Pressure`, `Job Satisfaction`, `Work Hours`, `Financial Stress`  
- Lifestyle: `Sleep Duration`, `Dietary Habits`, `Family History`, `Suicidal Thoughts`  
- Target Variable: `Depression (0 = No, 1 = Yes)`  

### **3️⃣ Fine-Tuned LLM Dataset (`X.csv`)**  
- `User Queries`: Mental health-related questions from users  
- `Therapist Responses`: Professional therapist-style responses  
- Used to fine-tune `T5` for **empathetic response generation**  

---

## **📊 Exploratory Data Analysis (EDA) & Preprocessing**  

EDA was conducted to identify key patterns and correlations in mental health conditions.  

- **`EDA_and_feature_engineering.ipynb`** contains:  
  - Distribution of depression cases  
  - Feature importance analysis using SHAP values  
  - Data imbalance handling strategies  

- **Preprocessing Steps (`data_preprocessing.ipynb`)**:  
  - **Handling Missing Values**: Imputation for incomplete records  
  - **Categorical Encoding**: One-hot encoding for categorical variables  
  - **Feature Scaling**: Standardization for numerical features  
  - **Pipelines**:  
    - `pipeline_professional.pkl` for professionals  
    - `pipeline_student.pkl` for students  

---

## **🛠️ Project Structure**  

```bash
📦 Mental Health Prediction  
├── 📂 model_folder/                  # Trained ML models  
│   ├── network.zip                   # TabNet model for professionals  
│   ├── xgboost_model.pkl              # XGBoost model for students  
│   ├── pipeline_professional.pkl      # Preprocessing pipeline (professionals)  
│   ├── pipeline_student.pkl           # Preprocessing pipeline (students)  
├── 📂 fine_tuned_t5/                  # AI Therapist LLM model  
├── 📜 EDA_and_feature_engineering.ipynb # Exploratory Data Analysis  
├── 📜 LLM Experimentation Report.pdf   # LLM training insights  
├── 📜 T5_LLM_Experimentation.ipynb     # Fine-tuning experiments  
├── 📜 data_preprocessing.ipynb         # Data cleaning & transformation  
├── 📜 predict_mental_health.ipynb      # Model inference for mental health prediction  
├── 📜 mental_health_ui.py              # Streamlit web application  
├── 📜 pipeline.py                      # Preprocessing pipeline implementation  
├── 📜 X.csv                            # Dataset for fine-tuning LLM  
├── 📜 student_mental.csv               # Student dataset  
├── 📜 professional_mental.csv          # Professional dataset  
└── 📜 requirements.txt                 # Dependencies  
