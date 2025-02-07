# **Mental Health Prediction & AI Therapist Chatbot: Empowering Students and Professionals**

This project introduces a **Streamlit-based web application** designed to proactively address mental health concerns among **students and professionals**. By leveraging the power of Artificial Intelligence, this application offers two core functionalities: **mental health condition prediction** and an **AI therapist chatbot** for immediate support. Integrating advanced machine learning models like **TabNet**, **XGBoost**, and a **fine-tuned T5 transformer model**, the project aims to provide accurate predictions and empathetic, context-aware therapeutic interactions.

---

## **📌 Project Overview: A Dual Approach to Mental Well-being**

This innovative project is structured around two key objectives:

- **Predictive Mental Health Assessment:**  Employing machine learning algorithms, the application predicts potential mental health risks. It utilizes **XGBoost** specifically for student profiles and **TabNet** for professional profiles, tailoring the prediction models to the unique challenges faced by each group. Users input relevant personal and lifestyle information to receive a personalized mental health status assessment.

- **AI-Powered Therapy Chatbot:**  For immediate and accessible mental health support, the application features an **AI therapist chatbot**. This chatbot is powered by a **fine-tuned T5 transformer model**, enabling it to understand and respond to user queries with empathy and contextual awareness, simulating a supportive therapeutic conversation.

- **Empowering Self-Analysis:** The application empowers users to proactively assess their mental well-being. By inputting their symptoms and lifestyle factors, users gain insights into their potential mental health condition and can access immediate support through the AI therapist.

**Core Technologies:**

- **Machine Learning Models:**
    - **XGBoost:** Gradient Boosting for Student Mental Health Prediction
    - **TabNet:** Deep Learning for Professional Mental Health Prediction
- **Large Language Model (LLM):** **Fine-tuned T5 Transformer** for AI Therapist Chatbot
- **Key Libraries:** PyTorch, Hugging Face Transformers, scikit-learn, Streamlit
- **Data Preprocessing:** Customized pipelines for student and professional datasets (`pipeline_student.pkl`, `pipeline_professional.pkl`)

---

## **📂 Dataset Structure: Tailored Data for Targeted Prediction and Therapy**

The project utilizes three distinct datasets, each meticulously structured to address specific aspects of mental health prediction and AI therapy:

### **1️⃣ Student Mental Health Dataset (`student_mental.csv`): Factors Influencing Student Well-being**

This dataset focuses on factors relevant to student mental health, encompassing:

- **Demographics:** `Gender`, `Age`, `City`
- **Academic Environment:** `Academic Pressure` (Level of academic stress), `CGPA` (Cumulative Grade Point Average), `Study Satisfaction` (Satisfaction with their studies)
- **Lifestyle Habits:** `Sleep Duration` (Hours of sleep per night), `Dietary Habits` (Healthy, Moderate, Unhealthy), `Financial Stress` (Level of financial concern), `Work/Study Hours` (Hours spent on work or study per day)
- **Mental Health Background:** `Family History` (Family history of mental illness - Yes/No), `Suicidal Thoughts` (Ever had suicidal thoughts - Yes/No)
- **Target Variable:** `Depression` (Binary outcome: 0 = No Depression, 1 = Depression)

### **2️⃣ Professional Mental Health Dataset (`professional_mental.csv`): Understanding Workplace Mental Health**

This dataset shifts focus to the professional environment, capturing factors impacting mental health in the workplace:

- **Demographics:** `Gender`, `Age`, `City`, `Profession` (Occupation), `Degree` (Highest educational degree)
- **Workplace Dynamics:** `Work Pressure` (Level of work-related stress), `Job Satisfaction` (Satisfaction with current job), `Work Hours` (Hours worked per week), `Financial Stress` (Level of financial concern)
- **Lifestyle & History:** `Sleep Duration` (Hours of sleep per night), `Dietary Habits` (Healthy, Moderate, Unhealthy), `Family History` (Family history of mental illness - Yes/No), `Suicidal Thoughts` (Ever had suicidal thoughts - Yes/No)
- **Target Variable:** `Depression` (Binary outcome: 0 = No Depression, 1 = Depression)

### **3️⃣ AI Therapist LLM Dataset (`Context-Response Dataset.csv`):  Enabling Empathetic AI Conversations**

*Note: The actual filename is `Context-Response Dataset.csv`*

This dataset is specifically designed for fine-tuning the T5 transformer model to function as an empathetic AI therapist:

- **`Context` (User Queries):**  A collection of mental health-related questions, concerns, and statements representing user inputs to the chatbot.
- **`Response` (Therapist Responses):**  Professionally crafted, therapist-style responses designed to be empathetic, supportive, and contextually relevant to the user queries.

---

## **📊 Exploratory Data Analysis (EDA) & Preprocessing: Unveiling Insights and Preparing Data**

**Exploratory Data Analysis (EDA)** was crucial in understanding the datasets and identifying key factors influencing mental health. The `EDA_and_feature_engineering.ipynb` notebook details this process, which included:

- **Depression Distribution Analysis:** Examining the prevalence of depression within both student and professional datasets to understand the scope of the issue.
- **Feature Importance Analysis (SHAP Values):** Utilizing SHAP (SHapley Additive exPlanations) values to determine the most influential features contributing to depression prediction in each dataset. This helped identify key risk factors for students and professionals separately.
- **Data Imbalance Handling:** Addressing potential class imbalance in the depression target variable to ensure robust and unbiased model training.

**Data Preprocessing**, documented in `data_preprocessing.ipynb`, focused on preparing the datasets for machine learning models. Key steps included:

- **Missing Value Imputation:** Strategies to handle incomplete data entries, ensuring no data is lost and models can train effectively.
- **Categorical Encoding (One-Hot Encoding):** Converting categorical features (like Gender, City, Degree, Profession) into numerical representations suitable for machine learning algorithms.
- **Feature Scaling (Standardization):** Standardizing numerical features to have zero mean and unit variance. This is essential for algorithms sensitive to feature scaling, like TabNet and XGBoost, ensuring optimal performance.
- **Preprocessing Pipelines:** Creation of distinct pipelines for professionals (`pipeline_professional.pkl`) and students (`pipeline_student.pkl`). These pipelines encapsulate all preprocessing steps, ensuring consistent data transformation during model training and deployment.

---

## **🤖 Machine Learning Models: Prediction Engines for Mental Health Assessment**

The project employs distinct machine learning models optimized for each user group:

| Model        | Dataset       | Algorithm                         | Accuracy | Key Predictive Features                                       | Model File          |
|--------------|---------------|-----------------------------------|----------|---------------------------------------------------------------|----------------------|
| **XGBoost**  | Students      | Gradient Boosting Decision Trees   | 85%      | Academic Pressure, CGPA, Study Satisfaction, Financial Stress, Sleep Duration | `xgboost_model.pkl` |
| **TabNet**   | Professionals | Deep Learning-based TabNet Classifier | 96%      | Work Pressure, Job Satisfaction, Financial Stress, Sleep Duration | `network.zip`        |

**1️⃣ Student Model: XGBoost (`xgboost_model.pkl`)**

- **Algorithm:** XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting framework known for its high accuracy and efficiency. It combines multiple decision trees to create a robust predictive model.
- **Performance:** Achieved an accuracy of **85%** in predicting depression risk among students.
- **Key Features:** The model highlighted **Academic Pressure, CGPA, Study Satisfaction, Financial Stress, and Sleep Duration** as the most significant factors influencing student mental health, aligning with common stressors faced by students.

**2️⃣ Professional Model: TabNet (`network.zip`)**

- **Algorithm:** TabNet is a deep learning model specifically designed for tabular data. Its attention mechanism allows for interpretable feature selection and robust performance, even with complex datasets.
- **Performance:** Demonstrated a high accuracy of **96%** in predicting depression risk among professionals.
- **Key Features:**  **Work Pressure, Job Satisfaction, Financial Stress, and Sleep Duration** emerged as crucial predictors of professional mental health, reflecting the pressures and lifestyle factors impacting working individuals.

---

## **🧠 AI Therapist: Fine-Tuned T5 Transformer for Empathetic Support**

The AI Therapist chatbot is powered by a fine-tuned T5 transformer model, enabling it to engage in meaningful and supportive conversations.

### **1️⃣ Dataset and Tokenization**

- **Dataset:** The `Context-Response Dataset.csv` dataset, containing user queries (`Context`) and therapist-style responses (`Response`), was used for fine-tuning.
- **Preprocessing:**  Textual data from the dataset was tokenized using **Hugging Face AutoTokenizer**. Tokenization converts text into numerical tokens that the T5 model can understand and process.

### **2️⃣ Fine-Tuning Process (`T5_LLM_Experimentation.ipynb`)**

- **Base Model:** The fine-tuning process started with the pre-trained **`google/t5-base`** model from Hugging Face Transformers. `t5-base` provides a strong foundation for natural language understanding and generation.
- **Training Configuration:** The model was fine-tuned with the following configuration:
    - **Epochs:** 2 (Number of passes through the entire training dataset)
    - **Batch Size:** 24 (Number of samples processed in each training iteration)
    - **Learning Rate:** 1e-3 (Controls the step size during optimization)
    - **Weight Decay:** 0.01 (Regularization technique to prevent overfitting)

### **3️⃣ Checkpoint-Based Training and Performance**

- **Resuming Training:** Checkpoint-based training was implemented to ensure training progress could be saved and resumed, enhancing training stability and efficiency.
- **Best ROUGE-1 Score:** The fine-tuned T5 model achieved a **ROUGE-1 score of 0.41**. ROUGE-1 is a metric commonly used to evaluate the quality of text summarization and generation, with higher scores indicating better performance. A score of 0.41 suggests a reasonable level of text generation quality for the AI therapist.
- **Empathetic Response Generation:** The fine-tuned T5 model is capable of generating **empathetic and context-aware responses** to user queries. This allows the chatbot to provide more human-like and supportive interactions, crucial for a mental health support application.

---

## **🚀 Getting Started: Running the Mental Health Web Application**

To run this project, ensure you have the necessary libraries installed (PyTorch, Hugging Face Transformers, scikit-learn, Streamlit) and the model files (`xgboost_model.pkl`, `network.zip`, fine-tuned T5 model checkpoint) available in the correct project directories.  Refer to the project's `README.md` file for detailed setup and execution instructions.

## **🔍 Usage: Navigating the Application**

### **📌 Depression Risk Prediction: Assessing Your Mental Health Status**

1. **Select User Type:** On the application's interface, choose whether you are a **"Professional"** or **"Student"**. This selection determines which prediction model (TabNet or XGBoost) will be used.
2. **Input Personal & Lifestyle Information:**  Fill out the form with relevant details about your demographics, academic/workplace factors, lifestyle habits, and mental health history. The specific input fields will vary slightly depending on whether you selected "Professional" or "Student" to match the respective datasets.
3. **Receive Depression Risk Assessment:** After submitting your information, the application will process your inputs using the selected machine learning model and provide a **depression risk assessment**. This assessment will indicate the predicted likelihood of depression based on your provided data.

### **📌 AI Therapist Chatbot: Seeking Immediate Mental Health Support**

1. **Enter Mental Health-Related Queries:** In the chatbot interface, type your mental health-related questions, concerns, or feelings. You can ask questions about stress, anxiety, sadness, or any other mental health topics.
2. **Receive AI-Generated Empathetic Responses:** The AI Therapist chatbot, powered by the fine-tuned T5 model, will generate and display a response. The response will be designed to be empathetic, contextually relevant, and supportive, offering a virtual therapeutic interaction.

---

## **📊 Model Performance Summary**

| Model        | Target User Group | Dataset             | Accuracy/Metric | Key Performance Indicators                                   |
|--------------|-------------------|----------------------|-----------------|---------------------------------------------------------------|
| **TabNet**   | Professionals     | `professional_mental.csv` | **96% Accuracy** | High accuracy in predicting depression risk for professionals |
| **XGBoost**  | Students        | `student_mental.csv`    | **85% Accuracy** | Robust prediction of depression risk for students              |
| **T5 LLM**   | AI Therapist     | `Context-Response Dataset.csv` | **ROUGE-1: 0.41**| Empathetic and context-aware response generation for therapy |

This project provides a valuable tool for preliminary mental health assessment and support, leveraging AI to make mental health resources more accessible and proactive for students and professionals. Remember that this application is for informational and support purposes and does not replace professional medical advice or treatment. If you are experiencing a mental health crisis, please seek help from qualified mental health professionals.

---


## **🛠️ Project Structure**  

```bash
📦 Mental Health Prediction  
├── 📂 model_folder/                  # Trained ML models  
│   ├── network.zip                   # TabNet model for professionals  
│   ├── model_params.json               # TabNet model for professionals 
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

