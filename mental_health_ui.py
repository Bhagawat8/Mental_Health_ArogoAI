import streamlit as st
import torch
import pickle
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import json
from custom_encoder import CustomEncoder , StudentEncoder

MODEL_ZIP = "model_folder/network.zip"
PIPELINE_PATH = "C:/Users/dell/Documents/Data science/campusX/arogo assignment/submission/pipeline_professional.pkl"

with open("model_folder/model_params.json", "r") as f:
    model_params = json.load(f)

for key in ["init_params", "class_attrs"]:
    model_params.pop(key, None)

model = TabNetClassifier(**model_params)

model.load_model(MODEL_ZIP)

with open(PIPELINE_PATH, "rb") as f:
    pipeline = pickle.load(f)

def predict_mental_health(features, user_type):
    print(features)
    input_data = pipeline.transform([features])

    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        output = model.predict_proba(input_tensor)
    print(output)

    predicted_class = np.argmax(output, axis=1)[0]
    print(predicted_class)
    if predicted_class == 0:
      return f"Mental Health status: You are not in Depression"
    if predicted_class == 1:
        return f"Mental Health status: You are in Depression"


st.title("Mental Health Predictor")



city_options = [
    "Kalyan", "Patna", "Vasai-virar", "Kolkata", "Ahmedabad", "Meerut",
    "Ludhiana", "Pune", "Rajkot", "Visakhapatnam", "Srinagar", "Mumbai",
    "Indore", "Agra", "Surat", "Varanasi", "Vadodara", "Hyderabad",
    "Kanpur", "Jaipur", "Thane", "Lucknow", "Nagpur", "Bangalore",
    "Chennai", "Ghaziabad", "Delhi", "Bhopal", "Faridabad", "Nashik"
]

profession_options = sorted([
    "Student", "Teacher", "Unknown", "Content Writer", "Architect", "Consultant", "HR Manager", 
    "Pharmacist", "Doctor", "Business Analyst", "Entrepreneur", "Chemist", "Chef", 
    "Educational Consultant", "Data Scientist", "Researcher", "Lawyer", "Customer Support", 
    "Marketing Manager", "Pilot", "Travel Consultant", "Plumber", "Sales Executive", "Manager", 
    "Judge", "Electrician", "Financial Analyst", "Software Engineer", "Civil Engineer", 
    "UX/UI Designer", "Digital Marketer", "Accountant", "Financial Analyst", "Mechanical Engineer", 
    "Graphic Designer", "Research Analyst", "Investment Banker", "Unemployed", "Family Consultant", 
    "Analyst"
])

degree_options = sorted([
    "Class 12", "B.Ed", "B.Arch", "B.Com", "B.Pharm", "BCA", "M.Ed", "MCA", "BBA", "BSc", 
    "MSc", "LLM", "M.Pharm", "M.Tech", "B.Tech", "LLB", "BHM", "MBA", "BA", "ME", "MD", "MHM", 
    "PhD", "BE", "M.Com", "MBBS", "MA", "M.Arch"
])

user_type = st.radio("Please select your user type:", ("Professional", "Student"))

if user_type == "Professional":
    st.header("Professional User Input")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    city = st.selectbox("City", city_options)
    profession = st.selectbox("Profession", profession_options)
    work_pressure = st.slider("Work Pressure (1-5)", min_value=0, max_value=5)
    job_satisfaction = st.slider("Job Satisfaction (1-5)", min_value=0, max_value=5)
    sleep_duration = st.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '6-8 hours', 'More than 8 hours'])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
    degree = st.selectbox("Degree", degree_options)
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    work_study_hours = st.number_input("Work Hours", min_value=0, max_value=24, value=8)
    financial_stress = st.slider("Financial Stress (1-5)", min_value=0, max_value=5, value=1)
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    if st.button("Predict Mental Health"):

        features = {
            "Gender": gender,
            "Age": age,
            "City": city,
            "Profession": profession,
            "Work Pressure": work_pressure,
            "Job Satisfaction": job_satisfaction,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": dietary_habits,
            "Degree": degree,
            "Suicidal Thoughts": suicidal_thoughts,
            "Work/Study Hours": work_study_hours,
            "Financial Stress": financial_stress,
            "Family History": family_history
        }

        prediction = predict_mental_health(features, user_type)
        st.success(f"Prediction: {prediction}")


elif user_type == "Student":
    st.header("Student User Input")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=18)
    city = st.selectbox("City", city_options)
    academic_pressure = st.slider("Academic Pressure (1-5)", min_value=1, max_value=5, value=1)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    study_satisfaction = st.slider("Study Satisfaction (1-5)", min_value=1, max_value=5, value=1)
    sleep_duration = st.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '6-8 hours', 'More than 8 hours'])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
    degree = st.selectbox("Degree", degree_options)
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    work_study_hours = st.number_input("Study Hours", min_value=0, max_value=24, value=8)
    financial_stress = st.slider("Financial Stress (1-5)", min_value=1, max_value=5, value=1)
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    if st.button("Predict Mental Health"):

        features = {
            "Gender": gender,
            "Age": age,
            "City": city,
            "Academic Pressure": academic_pressure,
            "CGPA": cgpa,
            "Study Satisfaction": study_satisfaction,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": dietary_habits,
            "Degree": degree,
            "Suicidal Thoughts": suicidal_thoughts,
            "Work/Study Hours": work_study_hours,
            "Financial Stress": financial_stress,
            "Family History": family_history
        }

        with open('pipeline_student.pkl', 'rb') as f:
            student_pipeline = pickle.load(f)
        input_data = student_pipeline.transform([features])
        print(input_data)

        with open('xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        prediction = xgb_model.predict(input_data)
        
        if prediction[0] == 0:
            prediction_text = "Mental Health status: You are not in Depression"
        else:
            prediction_text = "Mental Health status: You are in Depression"
            
        st.success(f"Prediction: {prediction_text}")


import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "C:/Users/dell/Documents/Data science/campusX/arogo assignment/submission/fine_tuned_t5_epoch_1"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_ai_response(message: str) -> str:
    """
    Generate a response from the finetuned T5 model given a user message.
    """

    input_text = f"Question: {message}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=75, num_return_sequences=1, do_sample=False)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.header("Chat with AI Therapist")
user_message = st.text_area("Enter your message for the AI Therapist:")

if st.button("Chat"):
    ai_response = get_ai_response(user_message)
    st.write(ai_response)




