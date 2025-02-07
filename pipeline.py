import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

degree_mapping = {
    "Class 12": 1, "B.Ed": 2, "B.Arch": 3, "B.Com": 4, "B.Pharm": 5, "BCA": 6, 
    "M.Ed": 7, "MCA": 8, "BBA": 9, "BSc": 10, "MSc": 11, "LLM": 12, 
    "M.Pharm": 13, "M.Tech": 14, "B.Tech": 15, "LLB": 16, "BHM": 17, 
    "MBA": 18, "BA": 19, "ME": 20, "MD": 21, "MHM": 22, "PhD": 23, 
    "BE": 24, "M.Com": 25, "MBBS": 26, "MA": 27, "M.Arch": 28
}

profession_mapping = {
    'Student': 1, 'Teacher': 2, 'Unknown': 3, 'Content Writer': 4, 'Architect': 5, 
    'Consultant': 6, 'HR Manager': 7, 'Pharmacist': 8, 'Doctor': 9, 
    'Business Analyst': 10, 'Entrepreneur': 11, 'Chemist': 12, 'Chef': 13, 
    'Educational Consultant': 14, 'Data Scientist': 15, 'Researcher': 16, 'Lawyer': 17, 
    'Customer Support': 18, 'Marketing Manager': 19, 'Pilot': 20, 
    'Travel Consultant': 21, 'Plumber': 22, 'Sales Executive': 23, 'Manager': 24, 
    'Judge': 25, 'Electrician': 26, 'Financial Analyst': 27, 'Software Engineer': 28, 
    'Civil Engineer': 29, 'UX/UI Designer': 30, 'Digital Marketer': 31, 'Accountant': 32, 
    'Financial Analyst': 33, 'Mechanical Engineer': 34, 'Graphic Designer': 35, 
    'Research Analyst': 36, 'Investment Banker': 37, 'Unemployed': 38, 
    'Family Consultant': 39, 'Analyst': 40
}

city_mapping = {
    'Kalyan': 1, 'Patna': 2, 'Vasai-virar': 3, 'Kolkata': 4, 'Ahmedabad': 5,
    'Meerut': 6, 'Ludhiana': 7, 'Pune': 8, 'Rajkot': 9, 'Visakhapatnam': 10,
    'Srinagar': 11, 'Mumbai': 12, 'Indore': 13, 'Agra': 14, 'Surat': 15,
    'Varanasi': 16, 'Vadodara': 17, 'Hyderabad': 18, 'Kanpur': 19, 'Jaipur': 20,
    'Thane': 21, 'Lucknow': 22, 'Nagpur': 23, 'Bangalore': 24, 'Chennai': 25,
    'Ghaziabad': 26, 'Delhi': 27, 'Bhopal': 28, 'Faridabad': 29, 'Nashik': 30
}

sleep_duration_mapping = {
    'More than 8 hours': 4, 'Less than 5 hours': 1, '5-6 hours': 2, '6-8 hours': 3
}

dietary_habits_mapping = {
    'Healthy': 3, 'Unhealthy': 1, 'Moderate': 2
}

yes_no_mapping = {
    'Yes': 1, 'No': 0
}

gender_mapping = {
    'Male': 1, 'Female': 0
}

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoded_data = []
        for row in X:
            encoded_row = [
                gender_mapping[row['Gender']],
                row['Age'],  
                city_mapping[row['City']],
                profession_mapping[row['Profession']],
                row['Work Pressure'],  
                row['Job Satisfaction'], 
                sleep_duration_mapping[row['Sleep Duration']],
                dietary_habits_mapping[row['Dietary Habits']],
                degree_mapping[row['Degree']],
                yes_no_mapping[row['Suicidal Thoughts']],
                row['Work/Study Hours'],  
                row['Financial Stress'], 
                yes_no_mapping[row['Family History']]
            ]
            encoded_data.append(encoded_row)
        return np.array(encoded_data)
    
class StudentEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        encoded_data = []
        for row in X:
            encoded_row = [
                gender_mapping[row['Gender']],
                row['Age'],
                city_mapping[row['City']],
                row['Academic Pressure'],     
                row['CGPA'],                  
                row['Study Satisfaction'],     
                sleep_duration_mapping[row['Sleep Duration']],
                dietary_habits_mapping[row['Dietary Habits']],
                degree_mapping[row['Degree']],
                yes_no_mapping[row['Suicidal Thoughts']],
                row['Work/Study Hours'],
                row['Financial Stress'],
                yes_no_mapping[row['Family History']]
            ]
            encoded_data.append(encoded_row)
        return np.array(encoded_data)

def create_pipeline(X_train):

    encoder = CustomEncoder()
    scaler = StandardScaler()
       
    scaler.fit(X_train)
    
    pipeline = Pipeline([
        ('encoder', encoder),
        ('scaler', scaler)
    ])
    return pipeline

def create_student_pipeline():

    encoder = StudentEncoder()

    pipeline = Pipeline([
        ('encoder', encoder),
       
    ])
    return pipeline


if __name__ == '__main__':
    
    X_train = pd.read_csv("X.csv")
    if "Unnamed: 0" in X_train.columns:
      X_train.drop(["Unnamed: 0"], axis=1, inplace=True)
    
    pipeline_professional = create_pipeline(X_train)

    with open('pipeline_professional.pkl', 'wb') as f:
        pickle.dump(pipeline_professional, f)
    
    print("Pipeline with fitted StandardScaler saved successfully.")


    student_pipeline = create_student_pipeline()
    with open('pipeline_student.pkl', 'wb') as f:
        pickle.dump(student_pipeline, f)
    print("Student pipeline saved successfully.")