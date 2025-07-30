from flask import Flask, render_template, request
import pandas as pd
import pickle

job_titles = [
    "Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate",
    "Director", "Product Manager", "Marketing Analyst", "Financial Analyst",
    "HR Manager", "Data Scientist", "Project Manager", "Accountant",
    "Software Developer", "Business Analyst", "UX Designer"
]

# Initialize the Flask app
app = Flask(__name__)

# Sample mapping function
education_mapping = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}

# Custom function to map Education Level
def map_education(X):
    return X.replace(education_mapping)

# Load the preprocessor and model
with open('Models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('Models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route
@app.route('/')
def home():
    return render_template('predict.html', job_titles=job_titles)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    gender = request.form['gender']
    education = request.form['education']
    job_title = request.form['job']
    experience = float(request.form['experience'])

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': experience
    }])

    transformed_input = preprocessor.transform(input_data)
    prediction = model.predict(transformed_input)[0]

    return render_template('predict.html',
                            prediction_text=f"Predicted Salary: ₹{round(prediction, 2)}",
                            age=age,
                            gender=gender,
                            education=education,
                            job=job_title,
                            experience=experience,
                            job_titles=job_titles)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
