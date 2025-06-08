
from flask import Flask, render_template, request, jsonify
import csv
import os

app = Flask(__name__)

# Define serious conditions that require immediate medical attention
SERIOUS_CONDITIONS = {
    'Heart Attack', 'Stroke', 'Pulmonary Embolism', 'Meningitis', 'Appendicitis',
    'Pancreatitis', 'Sepsis', 'Anaphylaxis', 'Deep Vein Thrombosis', 'Tetanus',
    'Rabies', 'Ebola', 'Pneumonia', 'Tuberculosis', 'Leukemia', 'Lymphoma',
    'Lung Cancer', 'Breast Cancer', 'Prostate Cancer', 'Colon Cancer', 'Skin Cancer',
    'Thyroid Cancer', 'Pancreatic Cancer', 'Ovarian Cancer', 'Cervical Cancer',
    'Kidney Disease', 'Liver Disease', 'Cirrhosis', 'Congestive Heart Failure',
    'Coronary Artery Disease', 'HIV/AIDS', 'Malaria', 'Dengue Fever'
}

# Load disease-symptom data from comprehensive CSV file
def load_disease_data():
    disease_data = {}
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'comprehensive_disease_data.csv')
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                disease = row['Disease']
                symptoms = [symptom.strip().lower() for symptom in row['Symptoms'].split(',')]
                disease_data[disease] = symptoms
    except FileNotFoundError:
        print("Comprehensive disease data file not found. Using fallback data.")
        # Fallback to enhanced mapping if CSV is not found
        disease_data = {
            "Flu": ["fever", "chills", "muscle aches", "fatigue", "cough"],
            "Migraine": ["headache", "nausea", "dizziness", "light sensitivity"],
            "COVID-19": ["fever", "cough", "shortness of breath", "loss of taste or smell"],
            "Heart Attack": ["chest pain", "shortness of breath", "dizziness", "cold sweat"],
            "Diabetes": ["weight loss", "excessive thirst", "blurred vision", "fatigue"],
            "Asthma": ["shortness of breath", "wheezing", "chest tightness", "coughing"]
        }
    
    return disease_data

# Load the disease data when the app starts
DISEASE_DATA = load_disease_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_symptoms = data.get('symptoms', '').lower()
    
    if not user_symptoms.strip():
        return jsonify({"possible_conditions": ["Please enter your symptoms."]})
    
    # Split user input into individual symptoms
    user_symptom_list = [symptom.strip() for symptom in user_symptoms.replace(',', ' ').split()]
    
    # Find matching diseases
    matching_diseases = []
    
    for disease, disease_symptoms in DISEASE_DATA.items():
        match_count = 0
        total_symptoms = len(disease_symptoms)
        
        # Count how many user symptoms match this disease's symptoms
        for user_symptom in user_symptom_list:
            for disease_symptom in disease_symptoms:
                if user_symptom in disease_symptom or disease_symptom in user_symptom:
                    match_count += 1
                    break
        
        # Calculate match percentage
        if match_count > 0:
            match_percentage = (match_count / total_symptoms) * 100
            matching_diseases.append({
                'disease': disease,
                'match_count': match_count,
                'match_percentage': match_percentage,
                'total_symptoms': total_symptoms
            })
    
    # Sort by match count (descending) and then by match percentage
    matching_diseases.sort(key=lambda x: (x['match_count'], x['match_percentage']), reverse=True)
    
    # Format results in simple, easy-to-understand language
    if matching_diseases:
        results = []
        for match in matching_diseases[:5]:  # Top 5 matches
            # Simple confidence levels
            if match['match_percentage'] >= 50:
                confidence_text = "High"
                simple_explanation = "Your symptoms strongly suggest this condition"
            elif match['match_percentage'] >= 25:
                confidence_text = "Medium"
                simple_explanation = "Your symptoms somewhat match this condition"
            else:
                confidence_text = "Low"
                simple_explanation = "Your symptoms partially match this condition"
            
            is_serious = match['disease'] in SERIOUS_CONDITIONS
            
            result = {
                "condition": match['disease'],
                "confidence": confidence_text,
                "explanation": simple_explanation,
                "serious": is_serious
            }
            results.append(result)
        
        response = {"possible_conditions": results}
    else:
        response = {"possible_conditions": []}
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
