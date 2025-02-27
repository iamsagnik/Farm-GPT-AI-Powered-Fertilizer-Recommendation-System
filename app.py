import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

app = Flask(__name__)

# Load artifacts
model = tf.keras.models.load_model('model/my_model.keras')
scaler = joblib.load('model/scaler.pkl')
soil_encoder = joblib.load('model/soil_encoder.pkl')
crop_encoder = joblib.load('model/crop_encoder.pkl')
fertilizer_encoder = joblib.load('model/fertilizer_encoder.pkl')

print("Soil Types:", soil_encoder.classes_)
print("Crop Types:", crop_encoder.classes_)

# Crop requirements data
crop_requirements = {
    'rice': {'Nitrogen': (80, 120), 'Phosphorus': (40, 60), 'Potassium': (60, 80)},
    'wheat': {'Nitrogen': (50, 90), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'sugarcane': {'Nitrogen': (100, 150), 'Phosphorus': (50, 70), 'Potassium': (70, 100)},
    'groundnut': {'Nitrogen': (20, 50), 'Phosphorus': (40, 60), 'Potassium': (40, 50)},
    'cotton': {'Nitrogen': (60, 100), 'Phosphorus': (40, 60), 'Potassium': (50, 70)},
    'maize': {'Nitrogen': (50, 80), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'tur': {'Nitrogen': (40, 70), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'ginger': {'Nitrogen': (70, 120), 'Phosphorus': (40, 60), 'Potassium': (60, 80)},
    'grapes': {'Nitrogen': (60, 90), 'Phosphorus': (40, 50), 'Potassium': (70, 90)},
    'moong': {'Nitrogen': (30, 50), 'Phosphorus': (40, 60), 'Potassium': (40, 50)},
    'urad': {'Nitrogen': (30, 50), 'Phosphorus': (40, 60), 'Potassium': (40, 50)},
    'gram': {'Nitrogen': (20, 40), 'Phosphorus': (30, 50), 'Potassium': (30, 50)},
    'turmeric': {'Nitrogen': (50, 90), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'soybean': {'Nitrogen': (40, 70), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'masoor': {'Nitrogen': (20, 40), 'Phosphorus': (30, 50), 'Potassium': (30, 50)},
    'millets': {'Nitrogen': (30, 60), 'Phosphorus': (30, 50), 'Potassium': (30, 50)},
    'pulses': {'Nitrogen': (20, 50), 'Phosphorus': (30, 50), 'Potassium': (30, 50)},
    'pomegranate': {'Nitrogen': (40, 80), 'Phosphorus': (30, 50), 'Potassium': (50, 80)},
    'barley': {'Nitrogen': (50, 90), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'oil seeds': {'Nitrogen': (40, 60), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'tobacco': {'Nitrogen': (60, 100), 'Phosphorus': (40, 60), 'Potassium': (50, 70)},
    'orange': {'Nitrogen': (60, 90), 'Phosphorus': (30, 50), 'Potassium': (50, 80)},
    'watermelon': {'Nitrogen': (50, 80), 'Phosphorus': (30, 50), 'Potassium': (50, 80)},
    'kidneybeans': {'Nitrogen': (40, 70), 'Phosphorus': (40, 60), 'Potassium': (40, 60)},
    'coffee': {'Nitrogen': (70, 120), 'Phosphorus': (40, 60), 'Potassium': (50, 80)},
}

def crop_specific_recommendation(raw_input):
    recommendations = []
    crop = raw_input[5].lower()

    if crop in crop_requirements:
        ideal = crop_requirements[crop]
        n = raw_input[2]
        p = raw_input[3]
        k = raw_input[4]

        # Nitrogen check
        if n < ideal['Nitrogen'][0]:
            recommendations.append(f"Nitrogen is low for {crop} (current: {n}, ideal: {ideal['Nitrogen'][0]}-{ideal['Nitrogen'][1]})")
        elif n > ideal['Nitrogen'][1]:
            recommendations.append(f"Nitrogen is high for {crop} (current: {n}, ideal: {ideal['Nitrogen'][0]}-{ideal['Nitrogen'][1]})")

        # Phosphorus check
        if p < ideal['Phosphorus'][0]:
            recommendations.append(f"Phosphorus is low for {crop} (current: {p}, ideal: {ideal['Phosphorus'][0]}-{ideal['Phosphorus'][1]})")
        elif p > ideal['Phosphorus'][1]:
            recommendations.append(f"Phosphorus is high for {crop} (current: {p}, ideal: {ideal['Phosphorus'][0]}-{ideal['Phosphorus'][1]})")

        # Potassium check
        if k < ideal['Potassium'][0]:
            recommendations.append(f"Potassium is low for {crop} (current: {k}, ideal: {ideal['Potassium'][0]}-{ideal['Potassium'][1]})")
        elif k > ideal['Potassium'][1]:
            recommendations.append(f"Potassium is high for {crop} (current: {k}, ideal: {ideal['Potassium'][0]}-{ideal['Potassium'][1]})")
    else:
        recommendations.append(f"No specific recommendations available for {crop}")

    return "\n".join(recommendations) if recommendations else f"All nutrient levels are optimal for {crop}"

@app.route('/')
def home():
    # Get classes from the encoders
    soil_types = soil_encoder.classes_.tolist()  # ['Black', 'Clay', 'Loamy', 'Sandy']
    crop_types = crop_encoder.classes_.tolist()  # Your crop list
    
    return render_template('index.html', 
                         soil_types=soil_types,
                         crop_types=crop_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'temperature': float(request.form['temperature']),
            'soil': request.form['soil'],
            'nitrogen': float(request.form['nitrogen']),
            'phosphorus': float(request.form['phosphorus']),
            'potassium': float(request.form['potassium']),
            'crop': request.form['crop'].lower()
        }

        # Process input for fertilizer prediction
        num_features = np.array([[data['temperature'], data['nitrogen'], 
                                 data['phosphorus'], data['potassium']]])
        scaled_num = scaler.transform(num_features)
        
        encoded_soil = soil_encoder.transform([data['soil']])[0]
        encoded_crop = crop_encoder.transform([data['crop']])[0]

        model_input = np.array([[
            scaled_num[0][0],    # Temperature
            encoded_soil,        # Soil
            scaled_num[0][1],    # Nitrogen
            scaled_num[0][2],    # Phosphorus
            scaled_num[0][3],    # Potassium
            encoded_crop         # Crop
        ]])

        # Make predictions
        pred = model.predict(model_input)
        fertilizer = fertilizer_encoder.inverse_transform([np.argmax(pred)])[0]
        
        # Generate crop recommendations
        raw_input = [
            data['temperature'],
            data['soil'],
            data['nitrogen'],
            data['phosphorus'],
            data['potassium'],
            data['crop']
        ]
        crop_advice = crop_specific_recommendation(raw_input)

        return render_template('index.html', 
                              prediction_text=f'Recommended Fertilizer: {fertilizer}',
                              crop_advice=crop_advice,
                              soil_types=soil_encoder.classes_,
                              crop_types=crop_encoder.classes_)

    except Exception as e:
        return render_template('index.html', 
                              prediction_text=f'Error: {str(e)}',
                              soil_types=soil_encoder.classes_,
                              crop_types=crop_encoder.classes_)

if __name__ == '__main__':
    app.run(debug=True)