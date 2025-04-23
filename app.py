from flask import Flask, jsonify, request
import joblib
import numpy as np

# Load your model (use the path where you stored the model)
model = joblib.load('models/heatstroke_model.joblib')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (e.g., temperature, heart rate)
        data = request.get_json()

        # Prepare data for prediction (make sure data is in the right format)
        input_data = np.array([[
            data['temperature'],
            data['heartRate'],
            # Add other features here as needed
        ]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        
        # Return the prediction (you can format it as needed)
        return jsonify({'prediction': prediction[0]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
