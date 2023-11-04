from flask import Flask, request, jsonify
import joblib
import shap
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

good_model = joblib.load('Good_model.joblib')
bad_model = joblib.load('Bad_model.joblib')

# Define a SHAP explainer for your good_model
explainer = shap.Explainer(good_model)

@app.route('/predict_bad', methods=['POST'])
def predict_bad():
    try:
        # Get input data as JSON
        data = request.json
        features = data['features']

        # Make predictions
        prediction = bad_model.predict([features])
        
        result = {"prediction": int(prediction[0])}
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_good', methods=['POST'])
def predict_good():
    try:
        # Get input data as JSON
        data = request.json
        features = data['features']

        # Make predictions
        prediction = good_model.predict([features])

        # Explain the prediction using SHAP
        shap_values = explainer.shap_values(np.array([features]))
        
        # Assuming prediction is binary (0 or 1), you can return the result as JSON
        result = {"prediction": int(prediction[0]), 'shap_values': shap_values[0].tolist()}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False)




