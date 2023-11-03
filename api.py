from flask import Flask, request, jsonify
import joblib
import shap
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = joblib.load('Good_model.joblib')

# Define a SHAP explainer for your model
explainer = shap.Explainer(model)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as JSON
        data = request.json
        features = data['features']

        # Make predictions
        prediction = model.predict([features])

        # Explain the prediction using SHAP
        shap_values = explainer.shap_values(np.array([features]))
        
        # Assuming prediction is binary (0 or 1), you can return the result as JSON
        result = {"prediction": int(prediction[0]), 'shap_values': shap_values[0].tolist()}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)




