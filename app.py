from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib  # or pickle if using sklearn
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained models
models = {
    "gbr": joblib.load("GBR_tuned_Combined_V1.pkl"),
    "svr": joblib.load("best_model_svr.pkl"),
    "linear": joblib.load("linear_model.pkl"),
}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([data["features"]])  # Extract features from request

    # Get predictions from all models
    predictions = {
        model_name: model.predict(features).tolist()
        for model_name, model in models.items()
    }

    return jsonify(predictions)


if __name__ == "__main__":
    app.run(debug=True)
