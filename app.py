import sklearn

print(sklearn.__version__)
from flask import Flask, request, jsonify
import joblib  # or pickle if using sklearn
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("GBR_tuned_Combined_V1.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([data["features"]])  # Extract features from request
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
