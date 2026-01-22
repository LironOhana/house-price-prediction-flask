import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Resolve paths relative to this file, not the current working directory
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
MODELS_DIR = BASE_DIR / "models"

rf_model = joblib.load(MODELS_DIR / "trained_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
scaler_y = joblib.load(MODELS_DIR / "scaler_y.pkl")


@app.route("/")
def home():
    return render_template("property_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    # The HTML form sends a list of feature values under the name "feature"
    features = request.form.getlist("feature")

    final_features = np.array(features, dtype=float).reshape(1, -1)
    final_features_scaled = scaler.transform(final_features)

    pred_scaled = rf_model.predict(final_features_scaled)[0]
    pred_scaled = np.array(pred_scaled).reshape(1, -1)

    prediction = scaler_y.inverse_transform(pred_scaled)[0][0]

    return render_template(
        "property_form.html",
        prediction_text=f"{prediction}"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
