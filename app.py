from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and scalers
models = {
    "diabetes": load_model("model/diabetes_ann.h5"),
    "heart": load_model("model/heart_ann.h5")
}

scalers = {
    "diabetes": joblib.load("model/scaler.pkl"),
    "heart": joblib.load("model/scaler_heart.pkl")
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        disease = request.form.get("disease")

        try:
            features = []

            if disease == "diabetes":
                # Use unique names
                features = [
                    float(request.form['pregnancies']),
                    float(request.form['glucose']),
                    float(request.form['bloodpressure']),
                    float(request.form['skinthickness']),
                    float(request.form['insulin']),
                    float(request.form['bmi']),
                    float(request.form['dpf']),
                    float(request.form['age_diabetes'])
                ]

            elif disease == "heart":
                features = [
                    float(request.form['age_heart']),
                    float(request.form['sex']),
                    float(request.form['cp']),
                    float(request.form['trestbps']),
                    float(request.form['chol']),
                    float(request.form['fbs']),
                    float(request.form['restecg']),
                    float(request.form['thalach']),
                    float(request.form['exang']),
                    float(request.form['oldpeak']),
                    float(request.form['slope']),
                    float(request.form['ca']),
                    float(request.form['thal'])
                ]

            # Check for negative values
            if any(f < 0 for f in features):
                error = "Please enter realistic positive values."
            else:
                scaled_features = scalers[disease].transform([features])
                result = models[disease].predict(scaled_features)[0][0]
                prediction = "Positive" if result > 0.5 else "Negative"

        except Exception as e:
            error = f"Invalid input. Please check values. ({str(e)})"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)




