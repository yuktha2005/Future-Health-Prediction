from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("model/diabetes_ann.h5")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            pregnancies = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            bloodpressure = float(request.form['bloodpressure'])
            skinthickness = float(request.form['skinthickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = float(request.form['age'])

            # Basic input validation
            if glucose <= 0 or bmi <= 0 or age <= 0:
                error = "Invalid input values. Please enter realistic health parameters."

            else:
                features = [
                    pregnancies, glucose, bloodpressure,
                    skinthickness, insulin, bmi, dpf, age
                ]

                scaled_features = scaler.transform([features])
                result = model.predict(scaled_features)[0][0]

                prediction = "Diabetic" if result > 0.5 else "Non-Diabetic"

        except ValueError:
            error = "Please enter valid numeric values."

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
