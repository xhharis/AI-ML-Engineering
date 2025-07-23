
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)[0]
    result = "Benign" if prediction == 1 else "Malignant"
    return render_template('index.html', prediction_text=f'Tumor is {result}')

if __name__ == "__main__":
    app.run(debug=True)
