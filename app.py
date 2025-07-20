from flask import Flask, render_template, request
import pandas as pd
from src.load_data import load_data
from src.model import predict_churn  # create this function in model.py

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Example features
        features = {
            'tenure': float(request.form['tenure']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        df = pd.DataFrame([features])
        result = predict_churn(df)  # Your custom function
        return render_template('predict.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
