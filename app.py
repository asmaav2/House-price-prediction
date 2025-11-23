from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['sqft_living']),
        float(request.form['bedrooms']),
        float(request.form['bathrooms']),
        float(request.form['floors']),
        float(request.form['view']),
        float(request.form['condition'])
    ]
    prediction = model.predict([np.array(data)])
    return render_template('index.html', prediction_text=f"Predicted Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
