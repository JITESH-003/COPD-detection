from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('gru_model.pkl', 'rb'))  # Update with your actual model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Format output
    if prediction[0] == 1:
        result = "Respiratory Disease Detected"
    else:
        result = "No Respiratory Disease Detected"
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
