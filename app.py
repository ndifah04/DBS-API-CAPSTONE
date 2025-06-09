from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load model
model = load_model('keras_model.h5')

@app.route('/')
def home():
    return "Model API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil data input dari request (misalnya, data dalam format JSON)
        data = request.get_json()
        
        # Lakukan preprocessing data sesuai dengan model yang dibutuhkan (misal normalisasi, reshape)
        # Misalnya jika model menerima data dalam bentuk array 2D
        input_data = np.array(data['input']).reshape(1, -1)  # Sesuaikan bentuk data sesuai model Anda

        # Lakukan prediksi dengan model
        prediction = model.predict(input_data)
        
        # Kirimkan hasil prediksi dalam bentuk JSON
        return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
