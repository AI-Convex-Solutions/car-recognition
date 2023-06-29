from flask import Flask, request, jsonify
from api_calls import predict_result, predict_color
from flask_cors import CORS

app = Flask(__name__)

CORS(app, origins=['http://localhost:3001', 'https://auto-ai.onrender.com', 'https://cardetect.tech'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        print(file)
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            if predict_result(img_bytes, check_if_car=True):
                result = predict_result(img_bytes)
                color_prediction = predict_color(img_bytes)
                result["color"] = color_prediction
                return jsonify(result)
            return jsonify({'error': 'Please upload a valid car image.'})
        except Exception as e:
            print(e)
            return jsonify({'error': 'error during prediction'})


if __name__ == "__main__":
    app.run(debug=True)
