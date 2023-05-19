from flask import Flask, request, jsonify
from api_calls import predict_result


app = Flask(__name__)

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

        # try:
        img_bytes = file.read()
        result = predict_result(img_bytes)

        return jsonify(result)
#         except:
            # return jsonify({'error': 'error during prediction'})
