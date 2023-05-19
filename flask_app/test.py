import requests

resp = requests.post(
    "http://localhost:5000/predict",
    files={
        'file': open('car_recognition_model/databases/Sampledb/audi_q7_2011/C3AJHI1E6V1NR08WF9DX.jpg', 'rb'),
    }
)

print(resp.text)
