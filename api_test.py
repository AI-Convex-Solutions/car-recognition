import requests

resp = requests.post(
    "http://localhost:5000/predict",
    files={
        'file': open('model/databases/Sampledb/bmw_x1_2011/8ZS20R65166AUE2SGYTD.jpg', 'rb'),
    }
)

print(resp.text)
