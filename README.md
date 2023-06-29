# DEMO
www.cardetect.tech

# Car Recognition (shqip)

### Modeli
Modeli është trajnuar duke përdorur fotografi të më shumë se 300.000 makinave. Çdo fotoje modeli i përgjigjet me:
- Kompaninë e automjetit (audi, bmw, etj.) - Saktësia 91%,
- Vitin e prodhimit (2013, 2019) - Saktësisa 81%,
- Modelin e automjetit (Golf 5, Clio) - Saktësia 33%,
- Ngjyrën e automjetit (Kuqe, Zezë) - [Model i gatshëm](https://github.com/nikalosa/Vehicle-Make-Color-Recognition). 


### Kushtet paraprake
1. Shkarko skedarin e modelit nga `https://www.dropbox.com/s/95tefoi8uz4fhor/alpha_model?dl=0`
dhe vendose në: `model/finished_models/alpha/`

2. Shkarko edhe skedarin tjetër `https://www.dropbox.com/s/hbftvpih9g4rshc/final_model_85.pt?dl=0`
dhe vendose në: `model/colors/`.

3. Shkarko skedarin `https://www.dropbox.com/s/4orqou90l7fp97v/resnet152-f82ba261.pth?dl=0` dhe vendose në dosjen bazë `./resnet152-f82ba261.pth`.

### Përdorimi
Fillimisht ndërtoje Dokerin: `docker build --tag car-recognition .`.
Pastaj filloje dokerin: `docker run -d -p 5000:5000 car-recogntion`.

### Kërkesat

```curl
curl --location --request POST 'http://localhost:5000/predict' \
--form 'file=@"/home/kryekuzhinieri/Desktop/audi_q3_2016.png"'
```

### Planet

- [] Viti duhet të ndryshohet nga `2013` në `2013, 2014, 2015` ose `2013-2015`.
- [x] Modelit duhet t'i shtohet `ngjyra`.
- [] Modeli duhet të ketë saktësi së paku 90% në të gjitha kategoritë.
- [] API-ji duhet të lejojë disa imazhe në një kërkesë.
- [] Pranimi i pikselave e jo i imazheve.

---

# Car Recognition (English)

### Model
Modeli është trajnuar duke përdorur fotografi të më shumë se 300.000 makinave. Çdo fotoje modeli i përgjigjet me:
The model is trained using 300,000+ car photos. The model returns:
- Car manufcaturer (audi, bmw, etj.) - Accuracy 91%,
- Manufactured year (2013, 2019) - Accuracy 81%,
- Car model (Golf 5, Clio) - Accuracy 33%,
- Car color (Red, Black) - [Pretrained](https://github.com/nikalosa/Vehicle-Make-Color-Recognition).

### Requirements
1. Download the file `https://www.dropbox.com/s/95tefoi8uz4fhor/alpha_model?dl=0`
and place it under: `model/finished_models/alpha/`

2. Download the file `https://www.dropbox.com/s/hbftvpih9g4rshc/final_model_85.pt?dl=0`
place it under: `model/colors/`.

3. Download the file `https://www.dropbox.com/s/4orqou90l7fp97v/resnet152-f82ba261.pth?dl=0` and place it under the root repository `./resnet152-f82ba261.pth`.

### Usage
Start by building the dockerfile: `docker build --tag car-recognition .`.
Then start docker: `docker run -d -p 5000:5000 car-recogntion`.

### Requests

```curl
curl --location --request POST 'http://localhost:5000/predict' \
--form 'file=@"/home/kryekuzhinieri/Desktop/audi_q3_2016.png"'
```

### Plans

- [] Year should change from one value to multiple years. 
- [x] Model should return color.
- [] Model must have an accuracy of 90% in all categories.
- [] User should be able to request multiple images at once.
- [] Accept pixels and not images for security.
