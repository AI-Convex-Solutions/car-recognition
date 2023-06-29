# Car Recognition (shqip)

DEMO: www.cardetect.tech

### Kushtet paraprake
1. Shkarko skedarin e modelit nga `https://www.dropbox.com/s/95tefoi8uz4fhor/alpha_model?dl=0`
dhe vendose në: `model/finished_models/alpha/`

2. Shkarko edhe skedarin tjetër `https://www.dropbox.com/s/hbftvpih9g4rshc/final_model_85.pt?dl=0`
dhe vendose në: `model/colors/`.

3. Shkarko skedarin `aa` dhe vendose në dosjen bazë `./resnet152-f82ba261.pth`.

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


# Car Recognition (English)

### Requirements
1. Download the file `https://www.dropbox.com/s/95tefoi8uz4fhor/alpha_model?dl=0`
and place it under: `model/finished_models/alpha/`

2. Download the file `https://www.dropbox.com/s/hbftvpih9g4rshc/final_model_85.pt?dl=0`
place it under: `model/colors/`.

3. Download the file `aa` and place it under the root repository `./resnet152-f82ba261.pth`.

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
