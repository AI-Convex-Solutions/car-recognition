# Car Recognition

### Kushtet paraprake
1. Shkarko skedarin e modelit nga `https://www.dropbox.com/s/95tefoi8uz4fhor/alpha_model?dl=0`
dhe vendose në: `model/finished_models/alpha/`

2. Shkarko edhe skedarin tjetër `https://www.dropbox.com/s/hbftvpih9g4rshc/final_model_85.pt?dl=0`
dhe vendose në: `model/colors/`.

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
