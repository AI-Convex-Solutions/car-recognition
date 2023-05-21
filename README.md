# Car Recognition

### Kushtet paraprake
Shkarko skedarin e modelit nga `https://www.dropbox.com/s/95tefoi8uz4fhor/alpha_model?dl=0`
dhe vendose në: `model/finished_models/alpha/`

### Përdorimi
Fillimisht ndërtoje Dokerin: `docker build --tag car-recognition .`.
Pastaj filloje dokerin: `docker run -d -p 5000:5000 car-recogntion`.

### Kërkesat

```curl
curl --location --request POST 'http://localhost:5000/predict' \
--form 'file=@"/home/kryekuzhinieri/Desktop/audi_q3_2016.png"'
```

### Planet

1. Viti duhet të ndryshohet nga `2013` në `2013, 2014, 2015`.
2. Modelit duhet t'i shtohet `ngjyra`.
3. Modeli duhet të ketë saktësi së paku 90% në të gjitha kategoritë.
4. API-ji duhet të lejojë disa imazhe në një kërkesë.
5. Pranimi i pikselave e jo i imazheve.
