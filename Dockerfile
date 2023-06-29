FROM python:3.10-slim-buster 

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

# Installing pytorch for cpu. Same version with development. 
RUN pip install \
  torch==1.13.1+cpu \
  torchvision==0.14.1+cpu \
  torchaudio==0.13.1+cpu \
  -f https://download.pytorch.org/whl/cpu/torch_stable.html \
  && rm -rf /root/.cache/pip

WORKDIR /car_recognition

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY resnet152-f82ba261.pth /root/.cache/torch/hub/checkpoints/
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 5000
ENV FLASK_APP=api_main.py

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]
