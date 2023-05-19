import io
import pickle

from PIL import Image
from torchvision import transforms
import json

from torchvision import models
import torch



LABELS = ["manufacturer", "car_model", "year"]
STATS_TRAIN_FILE_PATH = "/home/kryekuzhinieri/Documents/car_recognition_from_scratch/car_recognition_model/datasets/stats_train.pickle"
TEST_BEST_MODEL_PATH = "/home/kryekuzhinieri/Documents/car_recognition_from_scratch/car_recognition_model/models/best_model_2023-05-19 18:42:23.027538"
LABEL_CODES = "/home/kryekuzhinieri/Documents/car_recognition_from_scratch/car_recognition_model/datasets/label_codes.json"

def predict_result(image_bytes):
    # Process
    with open(STATS_TRAIN_FILE_PATH, "rb") as file:
        stats = pickle.load(file)
        num_classes, mean, std = stats["num_classes"], stats["mean"], stats["std"]

    with open(LABEL_CODES, "r") as file:
        json_codes = json.load(file)

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = transform(image).unsqueeze(0)

    # Predict
    model = Classifier(num_classes)
    model.load_state_dict(torch.load(TEST_BEST_MODEL_PATH))
    model.eval()

    with torch.no_grad():
        inputs_ = image
        outputs = model(inputs_)
        all_label_predictions = {}
        for label in LABELS:
            _, preds = torch.max(outputs[label], dim=1)
            all_label_predictions[label] = json_codes[label][preds]
        return all_label_predictions


class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT
        )
        number_of_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        # sets a value for each existing label. For example:
        # self.car_model = torch.nn.Linear(features, all_classes)
        for label in LABELS:
            setattr(
                self,
                label,
                torch.nn.Linear(
                    number_of_features,
                    len(set(num_classes[label]))
                )
            )

    def forward(self, x):
        x = self.resnet(x)
        data = {}
        # calls each label -> self.manufacturer(x)
        for label in LABELS:
            data[label] = getattr(self, label)(x)
        return data
