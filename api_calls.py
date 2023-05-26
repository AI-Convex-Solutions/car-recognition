import io
import json
import pickle

import torch
from PIL import Image
from torchvision import models
from torchvision import transforms

LABELS = ["manufacturer", "car_model", "year"]
STATS_TRAIN_FILE_PATH = "model/finished_models/alpha/datasets/stats_train.pickle"
TEST_BEST_MODEL_PATH = "model/finished_models/alpha/alpha_model.pt"
LABEL_CODES = "model/finished_models/alpha/datasets/label_codes.json"
COLOR_MODEL_PATH = "model/colors/final_model_85.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_result(image_bytes):
    # Process
    with open(STATS_TRAIN_FILE_PATH, "rb") as file:
        stats = pickle.load(file)
        num_classes, mean, std = (
            stats["num_classes"], stats["mean"], stats["std"]
        )

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
    model = Classifier(num_classes).to(device)
    model.load_state_dict(
        torch.load(
            TEST_BEST_MODEL_PATH,
            map_location=device
        )
    )
    model.eval()

    with torch.no_grad():
        inputs_ = image.to(device)
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
            weights=None  # models.ResNet152_Weights.DEFAULT
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


def predict_color(image_bytes):
    final_model = torch.load(COLOR_MODEL_PATH, map_location="cpu")
    class_names = [
        'Black',
        'Blue',
        'Brown',
        'Green',
        'Orange',
        'Red',
        'Silver',
        'White',
        'Yellow',
    ]

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)
    preds = final_model(image)
    probabilities = torch.nn.functional.softmax(preds, dim=1)
    top_probability, top_class = probabilities.topk(1, dim=1)
    return class_names[top_class.item()]
