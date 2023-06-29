import torch
from torchvision import transforms
from PIL import Image


final_model = torch.load("final_model_85.pt", map_location="cpu")
class_names = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow']

my_image = "/home/kryekuzhinieri/Desktop/folf_white.png"


image = Image.open(my_image).convert('RGB')
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
print(f"The car class is {class_names[top_class.item()]} with a probability of {top_probability.item()}")
