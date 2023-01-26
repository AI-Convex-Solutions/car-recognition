import matplotlib.pyplot as plt
import torch

import config
from train import create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model(test_data, test_loader, num_classes):
    model = create_model(num_classes)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    model.eval()

    accuracy = 0
    samples = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            inputs_ = batch_data["image"].to(device)
            labels_ = batch_data["label"].to(device)

            outputs = model(inputs_)
            _, preds = torch.max(outputs, dim=1)

            accuracy += torch.sum(preds == labels_.data).item()
            samples += preds.size(0)
    print(f"Model Accuracy: {(accuracy / samples):.4f}, Samples: {samples}")
    visualize_model(inputs_, preds, labels_)


def visualize_model(inputs, predicted, labels, num_images=4):
    inputs = inputs.cpu()
    fig = plt.figure()
    for i in range(num_images):
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title(f"Predicted: {predicted[i]} - Actual: {labels[i]}")
        plt.imshow(torch.movedim(inputs[i], 0, 2))
        plt.pause(0.001)
    plt.show()
