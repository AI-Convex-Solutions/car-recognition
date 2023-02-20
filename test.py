import matplotlib.pyplot as plt
import torch
import logging

import config
from train import create_model, Classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model(test_data, test_loader, num_classes):
    # model = create_model(num_classes)
    model = Classifier(num_classes).to(device)
    model.load_state_dict(torch.load(config.TEST_BEST_MODEL_PATH))
    model.eval()

    # accuracy = 0
    accuracy = {v: 0 for v in config.LABELS}
    samples = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            inputs_ = batch_data["image"].to(device)
            labels = {
                label: batch_data[label].to(device)
                for label in config.LABELS
            }

            outputs = model(inputs_)
            all_label_predictions = {}
            for label in config.LABELS:
                _, preds = torch.max(outputs[label], dim=1)
                logging.info(label)
                logging.info(preds.data.tolist())
                logging.info(labels[label].data.tolist())
                logging.info("")

                accuracy[label] += torch.sum(preds == labels[label].data).item()
                all_label_predictions[label] = preds

            # visualize_model(inputs_, all_label_predictions, labels, batch_idx)
            samples += inputs_.size(0)
    accuracy = {k: v / samples for k, v in accuracy.items()}
    logging.info(f"Model Accuracy: {accuracy}, Samples: {samples}")


def visualize_model(inputs, predicted, labels, batch_id, num_images=4):
    inputs = inputs.cpu()
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        plt.tight_layout()
        title = [
            f"{v} - A: {labels[v][i].data} - P: {predicted[v][i].data}\n"
            for v in config.LABELS
        ]
        print(title)
        ax.set_title("".join(title), fontsize=7)
        plt.imshow(torch.movedim(inputs[i], 0, 2).cpu())
        plt.pause(0.001)
    plt.savefig(f"{config.IMAGE_PATHS}test_samples_{batch_id}.png", dpi=200)
