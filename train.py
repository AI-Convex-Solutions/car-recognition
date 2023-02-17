import copy
import time
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import models

import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_model(num_classes):
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    # Resnet152 has a final layer with 1000 classes. We change it to the number of our own clases.
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model


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
        for label in config.LABELS:
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
        for label in config.LABELS:
            data[label] = getattr(self, label)(x)
        return data


def compute_loss(outputs, labels, criterion):
    all_preds = {}
    for label in config.LABELS:
        _, preds = torch.max(outputs[label], dim=1)
        all_preds[label] = preds

    losses = {k: criterion(outputs[k], labels[k]) for k, v in labels.items()}
    losses = sum(losses.values())
    return losses, all_preds


def save_checkpoint(
        epoch, model_state_dict, optimizer_state_dict, epoch_loss, epoch_acc):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "loss": epoch_loss,
        "accuracy": epoch_acc,
    }, config.CHECKPOINT_NAME)
    logging.info("-------Saved Checkpoint---------\n\n")


def load_checkpoint(model, optimizer):
    checkpoint = torch.load(config.CHECKPOINT_NAME)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_loss = checkpoint["loss"]
    epoch_acc = checkpoint["accuracy"]
    epoch = checkpoint["epoch"]
    logging.info(f"-------Loaded {config.CHECKPOINT_NAME} Checkpoint---------")
    return epoch, model, optimizer, epoch_loss, epoch_acc


def train_model(
        model, criterion, optimizer, scheduler, num_epochs,
        dataloaders, dataset_sizes, checkpoint=False):
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = {v: 0 for v in config.LABELS}
    train_loss, val_loss = [], []

    if checkpoint:
        previously_trained_epochs, model, optimizer, _, _ = load_checkpoint(
            model,
            optimizer
        )
        # Complete only the rest of epochs.
        num_epochs = num_epochs - previously_trained_epochs

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1} / {num_epochs}")
        logging.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = {v: 0 for v in config.LABELS}

            for batch_idx, batch_data in enumerate(dataloaders[phase]):
                inputs_ = batch_data["image"].to(device)
                labels = {
                    label: batch_data[label].to(device)
                    for label in config.LABELS
                }

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward and trach history if only train.
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs_)
                    losses, all_label_predictions = compute_loss(
                        outputs,
                        labels,
                        criterion
                    )

                    # backward + optimize only if in training phase.
                    if phase == "train":
                        # loss.backward()
                        losses.backward()
                        optimizer.step()

                # statistics
                running_loss += losses.item() * inputs_.size(0)
                for label in config.LABELS:
                    running_corrects[label] += torch.sum(
                        all_label_predictions[label] == labels[label].data
                    ).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = {
                k: v / dataset_sizes[phase]
                for k, v in running_corrects.items()
            }

            if phase == "train":
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)

            logging.info(
                f"Phase {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc}")

            # deep copy the model
            if phase == "val" and np.mean(list(epoch_acc.values())) > np.mean(list(best_accuracy.values())):
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

                save_checkpoint(
                    epoch,
                    best_model_weights,
                    optimizer.state_dict(),
                    epoch_loss,
                    epoch_acc
                )

    time_elapsed = time.time() - since
    logging.info(f"Training complete in {time_elapsed // 60:.0f}, {time_elapsed % 60:.0f}s")
    logging.info(f"Best val accuracy: {best_accuracy}")

    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="training loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="validation loss")
    plt.legend()
    plt.savefig(f"{config.IMAGE_PATHS}val_vs_training.png")
    return model
