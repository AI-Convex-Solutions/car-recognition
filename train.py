import copy
import time

import numpy as np
import torch
import config
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from dataset_preprocessing import VmmrdbDataset, DatasetPreprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, batch_data in enumerate(dataloaders[phase]):
                inputs_ = batch_data["image"].to(device)
                labels_ = batch_data["label"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward and trach history if only train.
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs_)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels_)

                    # backward + optimize only if in training phase.
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs_.size(0)
                running_corrects += torch.sum(preds == labels_.data).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"Phase {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}, {time_elapsed % 60:.0f}s")
    print(f"Best val accuracy: {best_accuracy}")

    model.load_state_dict(best_model_weights)
    return model


processor = DatasetPreprocessing(path=config.DATASET_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std)
])

dataset = VmmrdbDataset(csv_path=config.CSV_FILE_PATH, transform=transform)

# split into train and val data.
split = int(np.floor(len(dataset) * config.VAL_SPLIT_SIZE))
train_data, val_data = random_split(dataset, (len(dataset) - split, split))

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}

dataset_sizes = {"train": len(train_data), "val": len(val_data)}

# for batch, data in enumerate(train_loader):
#     print(data["image"])
#     print(data["label"])
#     break


# Initialze Pretrained Resnet152.
model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

# Resnet152 has a final layer with 1000 classes. We change it to the number of our own clases.
model.fc = torch.nn.Linear(model.fc.in_features, len(processor.count_classes()))

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config.LEARNING_RATE,
    momentum=config.MOMENTUM,
    # weight_decay=config.WEIGHT_DECAY
)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(
    model,
    criterion,
    optimizer,
    exp_lr_scheduler,
    config.NUM_EPOCHS
)
