import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import config
from dataset_preprocessing import VmmrdbDataset, DatasetPreprocessing
from test import test_model
from train import create_model, train_model

device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.get_num_threads())

parser = argparse.ArgumentParser(description="Car Recognition")
parser.add_argument("-p", "--preprocess", action="store_true", help="Make dataset ready for training.")
parser.add_argument("-t", "--train", action="store_true", help="Train the model with config.py")
parser.add_argument("-e", "--evaluate", action="store_true", help="Test the trained model.")
args = parser.parse_args()

processor = DatasetPreprocessing(path=config.DATASET_PATH)
num_classes = len(processor.count_classes())

if args.preprocess:
    processor.build_csv_from_dataset()

mean, std = processor.compute_dataset_mean_and_std(config.TRAIN_CSV_FILE_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

if args.train:
    dataset = VmmrdbDataset(csv_path=config.TRAIN_CSV_FILE_PATH, transform=transform)

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

    model = create_model(num_classes)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
    )

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = train_model(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        config.NUM_EPOCHS,
        dataloaders,
        dataset_sizes
        # checkpoint=True
    )

if args.evaluate:
    test_data = VmmrdbDataset(csv_path=config.TEST_CSV_FILE_PATH, transform=transform)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
    test_model(test_data, test_loader, num_classes)
