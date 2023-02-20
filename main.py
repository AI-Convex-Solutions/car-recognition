import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import config
from dataset_preprocessing import CustomDataset, DatasetPreprocessing
from test import test_model
from train import train_model, Classifier
from utils import clear_memory


Path(config.IMAGE_PATHS).mkdir(parents=True, exist_ok=True)
Path(config.CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
Path(config.LOGS_PATHS).mkdir(parents=True, exist_ok=True)

# Saving all terminal output in a txt file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f"{config.LOGS_PATHS}logs",
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.info("Starting script...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Delete any previous cache from cuda.
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="Car Recognition")
parser.add_argument("-p", "--preprocess", action="store_true", help="Make dataset ready for training.")
parser.add_argument("-t", "--train", action="store_true", help="Train the model with config.py")
parser.add_argument("-e", "--evaluate", action="store_true", help="Test the trained model.")
args = parser.parse_args()


# Clean the data for the first time.
if args.preprocess:
    preprocessor = DatasetPreprocessing(
        database_path=config.DATASET_PATH,
        csv_path=config.TRAIN_CSV_FILE_PATH,
    )
    preprocessor.remove_missing_data(augmentation=config.PERFORM_AUGMENTATION)
    preprocessor.build_csv_from_dataset()
    clear_memory(preprocessor)

processor = DatasetPreprocessing(
    database_path=config.DATASET_PATH,
    csv_path=config.TRAIN_CSV_FILE_PATH,
)
num_classes, mean, std = processor.count_classes_mean_and_std()
clear_memory(processor)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

if args.train:
    dataset = CustomDataset(
        csv_path=config.TRAIN_CSV_FILE_PATH,
        transform=transform
    )

    # split into train and val data.
    split = int(np.floor(len(dataset) * config.VAL_SPLIT_SIZE))
    train_data, val_data = random_split(dataset, (len(dataset) - split, split))

    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        num_workers=5,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.BATCH_SIZE,
        num_workers=5,
        shuffle=True
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    dataset_sizes = {"train": len(train_data), "val": len(val_data)}

    # model = create_model(num_classes)
    model = Classifier(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
    )

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1
    )

    model = train_model(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        config.NUM_EPOCHS,
        dataloaders,
        dataset_sizes,
        checkpoint=False
    )

if args.evaluate:
    processor = DatasetPreprocessing(
        database_path=config.DATASET_PATH,
        csv_path=config.TEST_CSV_FILE_PATH,
    )
    _, mean, std = processor.count_classes_mean_and_std()
    clear_memory(processor)
    test_data = CustomDataset(
        csv_path=config.TEST_CSV_FILE_PATH,
        transform=transform
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE,
        num_workers=5,
        shuffle=True
    )
    test_model(test_data, test_loader, num_classes)

