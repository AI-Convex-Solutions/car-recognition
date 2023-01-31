import os
import shutil

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class VmmrdbDataset(Dataset):
    """Vehicle Make and Model Recognition dataset (VMMRdb)"""

    def __init__(self, csv_path, transform=None):
        """"""
        self.cars_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.cars_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image_name = self.cars_frame.loc[idx, "image_name"]
        image = Image.open(self.cars_frame.loc[idx, "image_path"]).convert('RGB')
        label = self.cars_frame.loc[idx, "label"]

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": label,
        }

        return sample


class DatasetPreprocessing:
    """Preprocess the dataset for Pytorch."""

    def __init__(self, path):
        self.path = path

    def count_classes(self):
        """"""
        classes = [entry.name for entry in os.scandir(self.path) if entry.is_dir()]
        manufacturers, models = [], []
        for n in classes:
            values = n.split("_")
            manufacturers.append(values[0])
            models.append(values[1])
        print(f"Dataset has {len(classes)} different classes.")
        print(f"Dataset has {len(set(manufacturers))} different manufacturers.")
        print(f"Dataset has {len(set(models))} different car models.\n")
        return classes

    @staticmethod
    def compute_dataset_mean_and_std(path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = VmmrdbDataset(csv_path=path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=120, num_workers=0)
        number_of_images, mean, std = 0, 0, 0
        for batch_index, data in enumerate(dataloader):
            image = data["image"]
            # Rearrange the shape from [B, C, W, H] to be [B, C, W * H]:
            # [120, 3, 224, 224] -> [120, 3, 50176]
            image = image.view(image.size(0), image.size(1), -1)
            number_of_images += image.size(0)
            mean += image.mean(2).sum(0)
            std += image.std(2).sum(0)
        mean /= number_of_images
        std /= number_of_images
        print(f"The dataset mean is {mean} and the standard deviation: {std}.\n")
        return mean, std

    def build_csv_from_dataset(self):
        """"""
        data = []
        for entry in os.scandir(self.path):
            if entry.is_dir:
                for image in os.scandir(entry):
                    data.append(
                        {
                            "image_name": image.name,
                            "image_path": image.path,
                            "label_name": entry.name,
                        }
                    )
        data = pd.DataFrame(data=data)
        data["label"] = pd.factorize(data["label_name"])[0]
        train, test = train_test_split(data, test_size=config.TEST_SPLIT_SIZE)
        train.to_csv(path_or_buf=config.TRAIN_CSV_FILE_PATH, index=False)
        test.to_csv(path_or_buf=config.TEST_CSV_FILE_PATH, index=False)

    @staticmethod
    def remove_missing_data(path):
        for entry in os.scandir(path):
            if entry.is_dir:
                name = entry.name.split("_")
                if "" in name:
                    name.remove("")
                if "Tjetër" in name:
                    name.remove("Tjetër")
                if len(name) < 3:
                    shutil.rmtree(entry)
                    continue
                if "Mercedes-Benz" in name:
                    name[0] = "Mercedes Benz"
                name = [word.replace("ë", "e").lower() for word in name]
                name = "_".join(name)
                os.rename(entry, os.path.join(path, name))
        print("Dataset cleaned successfully!")
