import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class VmmrdbDataset(Dataset):
    """Vehicle Make and Model Recognition dataset (VMMRdb)"""

    def __init__(self, csv_path, transform=None):
        """"""
        self.cars_frame = pd.read_csv(csv_path).iloc[:1200, ]
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
        print(f"Dataset has {len(classes)} different classes.")
        # show how many car companies and how many models
        # here
        return classes

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
        data.to_csv(path_or_buf="dataset.csv", index=False)
