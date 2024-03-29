import json
import logging
import os
import pickle
import random
import shutil
import string
from multiprocessing.pool import ThreadPool

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import config


class CustomDataset(Dataset):
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
        image = Image.open(self.cars_frame.loc[idx, "image_path"]).convert(
            'RGB')
        label = self.cars_frame.loc[idx, "label"]
        manufacturer = self.cars_frame.loc[idx, "manufacturer"]
        car_model = self.cars_frame.loc[idx, "car_model"]
        year = self.cars_frame.loc[idx, "year"]

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": label,
            "manufacturer": manufacturer,
            "car_model": car_model,
            "year": year,
        }
        return sample


class DatasetPreprocessing:
    """Preprocess the dataset for Pytorch."""

    def __init__(self):
        pass

    @staticmethod
    def load_dataloader(csv_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = CustomDataset(csv_path=csv_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, num_workers=5)
        return dataloader

    @torch.no_grad()
    def count_classes_mean_and_std(self, csv_path, train_data=True):
        """"""
        logging.info("Started counting classes...")
        labels = ["label"]
        labels.extend(config.LABELS)
        results = {label: [] for label in labels}
        number_of_images, mean, std = 0, 0, 0
        dataloader = self.load_dataloader(csv_path)
        for batch_index, data in enumerate(dataloader):
            # Compute mean, std.
            image = data["image"]
            # Rearrange the shape from [B, C, W, H] to be [B, C, W * H]:
            # [120, 3, 224, 224] -> [120, 3, 50176]
            image = image.view(image.size(0), image.size(1), -1)
            number_of_images += image.size(0)
            mean += image.mean(2).sum(0)
            std += image.std(2).sum(0)

            # Count classes
            for col in labels:
                results[col].extend(data[col].tolist())

        mean /= number_of_images
        std /= number_of_images

        logging.info(
            f"The dataset mean is {mean} and the standard deviation: {std}.\n")
        logging.info(
            f"Dataset has {len(set(results['label']))} different classes.")
        logging.info(
            f"Dataset has {len(set(results['manufacturer']))} different manufacturers.")
        logging.info(
            f"Dataset has {len(set(results['car_model']))} different car models.")
        logging.info(
            f"Dataset has {len(set(results['year']))} different car years.\n")
        vals = {
            "num_classes": results,
            "mean": mean,
            "std": std,
        }
        filename = config.STATS_TRAIN_FILE_PATH if train_data else config.STATS_TEST_FILE_PATH
        with open(filename, "wb") as f:
            pickle.dump(vals, f, protocol=pickle.HIGHEST_PROTOCOL)

        return results, mean, std

    @torch.no_grad()
    def compute_dataset_mean_and_std(self, csv_path):
        dataloader = self.load_dataloader(csv_path)
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
        logging.info(
            f"The dataset mean is {mean} and the standard deviation: {std}.\n")
        return mean, std

    @staticmethod
    def build_csv_from_dataset(database_path):
        """"""
        logging.info("Building CSVs...")
        data = []
        for entry in os.scandir(database_path):
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
        label_codes = {}
        column_names = ["manufacturer", "car_model", "year"]
        data[column_names] = data["label_name"].str.split(
            pat="_", n=2,
            expand=True
        )
        column_names.append("label_name")
        for column in column_names:
            codes = pd.factorize(data[column])
            label_codes[column] = list(codes[1])
            if column == "label_name":
                data["label"] = codes[0]
            else:
                data[column] = codes[0]
        train, test = train_test_split(data, test_size=config.TEST_SPLIT_SIZE)
        train.to_csv(path_or_buf=config.TRAIN_CSV_FILE_PATH, index=False)
        test.to_csv(path_or_buf=config.TEST_CSV_FILE_PATH, index=False)
        with open(config.JSON_LABELS_FILE_PATH, "w") as f:
            json.dump(label_codes, f)
        logging.info("Dataset was build successfully!")

    @staticmethod
    def remove_missing_data(database_path, augmentation):
        logging.info("Removing missing data!")
        pool = ThreadPool()
        # transforms for the current image to make it smaller.
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        # transforms for the new images: rotating images randomly.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(0, 360))
        ])

        def worker(image):
            try:
                img = Image.open(image.path).convert('RGB')
                img = resize_transform(img)
                save_image(img, image.path)
                if augmentation:
                    new_image = transform(img)
                    image_name = ''.join(
                        random.choice(string.ascii_uppercase + string.digits)
                        for _ in range(10)
                    )
                    new_image_path = f"{os.path.dirname(image.path)}/{image_name}.jpg"
                    save_image(new_image, new_image_path)
            except Exception as e:
                os.remove(image)

        for entry in os.scandir(database_path):
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
                new_name = os.path.join(database_path, name)
                os.rename(entry, new_name)
                for image in os.scandir(new_name):
                    pool.apply_async(worker, (image,))
        pool.close()
        pool.join()
        logging.info("Dataset cleaned successfully!")

    @staticmethod
    def merge_datasets(path1, path2, new_dataset_path):
        os.makedirs(new_dataset_path, exist_ok=True)
        shutil.copytree(path1, new_dataset_path, dirs_exist_ok=True)
        shutil.copytree(path2, new_dataset_path, dirs_exist_ok=True)
        logging.info("Datasets merged successfully!")
