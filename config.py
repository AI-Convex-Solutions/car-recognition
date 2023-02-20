""""""
from datetime import datetime

# Datasets
DATASET_PATH = "./databases/VNSHdb"
TEST_CSV_FILE_PATH = "./datasets/test.csv"
TRAIN_CSV_FILE_PATH = "./datasets/train.csv"
JSON_LABELS_FILE_PATH = "./datasets/label_codes.json"

# Models
CHECKPOINT_PATH = "./models/checkpoints"
CHECKPOINT_NAME = f"{CHECKPOINT_PATH}/checkpoint"
BEST_MODEL_PATH = f"./models/best_model_{datetime.now()}"
TEST_BEST_MODEL_PATH = "./models/best_model_2023-02-17 18:19:55.448156"

# Files to save
IMAGE_PATHS = "./images/"
LOGS_PATHS = "./logs/"

# Model Parameters
BATCH_SIZE = 16
VAL_SPLIT_SIZE = 0.2
TEST_SPLIT_SIZE = 0.1
RANDOM_SEED = 42
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.1

# Label names
LABELS = ["manufacturer", "car_model", "year"]
