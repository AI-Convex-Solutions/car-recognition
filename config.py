""""""

# Datasets
DATASET_PATH = "./databases/Sampledb"
TEST_CSV_FILE_PATH = "./datasets/test.csv"
TRAIN_CSV_FILE_PATH = "./datasets/train.csv"
JSON_LABELS_FILE_PATH = "./datasets/label_codes.json"

# Models
CHECKPOINT_PATH = "./models/checkpoints"
CHECKPOINT_NAME = f"{CHECKPOINT_PATH}/checkpoint"
BEST_MODEL_PATH = "./models/best_model"

# Files to save
IMAGE_PATHS = "./images/"
LOGS_PATHS = "./logs/"

# Model Parameters
BATCH_SIZE = 16
VAL_SPLIT_SIZE = 0.2
TEST_SPLIT_SIZE = 0.1
RANDOM_SEED = 42
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.2
MULTI_OUTPUT = True

# Label names
LABELS = ["manufacturer", "car_model", "year"]
