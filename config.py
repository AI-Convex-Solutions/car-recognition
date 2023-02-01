""""""

DATASET_PATH = "./databases/VNSHdb"
CHECKPOINT_PATH = "./models/checkpoints"
CHECKPOINT_NAME = f"{CHECKPOINT_PATH}/checkpoint"
TRAIN_CSV_FILE_PATH = "./datasets/train.csv"
BEST_MODEL_PATH = "./models/best_model"
TEST_CSV_FILE_PATH = "./datasets/test.csv"
JSON_LABELS_FILE_PATH = "./datasets/label_codes.json"
IMAGE_PATHS = "./images/"
LOGS_PATHS = "./logs/"
BATCH_SIZE = 16
VAL_SPLIT_SIZE = 0.2
TEST_SPLIT_SIZE = 0.1
RANDOM_SEED = 42
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.2
