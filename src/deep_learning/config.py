from pathlib import Path

import torch
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "Train"
TEST_DATA_DIR = DATA_DIR / "Test"
VALIDATION_DATA_DIR = DATA_DIR / "Validation"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "resnet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
IMAGE_SIZE = models.ResNet50_Weights.DEFAULT.transforms().crop_size[0]

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5

NUM_EPOCHS_PHASE_1 = 5
NUM_EPOCHS_PHASE_2 = 20
