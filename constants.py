import os

import torch


IMG_SIZE               = 224
NUM_CLASSES            = 20
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_VAL_SPLIT_RATIO  = 0.15
BATCH_SIZE             = 100
LEARNING_RATE          = 0.01
NUM_EPOCHS             = 20

DEVICE                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH        = 'birds_classifier.pth'

CRITERION              = torch.nn.CrossEntropyLoss()
DATA_ROOT_DIR          = 'BirdsImages'
LABELS                 = os.listdir(DATA_ROOT_DIR)

USE_PRETRAINED         = True

