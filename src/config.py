
import os
from pathlib import Path
import tensorflow as tf

from utils import MulticlassROC_AUC

# Set the random seed for reproducibility
RANDOM_STATE = 42

# --- Path Resolution ---
# Get the absolute path to the source directory (where this config.py file is located)
SRC_DIR = Path(__file__).resolve().parent

# Get the absolute path to the project root (parent directory of src)
PROJECT_ROOT = SRC_DIR.parent
DATA_ROOT = PROJECT_ROOT.parent

# --- Data Configuration ---
DATA_DIR = os.path.join(DATA_ROOT, 'dataset/ham10000/')
METADATA_FILE = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR_PART1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_PART2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
# Or if images are in one folder:
# IMAGE_DIR = os.path.join(DATA_DIR, 'images/')

# --- Test Data Configuration ---
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'ISIC2018_Task3_Test_Images', 'ISIC2018_Task3_Test_Images')
TEST_METADATA_FILE = os.path.join(DATA_DIR, 'ISIC2018_Task3_Test_GroundTruth.csv')

# --- Classification Type ---
# Options: 'multiclass', 'binary'
# For binary classification, only melanoma ('mel') is malignant; all others are benigns
CLASSIFICATION_TYPE = 'multiclass'

# --- Model Configuration ---
MODEL_TYPE = 'densenet121' # Options: 'efficientnetb0', 'vgg16', 'resnet50', 'resnet101', 'densenet121', 'vit'

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
PATCH_SIZE = 16
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

if CLASSIFICATION_TYPE == 'binary':
    NUM_CLASSES = 1  # One output neuron with sigmoid activation.
    LOSS = tf.keras.losses.BinaryCrossentropy()
    METRICS = ['accuracy', tf.keras.metrics.AUC(name='auc')]
else:
    NUM_CLASSES = 7  # HAM10000 has 7 lesion types.
    LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
    METRICS = ['accuracy', MulticlassROC_AUC(NUM_CLASSES)]

# Pre-trained weights: 'imagenet' or None or path to custom weights
# For ViT/DINO, specific pre-trained sources might be needed depending on implementation in models.py
PRETRAINED_WEIGHTS = 'imagenet'

# --- Training Configuration ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 3e-5

# Data augmentation settings
AUGMENTATION = True
ROTATION_RANGE = 0.1
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False # Usually not recommended for skin lesions
FILL_MODE = 'nearest'

# --- K-Fold / Split Configuration ---
# If True, uses Stratified K-Fold. If False, uses a single train/validation split.
# Note: Triple Stratified K-Fold (like the Kaggle kernel) requires more complex implementation
# involving multiple grouping factors (e.g., patient_id, diagnosis).
# For simplicity, this example uses standard Stratified K-Fold or train/val split.
USE_KFOLD = True
N_SPLITS = 5 # Number of folds if USE_KFOLD is True
VALIDATION_SPLIT_RATIO = 0.2 # Used if USE_KFOLD is False
RANDOM_STATE = 42 # For reproducibility

# --- Output Configuration ---
MODEL_SAVE_DIR = f'models/{CLASSIFICATION_TYPE}/{MODEL_TYPE}/'
LOG_DIR = f'logs/{CLASSIFICATION_TYPE}/{MODEL_TYPE}/'

# --- Hardware Configuration ---
# Set to True if using mixed precision (requires compatible GPU)
USE_MIXED_PRECISION = False # Can speed up training on NVIDIA GPUs

# --- Best Model Configuration ---
# Define the naming convention for the model saved after K-Fold comparison
BEST_OVERALL_MODEL_NAME = f"{MODEL_TYPE.upper()}_{CLASSIFICATION_TYPE}_best_overall.keras"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, BEST_OVERALL_MODEL_NAME)
# Note: MODEL_SAVE_DIR depends on CLASSIFICATION_TYPE and MODEL_TYPE

# --- Output Configuration ---
# Add a specific subdir for test logs/outputs
TEST_LOG_DIR = os.path.join(LOG_DIR, 'test/')