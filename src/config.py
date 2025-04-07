import tensorflow as tf
import os
from utils import MulticlassROC_AUC

# --- Data Configuration ---
DATA_DIR = '../dataset/ham10000/'
METADATA_FILE = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR_PART1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_PART2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
# Or if images are in one folder:
# IMAGE_DIR = os.path.join(DATA_DIR, 'images/')

# --- Classification Type ---
# Options: 'multiclass', 'binary'
# For binary classification, only melanoma ('mel') is malignant; all others are benign.s
CLASSIFICATION_TYPE = 'binary'

# --- Model Configuration ---
MODEL_TYPE = 'efficientnetb0' # Options: 'efficientnetb0', 'resnet50', 'resnet101', 'densenet121', 'vit'

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
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
BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 1e-3

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
MODEL_SAVE_DIR = f'trained_models/{CLASSIFICATION_TYPE}/{MODEL_TYPE}/'
LOG_DIR = f'logs/{CLASSIFICATION_TYPE}/{MODEL_TYPE}/'

# --- Hardware Configuration ---
# Set to True if using mixed precision (requires compatible GPU)
USE_MIXED_PRECISION = False # Can speed up training on NVIDIA GPUs