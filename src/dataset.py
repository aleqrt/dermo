import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import config # Import configuration settings

# --- Data Loading ---

def load_metadata(metadata_path):
    """Loads the metadata CSV file."""
    try:
        df = pd.read_csv(metadata_path)
        print(f"Metadata loaded successfully from {metadata_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return None

def get_image_path(image_id):
    """Constructs the full path for a given image ID."""
    # --- Adjust this logic based on your image folder structure ---
    # Example 1: Images split into two folders
    path1 = os.path.join(config.IMAGE_DIR_PART1, f"{image_id}.jpg")
    path2 = os.path.join(config.IMAGE_DIR_PART2, f"{image_id}.jpg")
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    # Example 2: All images in a single folder
    # path = os.path.join(config.IMAGE_DIR, f"{image_id}.jpg")
    # if os.path.exists(path):
    #     return path
    else:
        # print(f"Warning: Image file not found for ID: {image_id}")
        return None # Handle missing images if necessary

def prepare_data(metadata_df):
    """Prepares image paths and labels from the metadata."""
    metadata_df['image_path'] = metadata_df['image_id'].apply(get_image_path)
    # Drop rows where image path couldn't be found
    metadata_df = metadata_df.dropna(subset=['image_path'])
    print(f"Data prepared. Found {metadata_df.shape[0]} images.")

    # Encode labels (dx column contains the lesion type)
    label_encoder = LabelEncoder()
    metadata_df['label_encoded'] = label_encoder.fit_transform(metadata_df['dx'])

    # Store class names for later use if needed
    class_names = label_encoder.classes_
    print("Classes:", class_names)
    print("Label encoding:", dict(zip(class_names, range(len(class_names)))))

    image_paths = metadata_df['image_path'].values
    labels = metadata_df['label_encoded'].values

    # Return paths, labels, and potentially other columns needed for stratification (like patient_id)
    # For standard StratifiedKFold, we only need labels.
    # For Triple Stratification, you'd return metadata_df[['patient_id', 'dx', ...]] as well
    return image_paths, labels, class_names


# --- Data Splitting ---

def get_train_val_split(image_paths, labels):
    """Splits data into training and validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels,
        test_size=config.VALIDATION_SPLIT_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=labels # Ensure distribution of classes is similar
    )
    print(f"Data split into Training ({len(X_train)}) and Validation ({len(X_val)}) sets.")
    return X_train, X_val, y_train, y_val

def get_kfold_splits(image_paths, labels):
    """Generates indices for K-Fold cross-validation splits."""
    # Note: For Triple Stratified K-Fold, you would use a custom implementation
    # that takes multiple grouping factors into account (e.g., patient_id, dx).
    # The library 'iterative-stratification' can sometimes handle multi-label stratification.
    # Here we use standard StratifiedKFold based on the target label ('dx').
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_indices = list(skf.split(image_paths, labels))
    print(f"Generated {len(fold_indices)} folds using StratifiedKFold.")
    return fold_indices


# --- Image Preprocessing and Augmentation ---

def decode_image(image_path):
    """Reads and decodes an image file."""
    img = tf.io.read_file(image_path)
    # Convert the compressed string to a 3D uint8 tensor.
    img = tf.io.decode_jpeg(img, channels=config.IMG_CHANNELS)
    # Resize the image to the desired size.
    img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH])
    # Normalize pixel values if needed (e.g., [0, 1] or [-1, 1])
    # Keras application models often have their own preprocessing layers or functions
    # img = img / 255.0
    return img

def build_augmentation_pipeline():
    """Builds a sequential model for data augmentation."""
    # Use Keras preprocessing layers for augmentation
    # These layers are active only during training.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal" if config.HORIZONTAL_FLIP else "vertical" if config.VERTICAL_FLIP else "horizontal_and_vertical" if config.HORIZONTAL_FLIP and config.VERTICAL_FLIP else "",
             input_shape=config.INPUT_SHAPE),
        tf.keras.layers.RandomRotation(config.ROTATION_RANGE),
        tf.keras.layers.RandomZoom(config.ZOOM_RANGE),
        tf.keras.layers.RandomTranslation(height_factor=config.HEIGHT_SHIFT_RANGE, width_factor=config.WIDTH_SHIFT_RANGE),
        # Add other augmentations as needed (e.g., RandomContrast, RandomBrightness)
    ], name='data_augmentation')
    return data_augmentation

# Create the augmentation pipeline once
augmentation_pipeline = build_augmentation_pipeline() if config.AUGMENTATION else None

def preprocess_image(image_path, label, augment):
    """Loads, preprocesses, and potentially augments an image."""
    img = decode_image(image_path)

    # Apply model-specific preprocessing (e.g., for EfficientNet, ResNet, ViT)
    # This is often done *inside* the model definition or just before feeding to the model
    # Example for EfficientNet:
    # img = tf.keras.applications.efficientnet.preprocess_input(img)
    # Example for standard [0, 1] scaling:
    img = img / 255.0

    # Apply augmentation if specified
    if augment and augmentation_pipeline:
        # Add a batch dimension (now shape becomes (1, H, W, C))
        img = tf.expand_dims(img, axis=0)
        img = augmentation_pipeline(img, training=True) # Important: set training=True
        # Remove the extra batch dimension by reshaping explicitly to the input shape
        img = tf.reshape(img, config.INPUT_SHAPE)  # e.g. (224, 224, 3)
    return img, label


# --- tf.data.Dataset Creation ---

def create_dataset(image_paths, labels, batch_size, augment=False, shuffle=True):
    """Creates a tf.data.Dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        # Perfect shuffling requires estimating buffer size
        buffer_size = len(image_paths) # Use full dataset size for perfect shuffle (memory intensive)
        # buffer_size = tf.data.AUTOTUNE # Or let TF decide
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Use map with num_parallel_calls for efficiency
    dataset = dataset.map(lambda x, y: preprocess_image(x, y, augment),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the data
    dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset