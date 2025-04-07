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


def get_original_image_path(image_id):
    """Returns the original image path for a given image ID (ignoring augmented images)."""
    path1 = os.path.join(config.IMAGE_DIR_PART1, f"{image_id}.jpg")
    path2 = os.path.join(config.IMAGE_DIR_PART2, f"{image_id}.jpg")
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None
    

def get_image_path(image_id):
    """Constructs a list of full paths for a given image ID, including both original and augmented images.

    It searches for:
      - Augmented images in 'HAM10000_images_part_1_aug' and 'HAM10000_images_part_2_aug' folders.
      - Original images in 'HAM10000_images_part_1' and 'HAM10000_images_part_2' folders.
    
    Returns:
      A list of paths that exist. If no files are found, returns an empty list.
    """
    paths = []
    # Define augmented directories (assumes they are alongside the original ones)
    aug_dir_part1 = os.path.join(os.path.dirname(config.IMAGE_DIR_PART1), "HAM10000_images_part_1_aug")
    aug_dir_part2 = os.path.join(os.path.dirname(config.IMAGE_DIR_PART2), "HAM10000_images_part_2_aug")
    
    # Check for all augmented variants.
    for folder in [aug_dir_part1, aug_dir_part2]:
        for suffix in ['_flip', '_rot', '_zoom', '_trans']:
            candidate = os.path.join(folder, f"{image_id}{suffix}.jpg")
            if os.path.exists(candidate):
                paths.append(candidate)
    
    # Check original images in both parts.
    candidate1 = os.path.join(config.IMAGE_DIR_PART1, f"{image_id}.jpg")
    candidate2 = os.path.join(config.IMAGE_DIR_PART2, f"{image_id}.jpg")
    if os.path.exists(candidate1):
        paths.append(candidate1)
    elif os.path.exists(candidate2):
        paths.append(candidate2)
    
    return paths


def prepare_data(metadata_df):
    """Prepares image paths and labels from the metadata.
    
    If classification is multiclass, get_image_path returns a list of file paths (augmented and original)
    and we explode the list. If binary classification is selected, we load only the original images.
    Then, if binary classification is selected, we convert the 'dx' labels into binary labels
    (only 'mel' is malignant, all others benign).
    """
    # Load image paths differently depending on classification type.
    if config.CLASSIFICATION_TYPE == 'binary':
        # For binary classification, use only original images.
        metadata_df['image_path'] = metadata_df['image_id'].apply(get_original_image_path)
    else:
        # For multiclass, load all images (augmented and original).
        metadata_df['image_path'] = metadata_df['image_id'].apply(get_image_path)
        # Explode if the column contains lists.
        if metadata_df['image_path'].apply(lambda x: isinstance(x, list)).any():
            metadata_df = metadata_df.explode('image_path')
    
    # Drop rows where no image path was found.
    metadata_df = metadata_df.dropna(subset=['image_path'])
    print(f"Data prepared. Found {metadata_df.shape[0]} images.")

    if config.CLASSIFICATION_TYPE == 'binary':
        # For binary classification: only 'mel' is malignant (1), others benign (0).
        metadata_df['label_binary'] = metadata_df['dx'].apply(lambda x: 1 if x == 'mel' else 0)
        # Cast to float32 and reshape to (n_samples, 1) for BinaryCrossentropy.
        labels = metadata_df['label_binary'].values.astype('float32').reshape(-1, 1)
        class_names = ['benign', 'malignant']
        print("Binary classification: 'mel' => malignant (1), others => benign (0)")
    else:
        label_encoder = LabelEncoder()
        metadata_df['label_encoded'] = label_encoder.fit_transform(metadata_df['dx'])
        labels = metadata_df['label_encoded'].values
        class_names = label_encoder.classes_
        print("Classes:", class_names)
        print("Label encoding:", dict(zip(class_names, range(len(class_names)))))

    # Print the number of instances per class (will reflect the original distribution).
    counts = metadata_df['dx'].value_counts()
    print("Number of instances per class:")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")

    image_paths = metadata_df['image_path'].values
    return image_paths, labels, class_names




'''def prepare_data(metadata_df):
    """Prepares image paths and labels from the metadata, including augmented images.
    
    This function applies get_image_path to produce a list of image paths for each image_id,
    then explodes the list so each row corresponds to one file (original or augmented).
    """
    # Get list of image paths (each entry will be a list)
    metadata_df['image_paths'] = metadata_df['image_id'].apply(get_image_path)
    # Drop rows where no images were found
    metadata_df = metadata_df[metadata_df['image_paths'].apply(lambda x: len(x) > 0)]
    # Explode so each row is one image file
    metadata_df = metadata_df.explode('image_paths')
    # Rename column for consistency
    metadata_df = metadata_df.rename(columns={'image_paths': 'image_path'})
    
    print(f"Data prepared. Found {metadata_df.shape[0]} images (including augmented images).")

    # Encode labels (dx column contains the lesion type)
    label_encoder = LabelEncoder()
    metadata_df['label_encoded'] = label_encoder.fit_transform(metadata_df['dx'])

    # Store class names for later use if needed
    class_names = label_encoder.classes_
    print("Classes:", class_names)
    print("Label encoding:", dict(zip(class_names, range(len(class_names)))))

    # Print the number of instances per class
    counts = metadata_df['dx'].value_counts()
    print("Number of instances per class:")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")

    image_paths = metadata_df['image_path'].values
    labels = metadata_df['label_encoded'].values

    return image_paths, labels, class_names'''


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
    img = img / 255.0

    '''# Apply augmentation if specified and if not already an augmented image
    if augment and ("aug" not in image_path.lower()):
        if augmentation_pipeline:
            # Add a batch dimension, apply augmentation, and remove the batch dimension.
            img = tf.expand_dims(img, axis=0)
            img = augmentation_pipeline(img, training=True)
            img = tf.reshape(img, config.INPUT_SHAPE)'''
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