import os
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import config # Import configuration settings

# ---
# Data Loading 
# ---
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


def balance_data(metadata_df, label_column='dx'):
    """
    Balance the dataset by removing samples from majority classes.
    Uses the median class cardinality as a reference point.
    
    Parameters:
    -----------
    metadata_df : pandas.DataFrame
        DataFrame containing the dataset metadata
    label_column : str
        Column name containing the class labels
    
    Returns:
    --------
    pandas.DataFrame
        Balanced DataFrame with samples removed from majority classes
    dict
        Class distribution statistics before and after balancing
    """
    # Store original dataframe for reference
    original_df = metadata_df.copy()
    
    # Check class distribution
    class_counts = metadata_df[label_column].value_counts()
    n_classes = len(class_counts)
    
    # Determine if binary or multiclass
    classification_type = 'binary' if n_classes == 2 else 'multiclass'
    
    # Calculate median class cardinality as reference
    median_count = int(np.median(class_counts.values))
    
    print(f"Classification type: {classification_type} ({n_classes} classes)")
    print("Original class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    print(f"\nUsing median class cardinality as reference: {median_count} samples")
    
    # Initialize balanced dataframe
    balanced_df = pd.DataFrame()
    
    # Balance classes
    for cls, count in class_counts.items():
        class_samples = metadata_df[metadata_df[label_column] == cls]
        
        if count > median_count:
            # Randomly select 'median_count' samples from this class
            balanced_class = class_samples.sample(n=median_count, random_state=config.RANDOM_STATE)
            print(f"  {cls}: Reduced from {count} to {median_count} samples (-{count - median_count})")
        else:
            # Keep all samples for classes with count <= median_count
            balanced_class = class_samples
            print(f"  {cls}: Kept all {count} samples")
        
        balanced_df = pd.concat([balanced_df, balanced_class])
    
    # Reset index for the balanced dataframe
    balanced_df = balanced_df.reset_index(drop=True)
    
    # Calculate distribution statistics
    stats = {
        'original_distribution': class_counts.to_dict(),
        'balanced_distribution': balanced_df[label_column].value_counts().to_dict(),
        'classification_type': classification_type,
        'n_classes': n_classes,
        'median_cardinality': median_count,
        'original_total': len(original_df),
        'balanced_total': len(balanced_df),
        'removed_samples': len(original_df) - len(balanced_df)
    }
    
    return balanced_df, stats


def prepare_data(metadata_df, balance=True, label_column='dx'):
    """
    Prepares image paths and labels from the metadata.
    Optionally balances the dataset using the median cardinality strategy.
    
    Parameters:
    -----------
    metadata_df : pandas.DataFrame
        DataFrame containing the dataset metadata
    balance : bool
        Whether to balance the dataset by removing samples from majority classes
    label_column : str
        Column name containing the class labels
    
    Returns:
    --------
    tuple
        (image_paths, labels, class_names, stats)
    """     
    # Continue with your existing data preparation code
    classification_type = 'binary' if len(metadata_df[label_column].unique()) == 2 else 'multiclass'
    
    # Load image paths differently depending on classification type
    if classification_type == 'binary':
        # For binary classification, use only original images
        metadata_df['image_path'] = metadata_df['image_id'].apply(get_original_image_path)
    else:
        # For multiclass, load all images (augmented and original)
        metadata_df['image_path'] = metadata_df['image_id'].apply(get_image_path)
        # Explode if the column contains lists
        if metadata_df['image_path'].apply(lambda x: isinstance(x, list)).any():
            metadata_df = metadata_df.explode('image_path')
    
    # Drop rows where no image path was found
    metadata_df = metadata_df.dropna(subset=['image_path'])
    print(f"Data prepared. Found {metadata_df.shape[0]} images after path resolution.")

    if classification_type == 'binary':
        # For binary classification: only 'mel' is malignant (1), others benign (0)
        metadata_df['label_binary'] = metadata_df[label_column].apply(lambda x: 1 if x == 'mel' else 0)
        # Cast to float32 and reshape to (n_samples, 1) for BinaryCrossentropy
        labels = metadata_df['label_binary'].values.astype('float32').reshape(-1, 1)
        class_names = ['benign', 'malignant']
    else:
        label_encoder = LabelEncoder()
        metadata_df['label_encoded'] = label_encoder.fit_transform(metadata_df[label_column])
        labels = metadata_df['label_encoded'].values
        class_names = label_encoder.classes_
    
    image_paths = metadata_df['image_path'].values

    balancing_stats = None
    # Apply balancing if requested
    if balance:
        metadata_df, balancing_stats = balance_data(
            metadata_df, 
            label_column=label_column
        )
    
    # Include balancing stats in return
    return image_paths, labels, class_names, balancing_stats


# ---
# Data Splitting 
# ---
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
    skf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_indices = list(skf.split(image_paths, labels))
    print(f"Generated {len(fold_indices)} folds using KFold.")
    return fold_indices


# ---
# Image Preprocessing and Augmentation 
# ---
def decode_image(image_path):
    """Reads and decodes an image file."""
    img = tf.io.read_file(image_path)
    # Convert the compressed string to a 3D uint8 tensor.
    img = tf.io.decode_jpeg(img, channels=config.IMG_CHANNELS)
    # Resize the image to the desired size.
    img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH])
    # Normalize pixel values if needed (e.g., [0, 1] or [-1, 1])
    # img = img / 255.0
    return img


def build_augmentation_pipeline():
    """Builds a sequential model for data augmentation."""
    # Use Keras preprocessing layers for augmentation
    # These layers are active only during training.
    augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal" if config.HORIZONTAL_FLIP else "vertical" if config.VERTICAL_FLIP else None,
             input_shape=config.INPUT_SHAPE),
        tf.keras.layers.RandomRotation(config.ROTATION_RANGE),
        tf.keras.layers.RandomZoom(config.ZOOM_RANGE),
        tf.keras.layers.RandomTranslation(height_factor=config.HEIGHT_SHIFT_RANGE, width_factor=config.WIDTH_SHIFT_RANGE),
    ], name='augmentations')

    return augmentations

# Create the augmentation pipeline once
augmentation_pipeline = build_augmentation_pipeline() if config.AUGMENTATION else None


def preprocess_image(image_path, label, augment):
    """Loads, preprocesses, and potentially augments an image."""
    img = decode_image(image_path)
    
    # Apply model-specific preprocessing
    img = img / 255.0
    
    # Check for "aug" using TensorFlow string operations
    if augment:
        contains_aug = tf.strings.regex_full_match(
            tf.strings.lower(image_path), ".*aug.*")
        
        should_augment = tf.logical_not(contains_aug)
        
        # Conditionally apply augmentation
        def apply_aug():
            augmented = tf.expand_dims(img, axis=0)
            augmented = augmentation_pipeline(augmented, training=True)
            return tf.reshape(augmented, config.INPUT_SHAPE)
        
        img = tf.cond(should_augment, 
                     true_fn=apply_aug,
                     false_fn=lambda: img)
    
    return img, label


# ---
# tf.data.Dataset Creation 
# ---
def create_dataset(image_paths, labels, batch_size, augment=False, shuffle=True):
    """Creates a tf.data.Dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
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

def load_test_metadata(metadata_path=config.TEST_METADATA_FILE):
    """Loads the test metadata CSV file."""
    try:
        df = pd.read_csv(metadata_path)
        print(f"Test metadata loaded successfully from {metadata_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Test metadata file not found at {metadata_path}")
        return None


def get_test_image_path(image_id, test_image_dir=config.TEST_IMAGES_DIR):
    """Returns the test image path for a given image ID."""
    path = os.path.join(test_image_dir, f"{image_id}.jpg")
    if os.path.exists(path):
        return path
    else:
        return None


def prepare_test_data(metadata_df, label_column='dx'):
    """
    Prepares test image paths and labels from the test metadata.
    
    Parameters:
    -----------
    metadata_df : pandas.DataFrame
        DataFrame containing the test metadata
    label_column : str
        Column name containing the class labels
    
    Returns:
    --------
    tuple
        (image_paths, labels, class_names)
    """
    # Get the mapping of class names from the training dataset
    # For ISIC2018 Task3, the classes should match HAM10000
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Ensure the DataFrame has an 'image_id' column
    if 'image' in metadata_df.columns:
        metadata_df['image_id'] = metadata_df['image'].str.replace('.jpg', '')
    
    # Get image paths
    metadata_df['image_path'] = metadata_df['image_id'].apply(get_test_image_path)
    
    # Drop rows where no image path was found
    metadata_df = metadata_df.dropna(subset=['image_path'])
    print(f"Test data prepared. Found {metadata_df.shape[0]} images after path resolution.")

    # Encode labels - ISIC2018 Task3 has ground truth in one-hot format
    # Convert one-hot encoded ground truth to class indices
    if all(col in metadata_df.columns for col in class_names):
        # The columns are already one-hot encoded
        labels = np.argmax(metadata_df[class_names].values, axis=1)
    else:
        # The column contains class names
        # Create a mapping of class names to indices
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        labels = metadata_df[label_column].map(class_to_idx).values
    
    image_paths = metadata_df['image_path'].values
    
    return image_paths, labels, class_names