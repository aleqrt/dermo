import os
import numpy as np
import tensorflow as tf
import pandas as pd
import config
from tensorflow.keras.preprocessing.image import apply_affine_transform
from dataset import get_image_path, decode_image

# Define output directories based on original image folder names.
OUTPUT_DIR_PART1 = os.path.join(os.path.dirname(config.IMAGE_DIR_PART1), "HAM10000_images_part_1_aug")
OUTPUT_DIR_PART2 = os.path.join(os.path.dirname(config.IMAGE_DIR_PART2), "HAM10000_images_part_2_aug")
os.makedirs(OUTPUT_DIR_PART1, exist_ok=True)
os.makedirs(OUTPUT_DIR_PART2, exist_ok=True)

def save_augmented_image(image_np, save_path):
    """Converts the image to uint8 and saves it as a JPEG."""
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    encoded_image = tf.io.encode_jpeg(image_np)
    tf.io.write_file(save_path, encoded_image)
    print(f"Saved augmented image to {save_path}")

def augment_and_save_image(image_path):
    """
    Loads an image from the given path, applies four augmentations (flip, rotation, zoom, translation)
    to the image (each applied separately), and saves the augmented images in the corresponding augmented folder.
    """
    # Determine output directory based on which part the image belongs to.
    if "HAM10000_images_part_1" in image_path:
        output_dir = OUTPUT_DIR_PART1
    elif "HAM10000_images_part_2" in image_path:
        output_dir = OUTPUT_DIR_PART2
    else:
        # Fallback in case the image path does not match either folder.
        output_dir = os.path.join(os.getcwd(), "augmented_images")
        os.makedirs(output_dir, exist_ok=True)

    # Use the base filename (without extension) to create new filenames.
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Load and resize the image using the existing decode function.
    # decode_image returns a tensor resized to config.IMG_HEIGHT x config.IMG_WIDTH.
    image_tensor = decode_image(image_path)
    # Ensure image is in uint8 format for augmentation functions.
    image_tensor = tf.cast(image_tensor, tf.uint8)
    # Convert tensor to a NumPy array (this script runs eagerly).
    image_np = image_tensor.numpy()

    # 1. Horizontal Flip
    augmented_flip = tf.image.flip_left_right(image_tensor).numpy()
    save_augmented_image(augmented_flip, os.path.join(output_dir, base_filename + '_flip.jpg'))

    # 2. Rotation: apply a fixed rotation angle (e.g. 36 degrees)
    # Note: apply_affine_transform works on numpy arrays.
    augmented_rot = apply_affine_transform(image_np, theta=36)
    save_augmented_image(augmented_rot, os.path.join(output_dir, base_filename + '_rot.jpg'))

    # 3. Zoom: use a fixed zoom factor (e.g. zoom in by 10% -> factor 1.1)
    augmented_zoom = apply_affine_transform(image_np, zx=1.1, zy=1.1)
    save_augmented_image(augmented_zoom, os.path.join(output_dir, base_filename + '_zoom.jpg'))

    # 4. Translation: shift image by a fraction of its dimensions
    tx = int(config.HEIGHT_SHIFT_RANGE * config.IMG_HEIGHT)  # vertical translation
    ty = int(config.WIDTH_SHIFT_RANGE * config.IMG_WIDTH)    # horizontal translation
    augmented_trans = apply_affine_transform(image_np, tx=tx, ty=ty)
    save_augmented_image(augmented_trans, os.path.join(output_dir, base_filename + '_trans.jpg'))

def augment_images_for_class(image_paths, class_name):
    """Augment and save images for a specific class."""
    print(f"Augmenting {len(image_paths)} images for class '{class_name}'")
    for img_path in image_paths:
        try:
            augment_and_save_image(img_path)
        except Exception as e:
            print(f"Failed to augment {img_path}: {e}")

if __name__ == "__main__":
    # Load metadata and determine the low-frequency classes.
    metadata = pd.read_csv(config.METADATA_FILE)
    # Get full image paths using the same logic as in dataset.py.
    metadata['image_path'] = metadata['image_id'].apply(get_image_path)
    metadata = metadata.dropna(subset=['image_path'])

    # Print overall class counts for reference.
    class_counts = metadata['dx'].value_counts()
    print("Class counts:")
    print(class_counts)

    # Define the low-frequency classes; adjust this list as needed.
    low_freq_classes = ['df', 'akiec', 'vasc', 'bcc']

    # For each low-frequency class, perform augmentation.
    for cls in low_freq_classes:
        cls_df = metadata[metadata['dx'] == cls]
        image_paths = cls_df['image_path'].tolist()
        augment_images_for_class(image_paths, cls)
