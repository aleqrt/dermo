import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
import config
import dataset
import models
import utils

# --- Setup ---
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Set paths for test data
TEST_IMAGES_DIR = '../dataset/ham10000/ISIC2018_Task3_Test_Images/ISIC2018_Task3_Test_Images/'
TEST_METADATA_FILE = '../dataset/ham10000/ISIC2018_Task3_Test_GroundTruth.csv'

# Define the path to the best model
MODEL_NAME = f"{config.MODEL_TYPE.upper()}_multiclass_best_overall.keras"
BEST_MODEL_PATH = os.path.join(config.MODEL_SAVE_DIR, MODEL_NAME)

# Make sure the logging directory exists
os.makedirs(config.LOG_DIR, 'test', exist_ok=True)


def load_test_metadata(metadata_path):
    """Loads the test metadata CSV file."""
    try:
        df = pd.read_csv(metadata_path)
        print(f"Test metadata loaded successfully from {metadata_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Test metadata file not found at {metadata_path}")
        return None


def get_test_image_path(image_id):
    """Returns the test image path for a given image ID."""
    path = os.path.join(TEST_IMAGES_DIR, f"{image_id}.jpg")
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


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots and saves the confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True class indices
    y_pred : array-like
        Predicted class indices
    class_names : list
        List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    save_path = os.path.join(config.LOG_DIR, 'test', 'test_confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    print("\n===== Starting Model Testing =====")
    
    # Load test metadata
    test_metadata_df = load_test_metadata(TEST_METADATA_FILE)
    
    if test_metadata_df is not None:
        # Prepare test data
        test_image_paths, test_labels, class_names = prepare_test_data(test_metadata_df)
        
        # Create dataset for testing
        test_ds = dataset.create_dataset(
            test_image_paths, test_labels, config.BATCH_SIZE, 
            augment=False, shuffle=False
        )
        
        # Check if the best model exists
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading the best model from: {BEST_MODEL_PATH}")
            best_model = tf.keras.models.load_model(
                BEST_MODEL_PATH, 
                custom_objects={'MulticlassROC_AUC': utils.MulticlassROC_AUC}
            )
            
            # Evaluate the model on test data
            print("Evaluating model on test data...")
            test_loss, test_accuracy, test_auc = best_model.evaluate(test_ds)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test AUC: {test_auc:.4f}")
            
            # Get predictions
            print("Generating predictions...")
            y_pred_probs = best_model.predict(test_ds)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            # Calculate and print classification report
            print("\nClassification Report:")
            report = classification_report(test_labels, y_pred_classes, target_names=class_names)
            print(report)
            
            # Save classification report to file
            with open(os.path.join(config.LOG_DIR, 'test', 'test_classification_report.txt'), 'w') as f:
                f.write(report)
            
            # Generate confusion matrix
            plot_confusion_matrix(test_labels, y_pred_classes, class_names)
            
            # Calculate and save class-wise AUC scores (one-vs-rest)
            print("\nClass-wise AUC Scores (One-vs-Rest):")
            # Convert labels to one-hot encoding for ROC AUC calculation
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(test_labels, classes=list(range(len(class_names))))
            
            class_auc_scores = {}
            for i, class_name in enumerate(class_names):
                auc_score = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
                class_auc_scores[class_name] = auc_score
                print(f"{class_name}: {auc_score:.4f}")
            
            # Plot ROC curves
            utils.plot_roc_curves(test_labels, y_pred_probs, class_names, fold=None)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'Class': class_names,
                'AUC': [class_auc_scores[c] for c in class_names]
            })
            metrics_df.to_csv(os.path.join(config.LOG_DIR, 'test', 'test_metrics.csv'), index=False)
            
            print("\nTesting completed successfully.")
        else:
            print(f"Error: Best model not found at {BEST_MODEL_PATH}")
    else:
        print("Could not load test metadata. Exiting.")