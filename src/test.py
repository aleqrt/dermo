import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# Import project modules
import config
from dataset import load_test_metadata, prepare_test_data, create_dataset
from utils import plot_confusion_matrix, plot_roc_curves, MulticlassROC_AUC

def parse_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description="Test skin lesion classification model")
    
    # Model selection
    parser.add_argument("--model-path", help="Path to model file (overrides default)")
    
    # Test data configuration
    parser.add_argument("--test-metadata-path", help="Path to test metadata CSV")
    parser.add_argument("--test-images-dir", help="Path to test images directory")
    
    # Other settings
    parser.add_argument("--batch-size", type=int, help="Batch size for testing")
    parser.add_argument("--output-dir", help="Directory to save test results")
    
    return parser.parse_args()

def update_config(args):
    """Update config based on arguments."""
    if args.test_metadata_path:
        config.TEST_METADATA_FILE = args.test_metadata_path
        print(f"Test metadata path updated to: {config.TEST_METADATA_FILE}")
    
    if args.test_images_dir:
        config.TEST_IMAGES_DIR = args.test_images_dir
        print(f"Test images directory updated to: {config.TEST_IMAGES_DIR}")
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        print(f"Batch size updated to: {config.BATCH_SIZE}")
    
    if args.output_dir:
        config.TEST_LOG_DIR = args.output_dir
        print(f"Test output directory updated to: {config.TEST_LOG_DIR}")

def main():
    # Process command line arguments if any
    args = parse_args()
    update_config(args)
    
    # --- Setup ---
    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
    
    # Define the path to the best model (from args or config)
    MODEL_PATH = args.model_path if args.model_path else config.BEST_MODEL_PATH
    print(f"Using model path: {MODEL_PATH}")
    
    # Make sure the logging directory exists
    os.makedirs(config.TEST_LOG_DIR, exist_ok=True)
    
    print("\n===== Starting Model Testing =====")
    
    # Load test metadata
    test_metadata_df = load_test_metadata(config.TEST_METADATA_FILE)
    
    if test_metadata_df is not None:
        # Prepare test data
        test_image_paths, test_labels, class_names = prepare_test_data(test_metadata_df)
        
        if test_image_paths is None or len(test_image_paths) == 0:
            print(f"Error: No valid test images found. Check if the test images directory exists: {config.TEST_IMAGES_DIR}")
            return 1
            
        print(f"Found {len(test_image_paths)} test images with {len(class_names)} classes")
        
        # Create dataset for testing
        test_ds = create_dataset(
            test_image_paths, test_labels, config.BATCH_SIZE, 
            augment=False, shuffle=False
        )
        
        # Check if the model exists
        if os.path.exists(MODEL_PATH):
            print(f"Loading the model from: {MODEL_PATH}")
            try:
                best_model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    custom_objects={'MulticlassROC_AUC': MulticlassROC_AUC}
                )
                
                # Evaluate the model on test data
                print("Evaluating model on test data...")
                evaluation_results = best_model.evaluate(test_ds)
                
                # Handle variable number of metrics
                metrics_names = best_model.metrics_names
                for i, metric_name in enumerate(metrics_names):
                    print(f"Test {metric_name}: {evaluation_results[i]:.4f}")
                
                # Get predictions
                print("Generating predictions...")
                y_pred_probs = best_model.predict(test_ds)
                
                # Handle binary vs multiclass
                if len(y_pred_probs.shape) == 2 and y_pred_probs.shape[1] > 1:
                    # Multiclass
                    y_pred_classes = np.argmax(y_pred_probs, axis=1)
                else:
                    # Binary
                    y_pred_probs = np.squeeze(y_pred_probs)
                    y_pred_classes = (y_pred_probs > 0.5).astype(int)
                
                # Calculate and print classification report
                print("\nClassification Report:")
                report = classification_report(test_labels, y_pred_classes, target_names=class_names)
                print(report)
                
                # Save classification report to file
                report_path = os.path.join(config.TEST_LOG_DIR, 'test_classification_report.txt')
                with open(report_path, 'w') as f:
                    f.write(f"Model: {MODEL_PATH}\n\n")
                    for i, metric_name in enumerate(metrics_names):
                        f.write(f"Test {metric_name}: {evaluation_results[i]:.4f}\n")
                    f.write("\n")
                    f.write(report)
                
                print(f"Classification report saved to {report_path}")
                
                # Generate confusion matrix
                try:
                    plot_confusion_matrix(test_labels, y_pred_classes, class_names)
                    print("Confusion matrix generated and saved.")
                except Exception as e:
                    print(f"Error generating confusion matrix: {e}")
                
                # Calculate and save class-wise AUC scores (one-vs-rest)
                print("\nClass-wise AUC Scores (One-vs-Rest):")
                try:
                    # Handle binary vs multiclass for AUC calculation
                    if len(class_names) <= 2:
                        # Binary classification
                        auc_score = roc_auc_score(test_labels, y_pred_probs)
                        class_auc_scores = {'binary': auc_score}
                        print(f"Binary AUC: {auc_score:.4f}")
                    else:
                        # Convert labels to one-hot encoding for ROC AUC calculation
                        y_true_bin = label_binarize(test_labels, classes=list(range(len(class_names))))
                        
                        class_auc_scores = {}
                        for i, class_name in enumerate(class_names):
                            auc_score = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
                            class_auc_scores[class_name] = auc_score
                            print(f"{class_name}: {auc_score:.4f}")
                    
                    # Plot ROC curves
                    plot_roc_curves(test_labels, y_pred_probs, class_names, fold=None)
                    
                    # Save metrics to CSV
                    if len(class_names) <= 2:
                        metrics_df = pd.DataFrame({
                            'Metric': ['AUC'],
                            'Value': [class_auc_scores['binary']]
                        })
                    else:
                        metrics_df = pd.DataFrame({
                            'Class': class_names,
                            'AUC': [class_auc_scores[c] for c in class_names]
                        })
                    
                    metrics_path = os.path.join(config.TEST_LOG_DIR, 'test_metrics.csv')
                    metrics_df.to_csv(metrics_path, index=False)
                    print(f"Metrics saved to {metrics_path}")
                
                except Exception as e:
                    print(f"Error calculating AUC scores: {e}")
                
                print("\nTesting completed successfully.")
                return 0
            except Exception as e:
                print(f"Error during model evaluation: {e}")
                return 1
        else:
            print(f"Error: Model not found at {MODEL_PATH}")
            return 1
    else:
        print(f"Error: Could not load test metadata from {config.TEST_METADATA_FILE}")
        return 1

if __name__ == "__main__":
    exit(main())