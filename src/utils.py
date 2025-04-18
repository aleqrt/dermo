import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

import config 

# ---
# Utility functions for plotting and metrics
# ---
def plot_training_history(history, fold=None):
    """
    Plots the training and validation accuracy and loss.
    Saves the plot to a file.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add overall title with fold number if provided
    if fold is not None:
        plt.suptitle(f'Fold {fold+1} Training History', fontsize=16)
        plot_filename = f'training_history_fold_{fold+1}.png'
    else:
        plt.suptitle('Training History', fontsize=16)
        plot_filename = 'training_history.png'

    # Save the plot
    save_path = os.path.join(config.LOG_DIR, f"fold_{fold+1}", plot_filename)
    os.makedirs(os.path.join(config.LOG_DIR, f"fold_{fold+1}"), exist_ok=True)
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    # plt.show() # Optionally display the plot
    plt.close() # Close the figure to free memory


def plot_roc_curves(y_true, y_pred_probs, class_names, fold=None):
    """
    Computes and saves ROC curves.
    
    For binary classification (config.CLASSIFICATION_TYPE=='binary'):
      - Expects y_true as binary labels and y_pred_probs as a single probability per sample.
      - Plots one ROC curve.
    
    For multiclass classification:
      - Expects y_true as integer labels and y_pred_probs with shape (n_samples, n_classes).
      - Computes a one-vs-all ROC curve for each class.
    
    Parameters:
      y_true: array-like of shape (n_samples,) or (n_samples, 1)
          True labels.
      y_pred_probs: array-like of shape (n_samples, 1) for binary or (n_samples, n_classes) for multiclass.
          Predicted probabilities.
      class_names: list
          List of class names.
      fold: int, optional
          Fold number for saving the plots (used to create a fold-specific folder).
    """
    fold_subdir = f"fold_{fold+1}" if fold is not None else "test" # Use 'test' subdir if fold is None
    save_dir = os.path.join(config.LOG_DIR, fold_subdir) # Use config.LOG_DIR as base
    os.makedirs(save_dir, exist_ok=True)
    
    if config.CLASSIFICATION_TYPE == 'binary':
        # For binary classification, flatten predictions.
        y_pred_probs = np.squeeze(y_pred_probs)
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Binary Classification)')
        plt.legend(loc="lower right")
        
        plot_filename = "roc_curve_binary.png"
        save_path = os.path.join(save_dir, plot_filename)
        plt.savefig(save_path)
        print(f"ROC curve (binary) saved to {save_path}")
        plt.close()
    else:
        # Multiclass: one ROC curve per class.
        n_classes = len(class_names)
        # Convert labels to one-hot encoding.
        y_true_onehot = label_binarize(y_true, classes=list(range(n_classes)))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Class {i}: {class_names[i]}')
            plt.legend(loc="lower right")
            
            plot_filename = f"roc_curve_class_{class_names[i]}.png"
            save_path = os.path.join(save_dir, plot_filename)
            plt.savefig(save_path)
            print(f"ROC curve for class {class_names[i]} saved to {save_path}")
            plt.close()


# ---
# Define a custom multiclass ROC-AUC metric
# ---
@tf.keras.utils.register_keras_serializable(package='utils')
class MulticlassROC_AUC(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='roc_auc', **kwargs):
        super(MulticlassROC_AUC, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Use Keras's AUC metric configured for multi-label (one-hot) inputs.
        self.auc = tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse labels to one-hot encoding.
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes)
        self.auc.update_state(y_true_one_hot, y_pred, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()
    
    # Add get_config and from_config methods for proper serialization
    def get_config(self):
        config = super(MulticlassROC_AUC, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

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