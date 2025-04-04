import matplotlib.pyplot as plt
import os
import config # Use config for save paths if needed

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
    save_path = os.path.join(config.LOG_DIR, plot_filename)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    # plt.show() # Optionally display the plot
    plt.close() # Close the figure to free memory