import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import project modules
import config
import dataset
import models
import utils

# --- Setup ---
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
# tf.config.run_functions_eagerly(True)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Ensure output directories exist
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

# Mixed Precision (Optional, for performance on compatible GPUs)
if config.USE_MIXED_PRECISION:
    print("Using Mixed Precision Training.")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using Full Precision Training (float32).")


# --- Load and Prepare Data ---
print("Loading metadata...")
metadata_df = dataset.load_metadata(config.METADATA_FILE)

if metadata_df is not None:
    print("Preparing image paths and labels...")
    all_image_paths, all_labels, class_names = dataset.prepare_data(metadata_df)

    # --- Training Loop ---
    fold_histories = []
    fold_evaluations = []

    if config.USE_KFOLD:
        print(f"Starting {config.N_SPLITS}-Fold Cross-Validation...")
        fold_indices = dataset.get_kfold_splits(all_image_paths, all_labels)

        # Assuming `all_labels` contains the encoded labels
        class_weights = None
        if config.CLASSIFICATION_TYPE == 'classification':
            class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
            class_weights = dict(enumerate(class_weights))
            print("Computed class weights:", class_weights)

        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            print(f"\n===== FOLD {fold + 1}/{config.N_SPLITS} =====")

            # Get data for this fold
            X_train, X_val = all_image_paths[train_idx], all_image_paths[val_idx]
            y_train, y_val = all_labels[train_idx], all_labels[val_idx]
            print(f"Fold {fold+1}: Train samples={len(X_train)}, Validation samples={len(X_val)}")

            # Create tf.data.Dataset objects for this fold
            print("Creating training dataset...")
            train_ds = dataset.create_dataset(
                X_train, y_train, config.BATCH_SIZE, augment=config.AUGMENTATION, shuffle=True
            )
            print("Creating validation dataset...")
            val_ds = dataset.create_dataset(
                X_val, y_val, config.BATCH_SIZE, augment=False, shuffle=False
            )

            # Build the model (rebuild for each fold)
            print("Building model...")
            model = models.get_model(model_type=config.MODEL_TYPE)

            # Compile the model
            print("Compiling model...")
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
            model.compile(optimizer=optimizer,
                          loss=config.LOSS,
                          metrics=config.METRICS)
            # model.summary() # Optional: Print model summary

            # Define callbacks for this fold
            fold_model_save_path = os.path.join(config.MODEL_SAVE_DIR, f'{model.name}_fold_{fold+1}.keras')
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=fold_model_save_path,
                monitor='val_accuracy',     # Monitor validation accuracy
                save_best_only=True,        # Only save the best model
                save_weights_only=False,    # Save the entire model
                mode='max',                 # We want to maximize accuracy
                verbose=1
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',     # Monitor validation loss
                patience=20,            # Stop after 10 epochs with no improvement
                verbose=1,
                mode='min',             # Minimize loss
                restore_best_weights=True # Restore weights from the best epoch
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,             # Reduce LR by a factor of 5
                patience=10,             # Reduce after 5 epochs with no improvement
                verbose=1,
                min_lr=1e-6             # Minimum learning rate
            )
            # TensorBoard Logging (optional)
            # log_dir_fold = os.path.join(config.LOG_DIR, f"fold_{fold+1}")
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_fold, histogram_freq=1)

            callbacks_list = [checkpoint, early_stopping, reduce_lr] # Add tensorboard_callback if used

            # Train the model
            print(f"Starting training for fold {fold + 1}...")
            history = model.fit(
                train_ds,
                epochs=config.EPOCHS,
                validation_data=val_ds,
                callbacks=callbacks_list,
                class_weight=class_weights,
                verbose=1 # Set to 1 or 2 for progress updates
            )
            fold_histories.append(history)

            # Load the best model saved by ModelCheckpoint for evaluation
            print(f"Loading best model from fold {fold + 1} for evaluation...")
            best_model = tf.keras.models.load_model(fold_model_save_path)
            
            # Compute predictions on the validation set
            y_pred_probs = best_model.predict(val_ds)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)

            # Evaluate the best model on the validation set for this fold
            print(f"Evaluating fold {fold + 1} model on validation data...")
            evaluation = best_model.evaluate(val_ds, verbose=0)
            print(f"Fold {fold + 1} Validation Loss: {evaluation[0]:.4f}")
            print(f"Fold {fold + 1} Validation Accuracy: {evaluation[1]:.4f}")
            fold_evaluations.append(evaluation)

            # Plot training history for this fold
            utils.plot_training_history(history, fold=fold)
            utils.plot_roc_curves(y_val, y_pred_probs, class_names, fold=fold)

            # Clear session memory (important in K-Fold loops)
            tf.keras.backend.clear_session()

        # --- K-Fold Summary ---
        print("\n===== K-Fold Cross-Validation Summary =====")
        avg_loss = np.mean([eval[0] for eval in fold_evaluations])
        avg_acc = np.mean([eval[1] for eval in fold_evaluations])
        avg_roc = np.mean([eval[2] for eval in fold_evaluations])
        print(f"Average Validation Loss across {config.N_SPLITS} folds: {avg_loss:.4f}")
        print(f"Average Validation Accuracy across {config.N_SPLITS} folds: {avg_acc:.4f}")

    else:
        # --- Single Train/Validation Split Training ---
        print("Starting Single Train/Validation Split Training...")

        # Split data
        X_train, X_val, y_train, y_val = dataset.get_train_val_split(all_image_paths, all_labels)

        # Create datasets
        print("Creating training dataset...")
        train_ds = dataset.create_dataset(
            X_train, y_train, config.BATCH_SIZE, augment=config.AUGMENTATION, shuffle=True
        )
        print("Creating validation dataset...")
        val_ds = dataset.create_dataset(
            X_val, y_val, config.BATCH_SIZE, augment=False, shuffle=False
        )

        # Build the model
        print("Building model...")
        model = models.get_model(model_type=config.MODEL_TYPE)

        # Compile the model
        print("Compiling model...")
        model.compile(optimizer=config.OPTIMIZER,
                      loss=config.LOSS,
                      metrics=config.METRICS)
        model.summary() # Print model summary for single run

        # Define callbacks
        model_save_path = os.path.join(config.MODEL_SAVE_DIR, f'{config.MODEL_TYPE}_final.keras')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path, monitor='val_accuracy', save_best_only=True, save_weights_only=False, mode='max', verbose=1
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6
        )
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR, histogram_freq=1)
        callbacks_list = [checkpoint, early_stopping, reduce_lr] # Add tensorboard_callback if used

        # Train the model
        print("Starting training...")
        history = model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks_list,
            verbose=1
        )

        # Load the best model
        print("Loading best saved model for final evaluation...")
        best_model = tf.keras.models.load_model(model_save_path)

        # Evaluate the final model
        print("Evaluating final model on validation data...")
        evaluation = best_model.evaluate(val_ds, verbose=1)
        print(f"Final Validation Loss: {evaluation[0]:.4f}")
        print(f"Final Validation Accuracy: {evaluation[1]:.4f}")
        print(f"Final Validation ROC-AUC: {evaluation[2]:.4f}")
        

        # --- Optional: Detailed Evaluation (Confusion Matrix, Classification Report) ---
        print("\nGenerating detailed evaluation report on validation data...")
        y_pred_probs = best_model.predict(val_ds)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        # Need true labels (y_val) corresponding to the val_ds order
        # Re-create y_val if necessary or get it directly from the dataset
        # This assumes val_ds was created *without* shuffling
        print("Classification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=class_names))

        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred_classes))
        # You might want to plot the confusion matrix using seaborn/matplotlib


        # Plot training history
        utils.plot_training_history(history)

    print("\nTraining process finished.")

else:
    print("Could not load metadata. Exiting.")