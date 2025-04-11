import os
import random
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

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
    all_image_paths, all_labels, class_names, stats = dataset.prepare_data(metadata_df)
    
    print("\nFinal class distribution after preparing data:")
    if stats:
        for cls, count in stats['balanced_distribution'].items():
            print(f"  {cls}: {count}")

    print(f"\nTotal samples: {len(all_image_paths)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    # --- Training Loop ---
    fold_histories = []
    fold_evaluations = []
    
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
        optimizer = tf.keras.optimizers.Adam(
                        learning_rate=config.LEARNING_RATE,
                        clipnorm=1.0
                    )
        model.compile(optimizer=optimizer,
                      loss=config.LOSS,
                      metrics=config.METRICS
                      )
        # model.summary() # Optional: Print model summary

        # Define callbacks for this fold
        fold_dir = os.path.join(config.MODEL_SAVE_DIR, f"fold{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_model_save_path = os.path.join(fold_dir, f'{model.name}_fold_{fold+1}.keras')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=fold_model_save_path,
            monitor='val_accuracy',     
            save_best_only=True,       
            save_weights_only=False,    
            mode='max',                 
            verbose=1
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',     
            patience=20,            
            verbose=1,
            mode='min',             
            restore_best_weights=True 
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,             
            patience=5,             
            verbose=1,
            min_lr=1e-8            
        )
        
        callbacks_list = [checkpoint, early_stopping, reduce_lr]

        # Train the model
        print(f"Starting training for fold {fold + 1}...")
        history = model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        fold_histories.append(history)

        # Load the best model saved by ModelCheckpoint for evaluation
        print(f"Loading best model from fold {fold + 1} for evaluation...")
        best_fold_model = tf.keras.models.load_model(fold_model_save_path)
        
        # Compute predictions on the validation set
        y_pred_probs = best_fold_model.predict(val_ds)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        # Evaluate the best model on the validation set for this fold
        print(f"Evaluating fold {fold + 1} model on validation data...")
        evaluation = best_fold_model.evaluate(val_ds, verbose=0)
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

    # Find the best fold based on validation accuracy
    best_fold_idx = np.argmax([eval[1] for eval in fold_evaluations])
    best_fold_acc = fold_evaluations[best_fold_idx][1]
    best_fold_num = best_fold_idx + 1  # Convert to 1-based indexing for display
    print(f"\nBest performing fold: {best_fold_num} with validation accuracy: {best_fold_acc:.4f}")

    # Path to the best fold's model
    model_name = models.get_model(model_type=config.MODEL_TYPE).name
    best_fold_model_path = os.path.join(config.MODEL_SAVE_DIR, f"fold{best_fold_num}", f"{model_name}_fold_{best_fold_num}.keras")

    # Path for the overall best model
    best_overall_model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}_best_overall.keras")

    # Copy the best fold's model to be the overall best model
    shutil.copy(best_fold_model_path, best_overall_model_path)

    print(f"Best model from fold {best_fold_num} saved as overall best model at: {best_overall_model_path}")

    print("\nTraining process finished.")

else:
    print("Could not load metadata. Exiting.")
