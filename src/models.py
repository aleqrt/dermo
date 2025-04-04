import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50
import tensorflow_hub as hub 
from transformers import TFViTModel

import config

# --- Model Building Functions ---

def build_cnn_model(input_shape, num_classes, pretrained_weights='imagenet'):
    """
    Builds a Convolutional Neural Network (CNN) model.
    Uses EfficientNetB0 as an example backbone.
    """
    print("Building CNN model (EfficientNetB0)...")
    # Define input layer
    inputs = layers.Input(shape=input_shape)

    # --- Optional: Add application-specific preprocessing ---
    # Example for EfficientNet, often done in dataset.py but can be here too
    # x = tf.keras.applications.efficientnet.preprocess_input(inputs) # No! Done in dataset.py now
    x = inputs # Assuming normalization [0,1] was done in dataset.py

    # Load pre-trained base model (without top classification layer)
    base_model = EfficientNetB0(include_top=False,
                                weights=pretrained_weights,
                                input_shape=input_shape) # Input shape needed if weights are loaded

    # Freeze the base model layers initially (optional, for fine-tuning)
    # base_model.trainable = False

    # Pass input through the base model
    x = base_model(x, training=False if pretrained_weights else True) # Set training appropriately

    # Add custom classification head
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name='top_dropout')(x) # Increased dropout
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create the final model
    model = models.Model(inputs, outputs, name='EfficientNetB0_CNN')
    print("CNN Model Built.")
    return model


def build_vit_model(input_shape, num_classes, pretrained_weights='imagenet'):
    """
    Builds a Vision Transformer (ViT) model.
    Uses Keras ViT implementation (available from TF 2.16+) or TF Hub/Transformers.
    Note: ViT preprocessing (patching, embedding) is handled internally by these layers.
          Ensure input images are appropriately sized and normalized (usually [0,1] or [-1,1]).
    """
    print("Using TensorFlow Hub for ViT...")
        
    vit_handle = "https://tfhub.dev/google/vit_b16/1"  # Base ViT model from TF Hub
    inputs = layers.Input(shape=input_shape)
    
    # Ensure input matches Hub model expectations (often float32 [0, 1])
    hub_layer = hub.KerasLayer(vit_handle, trainable=False)
    x = hub_layer(inputs)
    
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = models.Model(inputs, outputs, name='ViT_Hub_Model')
    print("ViT Model (TF Hub) Built.")
    return model


# --- Model Selection Function ---

def get_model(model_type=config.MODEL_TYPE):
    """Selects and builds the specified model."""
    if model_type == 'CNN':
        model = build_cnn_model(
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES,
            pretrained_weights=config.PRETRAINED_WEIGHTS
        )
    elif model_type == 'ViT':
        model = build_vit_model(
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES,
            pretrained_weights=config.PRETRAINED_WEIGHTS # Adjust if ViT uses different source
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'CNN' or 'ViT'.")

    return model