import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_hub as hub 
from transformers import TFViTModel

import config

# --- Model Building Functions ---

def build_cnn_model(input_shape, num_classes, pretrained_weights='imagenet', backbone='efficientnetb0'):
    """
    Builds a Convolutional Neural Network (CNN) model.
    Supported backbones: 'efficientnetb0', 'resnet50', 'resnet101', 'densenet101' (substituted with DenseNet121).
    """
    # Define input layer
    inputs = layers.Input(shape=input_shape)

    # --- Optional: Add application-specific preprocessing ---
    # Normalization [0,1] is done in dataset.py, otherwise use the following:
    # x = tf.keras.applications.efficientnet.preprocess_input(inputs) # No! Done in dataset.py now
    
    x = inputs 

    # Select the backbone model based on the provided parameter.
    backbone = backbone.lower()
    if backbone == 'efficientnetb0':
        print("Building CNN model (EfficientNetB0)...")
        BaseModel = tf.keras.applications.EfficientNetB0
    elif backbone == 'resnet50':
        print("Building CNN model (ResNet50)...")
        BaseModel = tf.keras.applications.ResNet50
    elif backbone == 'resnet101':
        print("Building CNN model (ResNet101)...")
        BaseModel = tf.keras.applications.ResNet101
    elif backbone == 'densenet101':
        # TensorFlow Keras does not provide DenseNet101; using DenseNet121 as a proxy.
        print("Building CNN model (DenseNet101 -> DenseNet121)...")
        BaseModel = tf.keras.applications.DenseNet121
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Load pre-trained base model (without the top classification layers)
    base_model = BaseModel(include_top=False,
                           weights=pretrained_weights,
                           input_shape=input_shape)

    # Freeze the base model layers initially (optional, for fine-tuning)
    # base_model.trainable = False

    # Pass input through the base model
    x = base_model(x, training=False)

    # Add custom classification head with two extra dense layers
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name='top_dropout')(x)

    # First additional dense layer
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name='dense_1_dropout')(x)

    # Second additional dense layer
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name='dense_2_dropout')(x)

    # Final output layer based on classification type.
    if config.CLASSIFICATION_TYPE == 'binary':
        outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)
        model_name = f'{backbone.upper()}_binary'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        model_name = f'{backbone.upper()}_multiclass'
    
    # Create the final model
    model = models.Model(inputs, outputs, name=model_name)
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
    if model_type in ['efficientnetb0', 'resnet50', 'resnet101', 'densenet121']:
        model = build_cnn_model(config.INPUT_SHAPE, 
                                config.NUM_CLASSES, 
                                pretrained_weights=config.PRETRAINED_WEIGHTS, 
                                backbone=model_type)
    elif model_type == 'ViT':
        model = build_vit_model(
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES,
            pretrained_weights=config.PRETRAINED_WEIGHTS # Adjust if ViT uses different source
        )
    else:
        model = build_cnn_model(config.INPUT_SHAPE, 
                                config.NUM_CLASSES, 
                                pretrained_weights=config.PRETRAINED_WEIGHTS, 
                                backbone='resnet50')
        raise ValueError(f"Unknown model type: {model_type}. Build default ResNet50 backbone model.")

    return model