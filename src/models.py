import tensorflow as tf
from tensorflow.keras import layers, models

import config
from vit import Patches, PatchEncoder, mlp


# ---
# Model Building Functions 
# ---
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
    try:
        if backbone == 'efficientnetb0':
            print("Building CNN model (EfficientNetB0)...")
            BaseModel = tf.keras.applications.EfficientNetB0
        elif backbone == 'vgg16':
            print("Building CNN model (VGG16)...")
            BaseModel = tf.keras.applications.VGG16
        elif backbone == 'resnet50':
            print("Building CNN model (ResNet50)...")
            BaseModel = tf.keras.applications.ResNet50
        elif backbone == 'resnet101':
            print("Building CNN model (ResNet101)...")
            BaseModel = tf.keras.applications.ResNet101
        elif backbone == 'densenet121':
            print("Building CNN model DenseNet121...")
            BaseModel = tf.keras.applications.DenseNet121
    except AttributeError:
        BaseModel = tf.keras.applications.ResNet50
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Load pre-trained base model (without the top classification layers)
    base_model = BaseModel(include_top=False,
                           weights=pretrained_weights,
                           input_shape=input_shape)

    # Freeze the base model layers initially (optional, for fine-tuning)
    base_model.trainable = True

    # Pass input through the base model
    x = base_model(x, training=False)  # Set training=False to avoid batch norm updates

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


def build_vit_model(input_shape=(224, 224, 3), patch_size=16, num_classes=7, projection_dim=64, transformer_layers=8,
                    num_heads=4, transformer_units=[128, 64], mlp_head_units=[2048, 1024], dropout_rate=0.1):
    print("Building ViT-B16 model ...")
    inputs = layers.Input(shape=input_shape)
    # Data augmentation
    augmented = layers.Rescaling(1.0 / 255)(inputs)
    augmented = layers.Resizing(input_shape[0], input_shape[1])(augmented)
    augmented = layers.RandomFlip("horizontal")(augmented)
    augmented = layers.RandomRotation(factor=0.02)(augmented)
    augmented = layers.RandomZoom(height_factor=0.2, width_factor=0.2)(augmented)

    # Create patches.
    patches = Patches(patch_size)(augmented)
    num_patches = (input_shape[0] // patch_size) ** 2

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = models.Model(inputs=inputs, outputs=logits)
    return model


# ---
# Model Selection Function 
# ---
def get_model(model_type=config.MODEL_TYPE):
    """Selects and builds the specified model."""
    if model_type in ['efficientnetb0', 'resnet50', 'resnet101', 'densenet121']:
        model = build_cnn_model(config.INPUT_SHAPE, 
                                config.NUM_CLASSES, 
                                pretrained_weights=config.PRETRAINED_WEIGHTS, 
                                backbone=model_type)
    elif model_type == 'vit':  
        raise ValueError(f"Model type: {model_type} under maintanance.")
        #model = build_vit_model(
        #    input_shape=config.INPUT_SHAPE,
        #    num_classes=config.NUM_CLASSES,
        #                                 patch_size=config.PATCH_SIZE,
        #)
    else:
        model = build_cnn_model(config.INPUT_SHAPE, 
                                config.NUM_CLASSES, 
                                pretrained_weights=config.PRETRAINED_WEIGHTS, 
                                backbone='resnet50')
        raise ValueError(f"Unknown model type: {model_type}. Build default ResNet50 backbone model.")
    
    return model