from tensorflow.keras import layers, Model, Input

def identity_block_conv(x, filters, kernel_size=3, dropout_rate=0.3):
    """Residual block with no dimension change"""
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x

def conv_block(x, filters, kernel_size=3, dropout_rate=0.3):
    """Residual block with dimension change"""
    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)
    shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x

def identity_block_fc(x, units, dropout_rate=0.4):
    """Residual block for dense layers"""
    shortcut = x
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x

def build_improved_resnet_model(input_shape):
    """Constructs the Custom Residual CNN for Brain Age"""
    inputs = Input(shape=input_shape)  # (8, 30, 2400)
    
    # Permute to (Freq, Time, Channels) -> Treat channels as depth
    x = layers.Permute((2, 3, 1))(inputs) 
    
    # Initial reduction
    x = layers.Conv2D(32, (3, 7), strides=(1, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((1, 4), strides=(1, 4), padding='same')(x)
    
    # Residual Stages
    x = conv_block(x, 64, kernel_size=(3, 5), dropout_rate=0.3)
    x = identity_block_conv(x, 64, kernel_size=(3, 5), dropout_rate=0.3)
    x = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)
    
    x = conv_block(x, 128, kernel_size=(3, 3), dropout_rate=0.4)
    x = identity_block_conv(x, 128, kernel_size=(3, 3), dropout_rate=0.4)
    x = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)
    
    # Feature Aggregation
    x = layers.Conv2D(192, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Regressor Head
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x)
    
    x = identity_block_fc(x, 256, dropout_rate=0.5)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    return Model(inputs, outputs, name="BrainAge_ResNet8ch")