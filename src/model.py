from tensorflow.keras import layers, models

IMG_SIZE = 128     # size of the images fed to the model (scripts resize crops to this)
MODEL_SIZE = 64    # model internally shrinks to this -> much faster on CPU (try 96 later)


def conv_block(x, filters, n_convs=1):
    for _ in range(n_convs):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return layers.MaxPooling2D()(x)


def build_model(num_classes=36):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Input is 0..255; resize + scaling live inside the model so every script agrees.
    x = layers.Resizing(MODEL_SIZE, MODEL_SIZE)(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # Augmentation: active only during training, automatically off at inference.
    # No horizontal flip: it would turn one sign into a different/invalid one.
    x = layers.RandomRotation(0.06)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomZoom(0.12)(x)
    x = layers.RandomBrightness(0.25, value_range=(0, 1))(x)
    x = layers.RandomContrast(0.25)(x)

    x = conv_block(x, 32, 1)
    x = conv_block(x, 64, 1)
    x = conv_block(x, 128, 2)
    x = conv_block(x, 256, 1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


if __name__ == "__main__":
    build_model().summary()
