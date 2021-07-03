from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, Conv2D, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def conv_block(input, num_filter, kernel_size, stride, pad, conv_name, relu_name):
    x = Conv2D(num_filter, kernel_size, strides=stride, padding=pad, use_bias=False,
               kernel_initializer='normal', name=conv_name)(input)

    x = ReLU(name=relu_name)(x)

    return x


def get_classification_model():
    input = Input(shape=(224, 224, 3))

    block_0 = conv_block(input=input, num_filter=16, kernel_size=3, stride=2, pad='same',
                         conv_name='b1_cnv2d_1', relu_name='b1_relu_1')

    # 1
    block_1_1 = conv_block(input=block_0, num_filter=32, kernel_size=3, stride=2, pad='same',
                           conv_name='b1_cnv2d_2', relu_name='b1_relu_2')

    block_1_2 = conv_block(input=block_1_1, num_filter=32, kernel_size=3, stride=1, pad='same',
                           conv_name='b1_cnv2d_3', relu_name='b1_relu_3')

    b1_add = Add()([block_1_1, block_1_2])

    # 2
    block_2_1 = conv_block(input=b1_add, num_filter=64, kernel_size=3, stride=2, pad='same',
                           conv_name='b2_cnv2d_1', relu_name='b2_relu_1')

    block_2_2 = conv_block(input=block_2_1, num_filter=64, kernel_size=3, stride=1, pad='same',
                           conv_name='b2_cnv2d_2', relu_name='b2_relu_2')

    b2_add = Add()([block_2_1, block_2_2])

    # 3
    block_3_1 = conv_block(input=b2_add, num_filter=128, kernel_size=3, stride=2, pad='same',
                           conv_name='b3_cnv2d_1', relu_name='b3_relu_1')

    block_3_2 = conv_block(input=block_3_1, num_filter=128, kernel_size=3, stride=1, pad='same',
                           conv_name='b3_cnv2d_2', relu_name='b3_relu_2')

    b3_add = Add()([block_3_1, block_3_2])

    # 4
    block_4_1 = conv_block(input=b3_add, num_filter=256, kernel_size=3, stride=2, pad='same',
                           conv_name='b4_cnv2d_1', relu_name='b4_relu_1')

    block_4_2 = conv_block(input=block_4_1, num_filter=256, kernel_size=3, stride=1, pad='same',
                           conv_name='b4_cnv2d_2', relu_name='b4_relu_2')

    b4_add = Add()([block_4_1, block_4_2])

    # 5
    block_5 = conv_block(input=b4_add, num_filter=512, kernel_size=3, stride=2, pad='same',
                         conv_name='b5_cnv2d_1', relu_name='b5_relu')

    X = Flatten(name="flatten")(block_5)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = Dense(64, activation="relu")(X)
    X = Dropout(0.25)(X)

    output = Dense(3, name='model_output', activation="softmax", kernel_initializer='he_uniform')(X)

    model = Model(input, output)

    model.summary()

    return model


def train_custom(images, input_labels):
    lab_encoder = LabelEncoder()
    labels_transformed = lab_encoder.fit_transform(input_labels)
    labels_transformed = to_categorical(labels_transformed, num_classes=3)

    image_aug = ImageDataGenerator(
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        fill_mode="nearest"
    )

    model = get_classification_model()

    initial_LR = 1e-4
    num_epochs = 50
    batch_size = 8

    (train_images, test_images, train_labels, test_labels) = train_test_split(images, labels_transformed,
                                                                              test_size=0.2,
                                                                              stratify=labels_transformed,
                                                                              random_state=42)

    optim = Adam(lr=initial_LR, decay=initial_LR / num_epochs)

    model.compile(loss="categorical_crossentropy", optimizer=optim,
                  metrics=["accuracy"])

    check_points = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0,
                                   save_best_only=True, mode='min', save_freq='epoch')

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    callbacks = [early_stop, check_points]

    model.fit(
        image_aug.flow(train_images, train_labels, batch_size=batch_size),
        steps_per_epoch=len(train_images) // batch_size,
        validation_data=(test_images, test_labels),
        validation_steps=len(test_images) // batch_size,
        epochs=num_epochs,
        callbacks=callbacks)

    model.save('/path/to/save/model.h5')
