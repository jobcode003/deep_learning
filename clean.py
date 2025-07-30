import keras
import tensorflow as tf
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.applications import MobileNetV2


pretrained_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224, 3))
pretrained_model.trainable = False

train = tf.keras.preprocessing.image_dataset_from_directory(
    'crop_images',
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=16,
)
test = tf.keras.preprocessing.image_dataset_from_directory(
    'crop_images',
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224,224),
    batch_size=16,

)
data_augmentation=keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

class_names = train.class_names
num_classes = len(class_names)
normalization_layer = layers.Rescaling(1./255)

# 🔄 Apply augmentation and normalization
train = train.map(lambda x, y: (data_augmentation(normalization_layer(x)), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
test= test.map(lambda x, y: (normalization_layer(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

# ⚡ Optimize performance
train = train.shuffle(200).prefetch(buffer_size=tf.data.AUTOTUNE)
test = test.prefetch(buffer_size=tf.data.AUTOTUNE)
# Model
model=models.Sequential([
    pretrained_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax'),

])


early_stop =EarlyStopping(
    monitor='val_loss', patience=2,
    restore_best_weights=True,
    verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train, validation_data=test, epochs=5, callbacks=[early_stop, checkpoint])
model.evaluate(test)
