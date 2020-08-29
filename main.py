import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import pathlib
import skimage
import skimage.io as io
from skimage.transform import resize
CT_COVID_DATA = "./data/"

data_dir = pathlib.Path(CT_COVID_DATA)
image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.png')))
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
    print(f.numpy())

val_size = int(image_count * 0.15)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
print (train_ds)
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

batch_size = 24
img_dim = 200
epochs = 10

def process_image(image):
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image, channels=3),
        lambda: tf.image.decode_png(image, channels=3))
    image = tf.image.resize(image, [img_dim, img_dim])
    image = tf.image.rgb_to_grayscale(image)    
    image = tf.image.adjust_contrast(image, 2)
    # image = 255 - image
    return image

def process_path(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2] == ["CT_NonCOVID", "CT_COVID"]
    image = tf.io.read_file(file_path)
    image = process_image(image)
    return image, tf.argmax(label)

def configure_dataset(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in train_ds.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())

train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)

# image_batch, label_batch = next(iter(train_ds))

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_batch[i].numpy().astype("uint8"), cmap='gray')
#     plt.xlabel(label_batch[i].numpy(), {'size':10})
# plt.show()

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(1, img_dim, img_dim,1)),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    # layers.Dense(32, activation='relu'),
    layers.Dense(2)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Evaluate
image = tf.io.read_file(tf.convert_to_tensor("normal.jpeg", dtype=tf.string))
image = process_image(image)
plt.imshow(image.numpy(), cmap='gray')
plt.show()
image = np.expand_dims(image, axis=0)

print (image.shape)
print (model.summary())
predictions = model.predict(image)
score = tf.nn.softmax(predictions[0])

print(
    "This image is most likely {} with a {:.2f} percent confidence."
    .format(["not infected with Covid-19", "Infected with Covid-19"][np.argmax(score)], 100 * np.max(score))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
