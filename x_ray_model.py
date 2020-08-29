import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import pathlib
import skimage
import skimage.io as io
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import random

#Directories
CT_COVID_DATA = "./data/XRAY"

#Constants
BATCH_SIZE = 6
IMG_DIM = 224
EPOCHS = 15
LEARNING_RATE = 0.0001

def loadImage(path):
    image = io.imread(path, as_gray=True)
    image = resize(image, (IMG_DIM, IMG_DIM, 1))
    return image

def loadData():
  #LoadData
  data_dir = pathlib.Path(CT_COVID_DATA)
  paths = list(data_dir.glob('*/*.jpg')) + list(data_dir.glob('*/*.png')) + list(data_dir.glob('*/*.jpeg'))
  paths = list(map(lambda  x: x.as_posix(), paths))
  images = []
  labels = []
  print ("Loading images...")
  for path in paths:
      images.append(loadImage(path))
      label = path.split(os.sep)[-2]
      labels.append(label)
  images = np.array(images)
  labels = np.array(labels)
  lb = LabelBinarizer()
  labels = lb.fit_transform(labels)
  labels = to_categorical(labels)
  (trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=random.randint(0,100))
  return trainX, trainY, testX, testY

trainX, trainY, testX, testY = loadData()

print (trainX.shape, trainY.shape, testX.shape, testY.shape)
plt.figure(figsize=(10, 10))
for i in range(15):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(testX[i], cmap='gray')
    plt.xlabel(testY[i], {'size':10})
plt.show()
# # Evaluate
model = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(IMG_DIM, IMG_DIM,1)),
    layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.AveragePooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Flatten(),
    # layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    # layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2)
])
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
  optimizer=opt,
  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(
  trainX,
  trainY,
  epochs=EPOCHS,
  batch_size=BATCH_SIZE
)
model.evaluate(testX, testY, batch_size=BATCH_SIZE)
# print(
#     "This image is most likely {} with a {:.2f} percent confidence."
#     .format(["not infected with Covid-19", "Infected with Covid-19"][np.argmax(score)], 100 * np.max(score))
# )

acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
