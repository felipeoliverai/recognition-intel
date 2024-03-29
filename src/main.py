import os 
import time 
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# %matplotlib inline 
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import SpatialDropout2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical, get_file

seed = 42 
np.random.seed(seed)
tf.random.set_seed(seed=seed)

tf.config.list_physical_devices(device_type='GPU')


#### Dataset import

path = '/content/drive/My Drive/Deep Learning - Projetos/Intel Images/Dataset'


def load_dataset(path): 

  %%time 

  images = []
  labels = []
  data = []

  for dir in os.listdir(path):
    for file_name in tqdm(os.listdir(os.path.join(path, dir))):

      image_path = os.path.join(path, dir, file_name)
      image = load_img(image_path, target_size=(150, 150, 3), color_mode="rgb")
      image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
      image /= 255.0

      images.append(image)
      labels.append(dir)


  dataframe = pd.DataFrame({"Images": images, "Label": labels})
  return images, labels, dataframe

images, labels, data = load_dataset(path=path)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# samples per class 
data['Label'].value_counts()



#### Explore images 

def visualize_image(data, column, index):

  plt.figure(figsize=(12,6))

  img = data[column][index]
  img = array_to_img(img)
  plt.imshow(img)
  plt.title(data["Label"][index], fontsize=15) 
  plt.tight_layout()

visualize_image(data=data, column="Images", index=6005) # 7000


#visualize image

plt.figure(figsize=(12,6))

img = data["Images"][5600]
img = array_to_img(img)
plt.imshow(img)
plt.title(data["Label"][5600], fontsize=15) 
plt.tight_layout()

# list of images 
list_images = data["Images"]

plt.figure(figsize=(12,6))

list_images = data["Images"]

img = list_images[300]
img = array_to_img(img)
plt.imshow(img)
plt.title(data["Label"][300], fontsize=15) 
plt.tight_layout()




#### Splitting train/test images


# splitting image sets 
X = np.array(images)
y = np.array(data["Label"])


label = LabelEncoder()
y = label.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)


print("Train shape: {} ----- {} \n  Teste shape: {} ------ {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# class labels
print(label.inverse_transform([0,1,2,3,4,5]))


#### Building Architecture CNN

def CNN():

  # Sequential model 
  model = Sequential()
  model.add(Input(shape=(150,150,3)))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2), padding="valid"))
  model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2), padding="valid"))
  model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2), padding="valid"))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(units=128, activation="relu"))
  model.add(Dense(units=6, activation="softmax"))

  return model

model = CNN()
model.compile(optimizer=Adam(lr=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.summary()

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test)

def plot_performance_model(history):

  """ Show the performance model based in
  loss (Cross Entropy) and accuracy both train and test"""


  fig = plt.figure(figsize=(15,7))


  # loss 
  plt.subplot(1,2,1)
  plt.plot(history.history["loss"], color="blue", label="Train loss")
  plt.plot(history.history["val_loss"], color="orange", label="Test loss")
  plt.title("Loss", fontsize=15)
  plt.xlabel("Epochs", fontsize=13)
  plt.ylabel("Loss", fontsize=13)
  plt.legend()
  plt.tight_layout()
  plt.show()



  # accuracy 
  plt.subplot(1,2,2)
  plt.plot(history.history["accuracy"], color="blue", label="Train accuracy")
  plt.plot(history.history["loss"], color="orange", label="Test accuracy")
  plt.title("Accuracy", fontsize=15)
  plt.xlabel("Epochs", fontsize=13)
  plt.ylabel("Accuracy", fontsize=13)
  plt.legend()
  plt.tight_layout()
  plt.show()
