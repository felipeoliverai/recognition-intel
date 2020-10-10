import os 
import pandas as pd 
import numpy as np 
import time
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array


# load image dataset 
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
      image = image.astype("float32")
      image /= 255.0

      images.append(image)
      labels.append(dir)
 
  images = np.array(images, dtype = 'float32')
  labels = np.array(labels, dtype = 'int32')   
        
  data.append((images, labels))
  
  return data


if __main__ == "__name__":
  load_dataset()
