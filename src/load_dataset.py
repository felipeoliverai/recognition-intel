import os 
import time 
import pandas as pd 
import numpy as np 
from tqdm import tqdm


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

if __name__ == "__main__":
  load_dataset(path=path)
