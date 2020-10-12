import os 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import  get_file




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
