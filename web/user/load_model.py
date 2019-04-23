from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

graph = tf.get_default_graph()
model = load_model("checkpoint-240e-val_acc_0.78.hdf5")

def predict(i):
    imgs = []
    img = image.load_img(i, target_size=(224, 224))
    img_array = image.img_to_array(img)
    imgs.append(img_array)
    im = np.array(imgs)
    global graph
    with graph.as_default():
        result = model.predict(im)
    return result