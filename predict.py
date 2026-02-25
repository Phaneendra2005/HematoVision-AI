
import tensorflow as tf
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

IMG_SIZE = 224
CLASSES = ["eosinophil","lymphocyte","monocyte","neutrophil"]

model = tf.keras.models.load_model("hematovision_model.keras")

img = cv2.imread(args.image)
img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
img = img/255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0]
print("Prediction:", CLASSES[np.argmax(pred)])
