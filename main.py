import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("flower_cnn_model_ValAcc67.h5")

# Class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load image
img_path = input("Enter image path: ")
img = image.load_img(img_path, target_size=(128,128))

# Preprocess image
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted class:", predicted_class)
