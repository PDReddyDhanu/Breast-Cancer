import tensorflow as tf
import numpy as np
from preprocess import load_data


# Load data
X, y, _, _, _, _, _, _ = load_data("dataset/raw_qol_data.csv")

X = X.reshape(X.shape[0], X.shape[1], 1)


# Load model
model = tf.keras.models.load_model(
    "../saved_models/cnn_side_effect_model.h5"
)


# Predict
pred = model.predict(X)


# Accuracy
acc = np.mean(
    np.argmax(pred, axis=1) == y
)


print("Model Accuracy:", acc)
