import shap
import tensorflow as tf
from preprocess import load_data


# Load data
X, _, _, _, _, _, _, _ = load_data("dataset/raw_qol_data.csv")

X = X.reshape(X.shape[0], X.shape[1], 1)


# Load model
model = tf.keras.models.load_model(
    "../saved_models/cnn_side_effect_model.h5"
)


# SHAP Explainer
explainer = shap.DeepExplainer(model, X[:100])

shap_values = explainer.shap_values(X[:10])


# Plot
shap.summary_plot(shap_values, X[:10])
