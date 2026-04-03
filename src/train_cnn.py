import tensorflow as tf
from preprocess import load_data


# Load data
X, y, _, _, _, _, _, _ = load_data("dataset/raw_qol_data.csv")

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)


# CNN Model
model = tf.keras.Sequential([

    tf.keras.layers.Conv1D(
        64, 2, activation="relu",
        input_shape=(X.shape[1], 1)
    ),

    tf.keras.layers.MaxPooling1D(),

    tf.keras.layers.Conv1D(32, 2, activation="relu"),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(5, activation="softmax")
])


# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train
print("Training CNN...")
model.fit(X, y, epochs=25, batch_size=32)


# Save
model.save("../saved_models/cnn_side_effect_model.h5")

print("✅ CNN Model Saved")
