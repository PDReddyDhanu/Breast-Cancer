import gradio as gr
import tensorflow as tf
import joblib
import numpy as np


# Load trained models
cnn_model = tf.keras.models.load_model(
    "../saved_models/cnn_side_effect_model.h5"
)

severity_model = joblib.load(
    "../saved_models/regression_severity_model.pkl"
)

risk_model = joblib.load(
    "../saved_models/risk_classifier_model.pkl"
)


# Output labels
side_effect_labels = [
    "Fatigue",
    "Nausea",
    "Neuropathy",
    "Hematologic",
    "None"
]


def predict_result(
    age, stage, fatigue, pain, emotion,
    physical, social, cognitive,
    sleep, appetite, prev_nausea, prev_neuropathy
):
    """
    This function takes user input
    and returns prediction result
    """

    # Prepare input
    input_data = np.array([[
        age, stage, fatigue, pain, emotion,
        physical, social, cognitive,
        sleep, appetite, prev_nausea, prev_neuropathy
    ]])

    # Normalize (0-1)
    input_data = input_data / 100.0

    # For CNN
    cnn_input = input_data.reshape(
        1, input_data.shape[1], 1
    )

    # Predict Side Effect
    side_pred = cnn_model.predict(cnn_input)
    side_index = np.argmax(side_pred)
    side_effect = side_effect_labels[side_index]

    # Predict Severity Score
    toxicity_score = severity_model.predict(input_data)[0]

    # Predict Risk
    risk_pred = risk_model.predict(input_data)[0]
    risk_labels = ["Low", "Medium", "High"]
    risk_level = risk_labels[risk_pred]

    # Convert score to severity
    if toxicity_score > 70:
        severity = "High"
    elif toxicity_score > 40:
        severity = "Medium"
    else:
        severity = "Low"

    # Final output
    result = f"""
🩺 Prediction Result

-----------------------------
Side Effect : {side_effect}

Toxicity Score : {toxicity_score:.2f} / 100

Severity Level : {severity}

Overall Risk : {risk_level}

Confidence : {max(side_pred[0])*100:.2f} %
-----------------------------
"""

    return result


# Create Gradio UI
app = gr.Interface(

    fn=predict_result,

    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Cancer Stage (1-4)"),

        gr.Slider(0, 100, label="Fatigue Score"),
        gr.Slider(0, 100, label="Pain Score"),
        gr.Slider(0, 100, label="Emotional Score"),
        gr.Slider(0, 100, label="Physical Score"),
        gr.Slider(0, 100, label="Social Score"),
        gr.Slider(0, 100, label="Cognitive Score"),
        gr.Slider(0, 100, label="Sleep Score"),
        gr.Slider(0, 100, label="Appetite Score"),

        gr.Radio([0, 1], label="Previous Nausea (0=No, 1=Yes)"),
        gr.Radio([0, 1], label="Previous Neuropathy (0=No, 1=Yes)")
    ],

    outputs=gr.Textbox(label="Prediction Output"),

    title="AI-Based Chemotherapy Side Effect Predictor",

    description="""
Enter patient Quality of Life (QoL) scores to
predict chemotherapy side effects, severity,
and overall risk using AI.
"""
)

# Run App
app.launch(share=True)


