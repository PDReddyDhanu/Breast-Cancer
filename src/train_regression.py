import joblib
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data


# Load data
X, _, y_sev, _, _, _, enc2, _ = load_data("dataset/raw_qol_data.csv")


# Convert labels to scores
sev_text = enc2.inverse_transform(y_sev)

sev_map = {
    "Low": 20,
    "Medium": 60,
    "High": 90
}

y = [sev_map[s] for s in sev_text]


# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("Training Regression...")
model.fit(X, y)


# Save
joblib.dump(
    model,
    "../saved_models/regression_severity_model.pkl"
)

print("✅ Regression Model Saved")
