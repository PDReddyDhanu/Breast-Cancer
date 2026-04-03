import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# Load raw data
df = pd.read_csv("raw_qol_data.csv")


# Separate inputs and outputs
X = df.drop(["id","side_effect","severity","risk"], axis=1)

y_side = df["side_effect"]
y_sev = df["severity"]
y_risk = df["risk"]


# Convert text to numbers
X = pd.get_dummies(X)


le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

y_side = le1.fit_transform(y_side)
y_sev = le2.fit_transform(y_sev)
y_risk = le3.fit_transform(y_risk)


# Normalize (0–1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Combine everything
final_df = pd.DataFrame(X_scaled, columns=X.columns)

final_df["side_effect"] = y_side
final_df["severity"] = y_sev
final_df["risk"] = y_risk


# Save processed file
final_df.to_csv("processed_qol_data.csv", index=False)

print("Processed data saved")
