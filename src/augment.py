from imblearn.over_sampling import SMOTE
from preprocess import load_data


# Load data
X, y, _, _, _, _, _, _ = load_data("dataset/raw_qol_data.csv")


print("Original Data Size:", X.shape[0])


# Apply SMOTE
smote = SMOTE()

X_new, y_new = smote.fit_resample(X, y)


print("Augmented Data Size:", X_new.shape[0])
