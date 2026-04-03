from preprocess import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Load data
X, y, _, _, _, _, _, _ = load_data("dataset/raw_qol_data.csv")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {

    "Random Forest": RandomForestClassifier(),

    "Logistic Regression": LogisticRegression(max_iter=1000)
}


for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(name, "Accuracy:", acc)
