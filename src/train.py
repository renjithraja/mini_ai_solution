import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import TARGET_COLUMN, DROP_COLUMNS, RANDOM_STATE, TEST_SIZE
from preprocessing import build_preprocessor


def main():
    # Load dataset
    df = pd.read_csv("data/churn.csv")

    # Drop non-informative columns
    df.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")

    # Encode target variable
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        )
    }

    best_model = None
    best_f1 = 0.0

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        f1 = f1_score(y_test, preds)
        print(f"{name} | F1-score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline

    # Save final model
    joblib.dump(best_model, "model.joblib")
    print("Model training completed. Best model saved.")


if __name__ == "__main__":
    main()
