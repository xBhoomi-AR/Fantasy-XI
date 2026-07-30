from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


DATA_DIR = Path("pipeline/final_xgboost/data")
MODEL_DIR = Path("pipeline/final_xgboost/models")


def load_data():
    print("Loading XGBoost dataset...")

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train):
    print()
    print("Training XGBoost...")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


def evaluate_model(model, X_val, y_val):
    print()
    print("Validation Results")

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    correlation = np.corrcoef(y_val, predictions)[0, 1]

    print(f"MAE         : {mae:.4f}")
    print(f"RMSE        : {rmse:.4f}")
    print(f"Correlation : {correlation:.4f}")

    return model


def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "xgboost_model.pkl",
    )

    print()
    print("Model saved.")
    print("Location:", MODEL_DIR)


def main():
    X_train, y_train, X_val, y_val = load_data()

    model = train_model(
        X_train,
        y_train,
    )

    evaluate_model(
        model,
        X_val,
        y_val,
    )

    save_model(model)


if __name__ == "__main__":
    main()