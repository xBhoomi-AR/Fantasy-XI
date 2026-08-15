from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .evaluation import metrics_frame, range_weights
from .features import feature_columns
from .paths import MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs


POSITIONS = ["GK", "DEF", "MID", "FWD"]
TRAIN_END = 2023
VALID_SEASON = 2024
TEST_SEASON = 2025
RANDOM_SEED = 42


@dataclass
class ModelBundle:
    features: list[str]
    models: dict[str, dict[str, object]]
    ensemble_weights: dict[str, float]


def xgb_candidates() -> list[dict]:
    return [
        {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.035, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 2.0},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.025, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 3.0},
        {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 4.0},
    ]


def make_xgb(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=1,
        eval_metric="rmse",
        **params,
    )


def make_rf(position: str) -> RandomForestRegressor:
    n_estimators = 220 if position in {"GK", "FWD"} else 180
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=18,
        min_samples_leaf=3,
        max_features=0.65,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def fit_pipeline(model, x_train, y_train, sample_weight=None) -> Pipeline:
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    pipe.fit(x_train, y_train, **kwargs)
    return pipe


def score_for_selection(y_true, pred) -> float:
    metrics = metrics_frame(y_true, pred).set_index("slice")
    overall = metrics.loc["overall", "rmse"]
    high = 0.0
    if "6-9" in metrics.index:
        high += metrics.loc["6-9", "rmse"]
    if "10+" in metrics.index:
        high += metrics.loc["10+", "rmse"] * 1.25
    return float(overall + 0.35 * high)


def train_position_models(df: pd.DataFrame, features: list[str], position: str) -> tuple[dict[str, object], dict]:
    pos_df = df[df["position"] == position].copy()
    train = pos_df[pos_df["season_order"] <= TRAIN_END]
    valid = pos_df[pos_df["season_order"] == VALID_SEASON]

    x_train, y_train = train[features], train["target_points"]
    x_valid, y_valid = valid[features], valid["target_points"]

    best_score = np.inf
    best_pipe = None
    best_meta = {}
    for high_multiplier in [0.0, 0.75, 1.25]:
        weights = range_weights(y_train, high_multiplier=high_multiplier)
        for params in xgb_candidates():
            pipe = fit_pipeline(make_xgb(params), x_train, y_train, weights)
            pred = np.clip(pipe.predict(x_valid), 0, 20)
            score = score_for_selection(y_valid, pred)
            if score < best_score:
                best_score = score
                best_pipe = pipe
                best_meta = {"model": "xgboost", "params": params, "high_multiplier": high_multiplier, "selection_score": best_score}

    rf_weights = range_weights(y_train, high_multiplier=0.75)
    rf_pipe = fit_pipeline(make_rf(position), x_train, y_train, rf_weights)

    xgb_pred = np.clip(best_pipe.predict(x_valid), 0, 20)
    rf_pred = np.clip(rf_pipe.predict(x_valid), 0, 20)
    candidates = {
        "xgb_only": (1.0, 0.0),
        "rf_only": (0.0, 1.0),
        "blend_75_25": (0.75, 0.25),
        "blend_60_40": (0.60, 0.40),
        "blend_50_50": (0.50, 0.50),
    }
    best_blend_name = None
    best_blend = None
    best_blend_score = np.inf
    for name, (wx, wr) in candidates.items():
        pred = wx * xgb_pred + wr * rf_pred
        score = score_for_selection(y_valid, pred)
        if score < best_blend_score:
            best_blend_score = score
            best_blend_name = name
            best_blend = (wx, wr)

    model_dict = {"xgboost": best_pipe, "random_forest": rf_pipe}
    meta = {
        "position": position,
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "xgboost": best_meta,
        "ensemble": {"name": best_blend_name, "xgboost_weight": best_blend[0], "random_forest_weight": best_blend[1], "selection_score": best_blend_score},
    }
    return model_dict, meta


def predict_bundle(bundle: ModelBundle, df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    for position, models in bundle.models.items():
        mask = df["position"] == position
        if not mask.any():
            continue
        weights = bundle.ensemble_weights[position]
        x = df.loc[mask, bundle.features]
        pred = weights["xgboost"] * models["xgboost"].predict(x) + weights["random_forest"] * models["random_forest"].predict(x)
        out[mask.to_numpy()] = np.clip(pred, 0, 20)
    return out


def train_all() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED_DIR / "model_features.csv", engine="python")
    features = feature_columns(df)
    keep_mask = df["season_order"].isin([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    df = df[keep_mask & df["position"].isin(POSITIONS)].copy()
    features = [col for col in features if df[col].notna().any()]

    models: dict[str, dict[str, object]] = {}
    weights: dict[str, dict[str, float]] = {}
    metadata = {"features": features, "positions": {}}
    for position in POSITIONS:
        pos_models, meta = train_position_models(df, features, position)
        models[position] = pos_models
        weights[position] = {
            "xgboost": meta["ensemble"]["xgboost_weight"],
            "random_forest": meta["ensemble"]["random_forest_weight"],
        }
        metadata["positions"][position] = meta
        print(f"{position}: {meta['ensemble']['name']} selected; train={meta['rows_train']:,}, valid={meta['rows_valid']:,}")

    bundle = ModelBundle(features=features, models=models, ensemble_weights=weights)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "position_ensemble.joblib")
    (MODELS_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    test = df[df["season_order"] == TEST_SEASON].copy()
    test["predicted_points"] = predict_bundle(bundle, test)
    eval_df = metrics_frame(test["target_points"], test["predicted_points"], test["position"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(REPORTS_DIR / "evaluation_2025_26.csv", index=False)
    print(eval_df.to_string(index=False))

    prediction_cols = [
        "player_id",
        "team_id",
        "opponent_team_id",
        "fixture_id",
        "season",
        "gameweek",
        "position",
        "was_home_int",
        "fixture_difficulty",
        "value",
        "predicted_points",
    ]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    test[prediction_cols].sort_values(["gameweek", "predicted_points"], ascending=[True, False]).to_csv(
        PREDICTIONS_DIR / "test_2025_26_predictions.csv", index=False
    )

    latest_gw = int(test["gameweek"].max())
    latest = test[test["gameweek"] == latest_gw].copy()
    latest[prediction_cols].sort_values("predicted_points", ascending=False).to_csv(
        PREDICTIONS_DIR / "final_predictions_latest_gameweek.csv", index=False
    )
    print(f"Wrote latest available prediction file for season 2025-26 GW{latest_gw}.")


if __name__ == "__main__":
    train_all()
