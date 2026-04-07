import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_PATH = "training_data.csv"
MODEL_PATH = "model.joblib"

FEATURES = [
    "hour_of_day",
    "request_rate",
    "failed_attempts",
    "geo_risk_score",
    "device_trust_score",
    "sensitivity_level",
    "is_vpn",
    "is_tor",
]


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset shape: {df.shape}")

    X = df[FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )
    model.fit(X_scaled)

    scores = model.score_samples(X_scaled)

    print(f"\nAnomaly score range: min={scores.min():.4f}, max={scores.max():.4f}")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Std score:  {scores.std():.4f}")
    print(f"\nSample anomaly scores (first 10):")
    for i, s in enumerate(scores[:10]):
        print(f"  [{i}] score={s:.4f}")

    artifact = {
        "model": model,
        "scaler": scaler,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)
    test_score = loaded["model"].score_samples(
        loaded["scaler"].transform(X[:1])
    )
    print(f"Verification - score on first sample: {test_score[0]:.4f}")


if __name__ == "__main__":
    main()