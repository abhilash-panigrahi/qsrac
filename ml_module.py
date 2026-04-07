import os
import numpy as np
import joblib
import config

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

_model = None
_scaler = None


def _load_model():
    global _model, _scaler
    if _model is None:
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            _model = artifact["model"]
            _scaler = artifact.get("scaler", None)
        else:
            _model = artifact
            _scaler = None


def _extract_features(context_dict: dict) -> np.ndarray:
    features = [
        float(context_dict.get("hour_of_day", 12)),
        float(context_dict.get("request_rate", 1.0)),
        float(context_dict.get("failed_attempts", 0)),
        float(context_dict.get("geo_risk_score", 0.0)),
        float(context_dict.get("device_trust_score", 1.0)),
        float(context_dict.get("sensitivity_level", 1.0)),
        float(context_dict.get("is_vpn", 0)),
        float(context_dict.get("is_tor", 0)),
    ]
    return np.array(features).reshape(1, -1)


def _map_score_to_risk(score: float) -> str:
    if score >= config.RISK_THRESHOLD_LOW:
        return "Low"
    elif score >= config.RISK_THRESHOLD_MEDIUM:
        return "Medium"
    elif score >= config.RISK_THRESHOLD_HIGH:
        return "High"
    else:
        return "Critical"


def get_risk_score(context_dict: dict) -> str:
    _load_model()

    features = _extract_features(context_dict)

    if _scaler is not None:
        features = _scaler.transform(features)

    score = _model.score_samples(features)[0]

    return _map_score_to_risk(score)