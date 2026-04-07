"""
plot_scores.py — Visualizes the anomaly score separation for QSRAC.

This script regenerates the synthetic dataset and plots a histogram 
comparing Normal vs. Attack score distributions.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
RANDOM_SEED = 42
N_NORMAL = 800
N_ATTACK = 200

# ── Data Generation (Mirrors experiment_harness.py) ───────────────────────────

def generate_normal_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    hour_of_day      = rng.integers(0, 24, n).astype(float)
    request_rate     = rng.uniform(0.5, 2.0, n)
    failed_attempts  = rng.integers(0, 3, n).astype(float)
    geo_risk_score   = rng.uniform(0.0, 0.3, n)
    device_trust     = rng.uniform(0.7, 1.0, n)
    sensitivity      = rng.integers(1, 4, n).astype(float)
    is_vpn           = rng.choice([0.0, 1.0], n, p=[0.9, 0.1])
    is_tor           = np.zeros(n)

    return np.column_stack([
        hour_of_day, request_rate, failed_attempts, geo_risk_score,
        device_trust, sensitivity, is_vpn, is_tor,
    ])

def generate_attack_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    hour_of_day      = rng.integers(0, 24, n).astype(float)
    request_rate     = rng.uniform(5.0, 20.0, n)
    failed_attempts  = rng.integers(3, 11, n).astype(float)
    geo_risk_score   = rng.uniform(0.6, 1.0, n)
    device_trust     = rng.uniform(0.0, 0.4, n)
    sensitivity      = rng.integers(3, 6, n).astype(float)
    is_vpn           = rng.choice([0.0, 1.0], n, p=[0.2, 0.8])
    is_tor           = rng.choice([0.0, 1.0], n, p=[0.3, 0.7])

    return np.column_stack([
        hour_of_day, request_rate, failed_attempts, geo_risk_score,
        device_trust, sensitivity, is_vpn, is_tor,
    ])

# ── Main Execution ─────────────────────────────────────────────────────────────

def main():
    # 1. Load Model and Scaler
    print(f"Loading model from: {MODEL_PATH}")
    try:
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            model = artifact["model"]
            scaler = artifact.get("scaler", None)
        else:
            model = artifact
            scaler = None
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return

    # 2. Regenerate Data
    rng = np.random.default_rng(RANDOM_SEED)
    X_normal = generate_normal_samples(N_NORMAL, rng)
    X_attack = generate_attack_samples(N_ATTACK, rng)

    # 3. Compute Scores
    if scaler:
        X_normal = scaler.transform(X_normal)
        X_attack = scaler.transform(X_attack)
    
    normal_scores = model.score_samples(X_normal)
    attack_scores = model.score_samples(X_attack)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Histogram settings
    bins = 50
    plt.hist(normal_scores, bins=bins, alpha=0.6, label='Normal Traffic', color='skyblue', edgecolor='navy')
    plt.hist(attack_scores, bins=bins, alpha=0.6, label='Attack Traffic', color='salmon', edgecolor='darkred')

    # Formatting
    plt.title('QSRAC Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Isolation Forest Score (Higher = More Normal)', fontsize=12)
    plt.ylabel('Frequency (Sample Count)', fontsize=12)
    
    # Visual cues for thresholds (based on your ml_module.py)
    plt.axvline(x=-0.45, color='orange', linestyle='--', label='Low/Medium Threshold (-0.45)')
    plt.axvline(x=-0.65, color='red', linestyle='--', label='High/Critical Threshold (-0.65)')

    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # Save and Show
    output_plot = "score_distribution.png"
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()