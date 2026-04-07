import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

COLUMNS = [
    "hour_of_day",
    "request_rate",
    "failed_attempts",
    "geo_risk_score",
    "device_trust_score",
    "sensitivity_level",
    "is_vpn",
    "is_tor",
]

N_NORMAL = 800
N_ANOMALOUS = 200


def generate_normal_samples(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "hour_of_day": np.random.randint(0, 24, n),
        "request_rate": np.random.uniform(0.5, 2.0, n),
        "failed_attempts": np.random.randint(0, 3, n),
        "geo_risk_score": np.random.uniform(0.0, 0.3, n),
        "device_trust_score": np.random.uniform(0.7, 1.0, n),
        "sensitivity_level": np.random.randint(1, 4, n),
        "is_vpn": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "is_tor": np.zeros(n, dtype=int),
    })


def generate_anomalous_samples(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "hour_of_day": np.random.randint(0, 24, n),
        "request_rate": np.random.uniform(5.0, 20.0, n),
        "failed_attempts": np.random.randint(3, 11, n),
        "geo_risk_score": np.random.uniform(0.6, 1.0, n),
        "device_trust_score": np.random.uniform(0.0, 0.4, n),
        "sensitivity_level": np.random.randint(3, 6, n),
        "is_vpn": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "is_tor": np.random.choice([0, 1], n, p=[0.3, 0.7]),
    })


def main():
    normal = generate_normal_samples(N_NORMAL)
    anomalous = generate_anomalous_samples(N_ANOMALOUS)

    dataset = pd.concat([normal, anomalous], ignore_index=True)
    dataset = dataset[COLUMNS]
    dataset = dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    dataset["hour_of_day"] = dataset["hour_of_day"].astype(int)
    dataset["failed_attempts"] = dataset["failed_attempts"].astype(int)
    dataset["sensitivity_level"] = dataset["sensitivity_level"].astype(int)
    dataset["is_vpn"] = dataset["is_vpn"].astype(int)
    dataset["is_tor"] = dataset["is_tor"].astype(int)

    dataset.to_csv("training_data.csv", index=False)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Normal samples: {N_NORMAL}")
    print(f"Anomalous samples: {N_ANOMALOUS}")
    print("\nSample rows:")
    print(dataset.head(10).to_string(index=False))
    print("\nColumn dtypes:")
    print(dataset.dtypes)


if __name__ == "__main__":
    main()