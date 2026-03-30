import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import json

# ===============================
# Load models
# ===============================

clf = joblib.load("damage_classifier.pkl")
reg = joblib.load("forecast_regressor.pkl")
damage_scaler = joblib.load("damage_scaler.pkl")
kmeans = joblib.load("env_cluster_model.pkl")
env_scaler = joblib.load("env_scaler.pkl")

# ===============================
# Helper Functions
# ===============================

def compute_features(data):

    acc_magnitude = np.sqrt(
        data["acceleration_x"]**2 +
        data["acceleration_y"]**2 +
        data["acceleration_z"]**2
    )

    freq_energy_ratio = data["fft_magnitude"] / (data["fft_peak_freq"] + 1e-6)

    return acc_magnitude, freq_energy_ratio


def categorize_health(score):
    if score < 30:
        return "Healthy"
    elif score < 60:
        return "Minor Risk"
    elif score < 80:
        return "Moderate Risk"
    else:
        return "Severe Risk"


# ===============================
# Main Prediction Function
# ===============================

def predict_health(input_json):

    data = input_json

    # --- Environmental clustering
    env_features = np.array([[ 
        data["temperature_c"],
        data["humidity_percent"],
        data["wind_speed_mps"]
    ]])

    env_scaled = env_scaler.transform(env_features)
    env_cluster = int(kmeans.predict(env_scaled)[0])

    # --- Feature engineering
    acc_magnitude, freq_energy_ratio = compute_features(data)

    damage_features = np.array([[
        data["acceleration_x"],
        data["acceleration_y"],
        data["acceleration_z"],
        acc_magnitude,
        data["fft_peak_freq"],
        data["fft_magnitude"],
        freq_energy_ratio,
        acc_magnitude,  # rolling approx placeholder
        data["degradation_score"]
    ]])

    damage_scaled = damage_scaler.transform(damage_features)

    # --- Predictions
    damage_proba = clf.predict_proba(damage_scaled).max()
    forecast_score = reg.predict(damage_scaled)[0]

    # --- Simple anomaly proxy (distance from mean cluster center)
    anomaly_score = np.linalg.norm(env_scaled - kmeans.cluster_centers_[env_cluster])
    anomaly_risk = min(anomaly_score / 10, 1)

    # --- Normalize forecast roughly
    forecast_norm = min(forecast_score / 100, 1)

    # --- Composite Health Index
    health_index = (
        0.4 * damage_proba +
        0.35 * forecast_norm +
        0.25 * anomaly_risk
    ) * 100

    risk_level = categorize_health(health_index)

    return {
        "environment_cluster": env_cluster,
        "damage_probability": round(float(damage_proba), 3),
        "forecast_score_30d": round(float(forecast_score), 2),
        "health_index": round(float(health_index), 2),
        "risk_level": risk_level
    }


# ===============================
# Example Usage
# ===============================

if __name__ == "__main__":

    sample_input = {
        "acceleration_x": -0.2,
        "acceleration_y": 0.1,
        "acceleration_z": -0.3,
        "temperature_c": 28,
        "humidity_percent": 75,
        "wind_speed_mps": 5,
        "fft_peak_freq": 2.5,
        "fft_magnitude": 1.1,
        "degradation_score": 60
    }

    result = predict_health(sample_input)
    print(json.dumps(result, indent=4))