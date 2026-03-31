import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("bridge_health_results.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Acc magnitude
df["acc_mag"] = np.sqrt(
    df["acceleration_x"]**2 +
    df["acceleration_y"]**2 +
    df["acceleration_z"]**2
)

# Color mapping
color_map = {
    0: "green",
    1: "yellow",
    2: "orange",
    3: "red"
}

plt.figure(figsize=(15, 5))

for condition in df["risk_level"].unique():
    subset = df[df["risk_level"] == condition]
    plt.scatter(
        subset["timestamp"],
        subset["acc_mag"],
        color=color_map[condition],
        s=5,
        label=f"Condition {condition}"
    )

plt.title("Acceleration Magnitude with Risk Levels")
plt.xlabel("Time")
plt.ylabel("Acceleration Magnitude")
plt.legend()
plt.show()