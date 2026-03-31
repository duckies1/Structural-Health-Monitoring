import pandas as pd
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("bridge_health_results.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Acc magnitude
df["acc_mag"] = np.sqrt(
    df["acceleration_x"]**2 +
    df["acceleration_y"]**2 +
    df["acceleration_z"]**2
)

df = df.sort_values("timestamp").reset_index(drop=True)

# Smooth (VERY IMPORTANT)
df["acc_mag_smooth"] = df["acc_mag"].rolling(1000).mean()

# Create segments only when risk changes
df["segment"] = (df["risk_level"] != df["risk_level"].shift()).cumsum()

color_map = {
    "Healthy": "green",
    "Minor Risk": "yellow",
    "Moderate Risk": "orange",
    "Severe Risk": "red"
}

fig = go.Figure()

for _, segment in df.groupby("segment"):
    risk = segment["risk_level"].iloc[0]

    fig.add_trace(go.Scatter(
        x=segment["timestamp"],
        y=segment["acc_mag_smooth"],
        mode="lines",
        line=dict(color=color_map.get(risk, "white"), width=3),
        name=risk,
        hovertemplate=(
            "Time: %{x}<br>" +
            "Acc: %{y:.3f}<br>" +
            f"Risk: {risk}"
        )
    ))

fig.update_layout(
    title="Acceleration Magnitude Over Time (Clean Risk Visualization)",
    xaxis_title="Time",
    yaxis_title="Acceleration Magnitude",
    template="plotly_dark"
)

fig.show()