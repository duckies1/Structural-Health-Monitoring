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

# Color map
color_map = {
    "Healthy": "green",
    "Minor Risk": "yellow",
    "Moderate Risk": "orange",
    "Severe Risk": "red"
}

fig = go.Figure()

start = 0

for i in range(1, len(df)):
    if df["risk_level"].iloc[i] != df["risk_level"].iloc[i-1]:
        segment = df.iloc[start:i]

        fig.add_trace(go.Scatter(
            x=segment["timestamp"],
            y=segment["acc_mag"],
            mode="lines",
            line=dict(color=color_map.get(df["risk_level"].iloc[i-1], "black"), width=2),
            showlegend=False
        ))

        start = i

# Last segment
segment = df.iloc[start:]

fig.add_trace(go.Scatter(
    x=segment["timestamp"],
    y=segment["acc_mag"],
    mode="lines",
    line=dict(color=color_map.get(df["risk_level"].iloc[-1], "black"), width=2),
    showlegend=False
))

fig.update_layout(
    title="Acceleration Magnitude Over Time (Risk Colored)",
    xaxis_title="Time",
    yaxis_title="Acceleration Magnitude",
    template="plotly_dark"
)

fig.show()