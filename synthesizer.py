import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# =============================
# CONFIG
# =============================
WINDOW_SIZE = 24
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 50
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES = [
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "temperature_c",
    "humidity_percent",
    "wind_speed_mps",
    "fft_peak_freq",
    "fft_magnitude"
]

TEMP_IDX = FEATURES.index("temperature_c")
FREQ_IDX = FEATURES.index("fft_peak_freq")
HUM_IDX = FEATURES.index("humidity_percent")
WIND_IDX = FEATURES.index("wind_speed_mps")
MAG_IDX = FEATURES.index("fft_magnitude")

LAMBDA_TEMP = 0.1
LAMBDA_WIND = 0.1
LAMBDA_SMOOTH = 0.05

# =============================
# Dataset Class
# =============================
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# =============================
# Model
# =============================
class LSTMGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =============================
# Load Data
# =============================
df = pd.read_csv("bridge_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

groups = df.groupby(["bridge_id", "sensor_id"])

all_sequences = []
scalers = {}

for (bridge, sensor), group in groups:
    group = group.sort_values("timestamp")
    data = group[FEATURES].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    scalers[(bridge, sensor)] = scaler

    for i in range(len(data_scaled) - WINDOW_SIZE):
        x = data_scaled[i:i+WINDOW_SIZE]
        y = data_scaled[i+WINDOW_SIZE]
        all_sequences.append((x, y))

dataset = SequenceDataset(all_sequences)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================
# Train Model
# =============================
model = LSTMGenerator(len(FEATURES)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)

        mse_loss = criterion(pred, y)

        # Physics: Temp + Humidity → Frequency
        temp = pred[:, TEMP_IDX]
        hum = pred[:, HUM_IDX]
        freq = pred[:, FREQ_IDX]

        freq_physics = -0.01 * temp - 0.003 * hum + 0.5
        temp_loss = torch.mean((freq - freq_physics) ** 2)

        # Physics: Wind → Magnitude
        wind = pred[:, WIND_IDX]
        mag = pred[:, MAG_IDX]
        mag_physics = 0.05 * wind
        wind_loss = torch.mean((mag - mag_physics) ** 2)

        # Smoothness
        smooth_loss = torch.mean((pred - x[:, -1, :]) ** 2)

        loss = (
            mse_loss
            + LAMBDA_TEMP * temp_loss
            + LAMBDA_WIND * wind_loss
            + LAMBDA_SMOOTH * smooth_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.6f}")

# =============================
# Generate 1-Year Data
# =============================
synthetic_rows = []

steps = 365 * 24 * 4

for (bridge, sensor), group in groups:

    group = group.sort_values("timestamp")
    data = group[FEATURES].values
    scaler = scalers[(bridge, sensor)]
    data_scaled = scaler.transform(data)

    seed_seq = data_scaled[:WINDOW_SIZE]
    current_seq = seed_seq.copy()

    D = 0.05  # degradation
    damage_start = int(0.6 * steps)

    stress_history = []
    temp_history = []

    for t in range(steps):

        inp = torch.FloatTensor(current_seq[-WINDOW_SIZE:]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            next_val = model(inp).cpu().numpy()[0]

        # =============================
        # Stress-Driven Degradation
        # =============================
        stress = (
            0.5 * next_val[WIND_IDX] +
            0.3 * abs(next_val[MAG_IDX])
        )

        stress_history.append(stress)
        temp_history.append(next_val[TEMP_IDX])

        if t > damage_start:
            D += 0.00005 * stress

        D = min(D, 1.0)

        # Frequency reduces with degradation
        next_val[FREQ_IDX] *= (1 - 0.08 * D)

        current_seq = np.vstack([current_seq, next_val])

        real_val = scaler.inverse_transform([next_val])[0]

        # =============================
        # Structural Condition Mapping
        # =============================
        if D < 0.2:
            condition = 0
            label = "No Damage"
        elif D < 0.4:
            condition = 1
            label = "Minor"
        elif D < 0.7:
            condition = 2
            label = "Moderate"
        else:
            condition = 3
            label = "Severe"

        # -----------------------------
        # Forecast Score (Next 30 Days Risk Proxy)
        # -----------------------------
        window_30d = 30 * 24 * 4

        recent_stress = np.mean(stress_history[-window_30d:]) if len(stress_history) > 0 else 0
        temp_variability = np.std(temp_history[-window_30d:]) if len(temp_history) > 0 else 0

        forecast_score = (
            50 * D
            + 20 * recent_stress
            + 5 * temp_variability
        )

        forecast_score = float(np.clip(forecast_score, 0, 100))

        synthetic_rows.append({
            "timestamp": pd.Timestamp("2020-01-15") + pd.Timedelta(minutes=15*t),
            "bridge_id": bridge,
            "sensor_id": sensor,
            **dict(zip(FEATURES, real_val)),
            "degradation_score": D,
            "structural_condition": condition,
            "damage_class": label,
            "forecast_score_next_30d": forecast_score
        })

# =============================
# Save
# =============================
synthetic_df = pd.DataFrame(synthetic_rows)
synthetic_df.to_csv("synthetic_bridge_data.csv", index=False)

print("Synthetic multi-bridge dataset generated successfully.")