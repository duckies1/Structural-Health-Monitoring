import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 1️⃣ LOAD DATA
# =========================================================

df = pd.read_csv("synthetic_bridge_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Dataset Shape:", df.shape)

# =========================================================
# 2️⃣ FEATURE ENGINEERING
# =========================================================

df['acc_magnitude'] = np.sqrt(
    df['acceleration_x']**2 +
    df['acceleration_y']**2 +
    df['acceleration_z']**2
)

df['env_stress_index'] = (
    0.4 * df['temperature_c'] +
    0.3 * df['humidity_percent'] +
    0.3 * df['wind_speed_mps']
)

df['freq_energy_ratio'] = df['fft_magnitude'] / (df['fft_peak_freq'] + 1e-6)

df = df.sort_values("timestamp")
df['rolling_acc_energy'] = df.groupby('bridge_id')['acc_magnitude']\
    .rolling(window=5, min_periods=1)\
    .mean().reset_index(0, drop=True)

# =========================================================
# 3️⃣ ENVIRONMENTAL CLUSTERING (CONDITION AWARE STEP)
# =========================================================

env_features = ['temperature_c', 'humidity_percent', 'wind_speed_mps']
scaler_env = StandardScaler()
env_scaled = scaler_env.fit_transform(df[env_features])

kmeans = KMeans(n_clusters=4, random_state=42)
df['env_cluster'] = kmeans.fit_predict(env_scaled)

print("\nEnvironmental cluster distribution:")
print(df['env_cluster'].value_counts())

# =========================================================
# 4️⃣ ENVIRONMENTAL COMPENSATION
# =========================================================

for feature in ['acc_magnitude', 'fft_magnitude', 'rolling_acc_energy']:
    reg_env = LinearRegression()
    reg_env.fit(env_scaled, df[feature])
    predicted_env = reg_env.predict(env_scaled)
    df[feature + "_comp"] = df[feature] - predicted_env

# =========================================================
# 5️⃣ ENCODE TARGETS
# =========================================================

label_encoder = LabelEncoder()
df['damage_class_encoded'] = label_encoder.fit_transform(df['damage_class'])

# =========================================================
# 6️⃣ FEATURE SELECTION (DAMAGE ONLY + COMPENSATED)
# =========================================================

damage_features = [
    'acceleration_x','acceleration_y','acceleration_z',
    'acc_magnitude_comp',
    'fft_peak_freq',
    'fft_magnitude_comp',
    'freq_energy_ratio',
    'rolling_acc_energy_comp',
    'degradation_score'
]

X = df[damage_features]
y_class = df['damage_class_encoded']
y_reg = df['forecast_score_next_30d']

# =========================================================
# 7️⃣ TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42)

_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 8️⃣ DAMAGE CLASSIFICATION
# =========================================================

clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
clf.fit(X_train_scaled, y_class_train)

y_pred_class = clf.predict(X_test_scaled)

print("\n=== DAMAGE CLASSIFICATION REPORT ===")
print(classification_report(y_class_test, y_pred_class))

# =========================================================
# 9️⃣ FORECAST REGRESSION
# =========================================================

reg = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05)
reg.fit(X_train_scaled, y_reg_train)

y_pred_reg = reg.predict(X_test_scaled)

print("\n=== FORECAST REGRESSION ===")
print("MAE:", mean_absolute_error(y_reg_test, y_pred_reg))
print("R2:", r2_score(y_reg_test, y_pred_reg))

# =========================================================
# 🔟 CLUSTER-SPECIFIC ANOMALY DETECTION
# =========================================================

df['anomaly_flag'] = 0
df['anomaly_score'] = 0.0

print("\nRunning cluster-specific anomaly detection...")

for cluster_id in df['env_cluster'].unique():

    cluster_data = df[df['env_cluster'] == cluster_id]
    X_cluster = cluster_data[damage_features]

    scaler_cluster = StandardScaler()
    X_scaled_cluster = scaler_cluster.fit_transform(X_cluster)

    iso = IsolationForest(contamination=0.005, random_state=42)
    iso.fit(X_scaled_cluster)

    scores = iso.decision_function(X_scaled_cluster)
    preds = iso.predict(X_scaled_cluster)

    df.loc[df['env_cluster'] == cluster_id, 'anomaly_score'] = scores
    df.loc[df['env_cluster'] == cluster_id, 'anomaly_flag'] = preds

print("\nAnomaly distribution (cluster-aware):")
print(df['anomaly_flag'].value_counts())

# =========================================================
# 1️⃣4️⃣ COMPOSITE HEALTH INDEX
# =========================================================

print("\nComputing Composite Health Index...")

# --- Get damage probability (max class probability)
damage_proba = clf.predict_proba(scaler.transform(X))
df['damage_probability'] = damage_proba.max(axis=1)

# --- Normalize forecast score (0–1)
forecast_norm = (df['forecast_score_next_30d'] - df['forecast_score_next_30d'].min()) / \
                (df['forecast_score_next_30d'].max() - df['forecast_score_next_30d'].min())

# --- Normalize anomaly score (invert because lower = more abnormal)
anomaly_norm = (df['anomaly_score'] - df['anomaly_score'].min()) / \
               (df['anomaly_score'].max() - df['anomaly_score'].min())

anomaly_risk = 1 - anomaly_norm

# --- Weighted health index (0–100 scale)
df['health_index'] = (
    0.4 * df['damage_probability'] +
    0.35 * forecast_norm +
    0.25 * anomaly_risk
) * 100

# --- Risk Categorization
def categorize_health(score):
    if score < 30:
        return "Healthy"
    elif score < 60:
        return "Minor Risk"
    elif score < 80:
        return "Moderate Risk"
    else:
        return "Severe Risk"

df['risk_level'] = df['health_index'].apply(categorize_health)

print("\nRisk Distribution:")
print(df['risk_level'].value_counts())

# Save enriched dataset
df.to_csv("bridge_health_results.csv", index=False)

# =========================================================
# 1️⃣1️⃣ SHAP EXPLAINABILITY
# =========================================================

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_scaled)

print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png")
plt.close()

# =========================================================
# 1️⃣2️⃣ CONDITION DASHBOARD METRICS
# =========================================================

condition_summary = df.groupby(['bridge_id','env_cluster'])['degradation_score'].mean()

print("\nBridge Condition Summary (by environmental regime):")
print(condition_summary)

# =========================================================
# 1️⃣3️⃣ SAVE MODELS
# =========================================================

joblib.dump(clf, "damage_classifier.pkl")
joblib.dump(reg, "forecast_regressor.pkl")
# joblib.dump(scaler, "damage_scaler.pkl")
joblib.dump(kmeans, "env_cluster_model.pkl")
# joblib.dump(scaler_env, "env_scaler.pkl")

print("\nCluster-aware models saved successfully.")