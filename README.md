# 🏗️ Condition-Aware AI for Structural Health Monitoring (SHM)

## 📌 Overview

This project implements a **condition-aware AI framework for Structural Health Monitoring (SHM)** inspired by the paper:

> _“Condition-aware AI framework for automated structural health monitoring”_

The goal is to build a system that:

- Monitors structural health of bridges
- Detects anomalies
- Classifies damage severity
- Reduces false alarms caused by environmental variations

---

## 🧠 Core Problem

Traditional SHM systems assume:

> Any deviation from normal = damage

However, in real-world scenarios:

- Temperature changes
- Wind loads
- Humidity variations

can significantly alter sensor readings **without actual damage**.

This leads to:

❌ False positives  
❌ Unreliable monitoring systems

---

## 💡 Key Idea from the Paper

The paper introduces:

> **Condition-Aware Modeling**

Instead of learning **p(x | healthy)**,  
we learn **p(x | healthy, environment)**.

### 🔥 Insight:

Separate:

- Environmental effects (E)
- Structural damage effects (D)

So that:

**x = f(E) + g(D)**

---

## 🚀 Our Approach

Since real-world long-term SHM datasets are limited, we:

### 1️⃣ Built a Physics-Informed Synthetic Data Generator

### 2️⃣ Used Deep Learning (LSTM) to model temporal dynamics

### 3️⃣ Explicitly modeled environmental + structural effects

---

## 🧪 Dataset

Original dataset:

- Duration: 14 days
- Frequency: 15-minute intervals
- Features:
  - Acceleration (x, y, z)
  - Temperature
  - Humidity
  - Wind speed
  - FFT features
  - Structural labels

---

## ⚠️ Challenge

The dataset:

- Does NOT contain long-term degradation
- Does NOT include seasonal variations

So we cannot directly train models for:

- Year-long monitoring
- Degradation prediction

---

## 🏗️ Synthetic Data Generation Strategy

We created a **1-year synthetic dataset** using:

---

### 🔹 1. LSTM-Based Temporal Modeling

- Learns short-term dynamics: **Xₜ → Xₜ₊₁**
- Captures:
  - Sensor correlations
  - Time dependencies

---

### 🔹 2. Per Bridge-Sensor Modeling

Instead of mixing data:

- (B001, S1) → separate sequence
- (B002, S3) → separate sequence

✔ Preserves temporal consistency  
✔ Avoids cross-entity contamination

---

### 🔹 3. Physics-Informed Constraints

We embedded domain knowledge:

- 🌡 Temperature ↑ → Frequency ↓
- 💧 Humidity ↑ → Slight damping (frequency ↓)
- 🌬 Wind ↑ → Vibration amplitude ↑

These relationships are enforced via additional loss terms.

---

### 🔹 4. Stress-Driven Degradation Model

Instead of random degradation:

**Dₜ₊₁ = Dₜ + f(stress)**

Where stress depends on:

- Wind
- Vibration magnitude

✔ Physically meaningful  
✔ Environment-dependent  
✔ Gradual evolution

---

### 🔹 5. Structural Labels Generation

Derived from degradation:

| Degradation (D) | Condition | Class     |
| --------------- | --------- | --------- |
| 0 – 0.2         | 0         | No Damage |
| 0.2 – 0.4       | 1         | Minor     |
| 0.4 – 0.7       | 2         | Moderate  |
| 0.7 – 1         | 3         | Severe    |

---

### 🔹 6. Forecast Score (Optional)

We model a future risk score based on:

- Current degradation
- Environmental stress
- Temperature variability

Used for predictive maintenance (not required for core SHM).

---

## 🧠 Model Architecture

### LSTM Generator

- Input: Past 6 hours (window = 24)
- Output: Next timestep
- Multi-feature prediction

---

### Loss Function

Loss =  
MSE Loss

- Temperature–Frequency constraint
- Wind–Magnitude constraint
- Smoothness constraint

---

## 📊 Final Synthetic Dataset

Contains:

- timestamp
- bridge_id
- sensor_id
- acceleration_x / y / z
- temperature_c
- humidity_percent
- wind_speed_mps
- fft_peak_freq
- fft_magnitude
- degradation_score
- structural_condition
- damage_class
- forecast_score_next_30d

---

## 🎯 Why This Works

✔ Environmental variation dominates → realistic noise  
✔ Damage is subtle → realistic degradation  
✔ Condition-aware models outperform naive ones

---

## 🔬 Key Inferences

1. Environmental effects can mask damage
2. Condition-aware modeling reduces false positives
3. Physics + ML > Pure ML
4. Synthetic data must be controlled

---

## 🧪 Use Cases

- SHM system benchmarking
- Anomaly detection testing
- Condition-aware AI validation
- Predictive maintenance experiments

---

## 🚀 Future Improvements

- Multiple bridge types
- Sudden damage events
- Graph Neural Networks
- Transformer-based models
- Learned degradation dynamics

---

## 🏁 Conclusion

This project demonstrates:

> How to build a **realistic, physics-informed, condition-aware SHM system**

by combining:

- Domain knowledge
- Deep learning
- Synthetic data engineering

---

## 📚 Reference

Condition-aware AI framework for automated structural health monitoring
