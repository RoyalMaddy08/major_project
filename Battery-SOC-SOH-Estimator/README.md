# Battery Estimator Comparison Dashboard (EKF+CC vs ML-ESN)

An end-to-end Python ML and Flask-based Battery Management System (BMS) dashboard. The system simulates battery cell physics, saves real-time telemetry to MongoDB, and compares **Traditional BMS State Estimators (Extended Kalman Filter + Coulomb Counting)** side-by-side against **Modern Data-Driven Machine Learning (Reservoir Computing - ESN)** models in real-time.

---

## System Overview

Traditional battery state estimation relies on Kalman Filtering, which requires complex physical parameterization and is computationally heavy for real-time edge microcontrollers. This project implements **Reservoir Computing (RC)** — specifically an **Echo State Network (ESN)** — and compares it against a standard **Extended Kalman Filter (EKF)** and dynamic resistance-based SOH estimator, tracking how they perform relative to the physical ground truth under different vehicle drive cycles.

### Core Components
1. **Datasets (`datasets/`)**:
   - `training_ev_battery_dataset_multiclass.csv`: Split training dataset containing `Time, Voltage, Current, Temperature, SOC, SOH`.
   - `testing_ev_battery_dataset_multiclass.csv`: Split testing dataset for verification.
2. **ESN Training Module (`training/train_rc.py`)**: Implements an Echo State Network in pure NumPy and trains the readout weights via Ridge Regression.
3. **Battery Physics Simulator (`simulator/battery_simulator.py`)**: Implements a first-order Equivalent Circuit Model (ECM) with SOH degradation, dynamic thermal heating/cooling, and sensor noise simulation.
4. **Traditional Estimator Module (`simulator/traditional_estimator.py`)**: Implements a 2-state EKF (states: SOC, $V_1$) and a dynamic resistance SOH tracker.
5. **Flask Backend Server (`app.py`)**: Implements an on-demand catch-up loop (`sync_simulator`) based on elapsed time (making the app 100% serverless/Vercel compliant) that steps both the physics simulator and traditional estimators, logs readings to MongoDB, and runs ESN model inference.
6. **Interactive Dashboard (`templates/index.html`, `static/css/style.css`, `static/js/dashboard.js`)**: A premium, glassmorphic dark-theme UI featuring real-time Chart.js plots comparing True, EKF+CC, and ML-ESN curves side-by-side, with controls to configure simulation drive cycles and toggle accelerated degradation.

---

## Theoretical Foundation

### 1. Battery Physics Model
The simulator models a 3S (3 cells in series) Lithium-ion NMC battery pack using the following dynamics:
* **Terminal Voltage ($V_{\text{terminal}}$)**: 
  $$V_{\text{terminal}} = V_{\text{OCV}}(SOC) + I \cdot R_0 + V_1$$
  Where $V_{\text{OCV}}$ is interpolated from an empirical Open Circuit Voltage table, $R_0$ is the internal ohmic resistance, and $V_1$ is the voltage drop across the polarization RC branch ($R_1 \parallel C_1$).
* **Polarization Voltage Dynamics ($V_1$)**:
  $$\frac{dV_1}{dt} = \frac{I - \frac{V_1}{R_1}}{C_1}$$
* **Thermal Dynamics ($T$)**:
  $$\frac{dT}{dt} = \frac{(I^2 \cdot R_0 + |I \cdot V_1|) - h \cdot (T - T_{\text{ambient}})}{C_{\text{thermal}}}$$
* **Degradation Model (SOH capacity fade)**:
  $$\Delta SOH = -1.2 \times 10^{-7} \cdot |I|^{1.3} \cdot e^{0.06(T-25)} \cdot dt$$

### 2. Reservoir Computing (Echo State Network)
An ESN consists of a fixed, randomly initialized high-dimensional recurrent reservoir that projects input signals into a computational space:
* **Reservoir State Update**:
  $$\mathbf{x}(t) = (1 - \alpha)\mathbf{x}(t-1) + \alpha \tanh\left(\mathbf{W}_{\text{in}} [1; \mathbf{u}(t)] + \mathbf{W}_{\text{res}} \mathbf{x}(t-1)\right)$$
  Where $\mathbf{u}(t) = [Voltage, Current, Temperature]^T$, $\mathbf{W}_{\text{in}}$ is the input projection matrix, $\mathbf{W}_{\text{res}}$ is the recurrent reservoir weight matrix scaled to a spectral radius $\rho(\mathbf{W}_{\text{res}}) < 1$, and $\alpha$ is the leak rate.
* **Output Readout**:
  $$\mathbf{y}(t) = \mathbf{W}_{\text{out}} [1; \mathbf{u}(t); \mathbf{x}(t)]$$
  Where $\mathbf{y}(t) = [SOC, SOH]^T$. Readout weights $\mathbf{W}_{\text{out}}$ are optimized using regularized linear regression (Ridge Regression):
  $$\mathbf{W}_{\text{out}} = \mathbf{Y} \mathbf{X}^T (\mathbf{X} \mathbf{X}^T + \lambda \mathbf{I})^{-1}$$

---

## Setup & Running Instructions

### Prerequisites
- Python 3.8+
- MongoDB Community Server running locally on default port `27017` (If MongoDB is not running, the application automatically falls back to an in-memory data store).

### Step 1: Install Dependencies
Install required packages using pip:
```bash
pip install -r requirements.txt
```

### Step 2: Train the ESN Model
Train the Reservoir Computing model readout weights on the training dataset split:
```bash
python training/train_rc.py
```
This will train the ESN on `datasets/training_ev_battery_dataset_multiclass.csv` and export the model package to the root folder as `model_rc.pkl`.

### Step 3: Launch the BMS Dashboard
Run the Flask server:
```bash
python app.py
```

Open your web browser and navigate to:
```
http://localhost:5000/
```

---

## Dashboard Usage
- **Simulator Controls**: Start, pause, or reset the simulator. Resetting clears logs in MongoDB and resets EKF/SOH estimator parameters.
- **Drive Cycle Picker**: Swap between standard cycles:
  - **UDDS**: Urban stop-and-go driving with regenerative charge spikes.
  - **HWFET**: Constant, high-draw highway cruising.
  - **US06**: Aggressive, highly dynamic power acceleration spikes.
  - **Constant Discharge**: Steady continuous 1C draw.
  - **CCCV Charge**: Constant current charging tapering off into constant voltage.
- **Accelerated SOH Aging**: Toggle this switch to accelerate capacity degradation by 1500x. Observe SOH dropping and internal resistance growing.
- **Comparison Charts**: Compare real-time EKF+CC predictions (dashed orange line) and ESN ML predictions (solid blue line) against physical true values (solid green line).
