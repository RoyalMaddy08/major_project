# Advanced Cyber-Physical Battery State Estimator System

A comprehensive cyber-physical battery evaluation and monitoring system combining high-fidelity electro-thermal simulations, traditional state observers, and data-driven Machine Learning (ML).

The project is structured into two modular, independent components that run in a synchronized loop:
1. [Simulator Server](file:///d:/_Deployed_Projects_Vercel/major_project/software/simulator/README.md): A standalone physics engine on Port 8000 that models cell behavior, noise, and faults.
2. [Visualiser Dashboard](file:///d:/_Deployed_Projects_Vercel/major_project/software/visualiser/README.md): A Flask-based interactive comparative dashboard on Port 5000 that runs EKF & ESN estimators.

---

## 🏗️ System Architecture & Connectivity

The system can run in two configurations:

### 1. Segregated Mode (Cyber-Physical Loop)
The standalone simulator handles physics, and the visualizer handles estimators and UI. Communication occurs via REST APIs and MongoDB.
```
 ┌──────────────────────────────┐              ┌──────────────────────────────┐
 │   Visualiser (Port 5000)     │  API status  │     Simulator (Port 8000)    │
 │   - EKF & ESN Estimators     ├─────────────►│   - 2RC ECM Physics Model    │
 │   - Glassmorphic Frontend    │◄─────────────┤   - Thermal & Aging dynamics │
 └──────────────┬───────────────┘  readings    └──────────────┬───────────────┘
                │                                             │
                └───────────────►  MongoDB  ◄─────────────────┘
                                 readings
```

### 2. Standalone / Serverless Mode
If the simulator server is offline, the Visualizer acts serverlessly. It loads historical parameters from the database, catches up on simulator ticks locally on-demand based on elapsed time, and runs the estimation pipeline in a stateless execution loop (100% compliant with Vercel serverless functions).

---

## 📂 System File Hierarchy

The repository is organized into distinct functional directories:

- **[visualiser/](file:///d:/_Deployed_Projects_Vercel/major_project/software/visualiser/)**: Interactive dashboard and estimator runner.
  - [app.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/visualiser/app.py): Flask visualizer backend.
  - [config.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/visualiser/config.py): Visualizer configs and ESN hyperparameters.
  - [training/](file:///d:/_Deployed_Projects_Vercel/major_project/software/visualiser/training/): ESN machine learning model trainer.
    - [train_rc.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/visualiser/training/train_rc.py): Ridge regression ESN training.
    - [feature_engineering.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/visualiser/training/feature_engineering.py): Shared online/offline feature extractor.
- **[simulator/](file:///d:/_Deployed_Projects_Vercel/major_project/software/simulator/)**: Standalone battery cell physics generator.
  - [app.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/simulator/app.py): Flask simulator server.
  - [battery_simulator.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/simulator/battery_simulator.py): 2RC electro-thermal physics.
  - [battery_chemistry.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/simulator/battery_chemistry.py): NMC, LFP, and Lead-Acid OCV profiles.
- **Shared Core Code** (replicated in both subdirectories to maintain modular independence):
  - [traditional_estimator.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/simulator/traditional_estimator.py): 2RC EKF observer and temperature-compensated SOH tracker.
  - [estimator_pipeline.py](file:///d:/_Deployed_Projects_Vercel/major_project/software/simulator/estimator_pipeline.py): Orchestrates EKF, CC, and ESN estimators with diagnostics.

---

## 🧮 Mathematical Foundations

The estimators utilize a hybrid of physics-informed models and reservoir-based data mappings:

### 1. Equivalent Circuit Model (ECM)
Cell dynamics are modeled using a 2RC branch circuit:
$$V_{\text{terminal}} = V_{\text{OCV}}(SOC) + I \cdot R_0 + V_1 + V_2$$
$$\frac{dV_i}{dt} = \frac{I - \frac{V_i}{R_i}}{C_i}$$

### 2. Extended Kalman Filter (EKF)
State predictions are dynamically corrected using noisy terminal voltage readings:
$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B} I_k$$
$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + R_{\text{meas}})^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (V_{\text{meas}, k} - V_{\text{pred}, k})$$

### 3. SOH & Temperature-Compensated Resistance Tracking
Ohmic resistance is estimated using a dual-mode observer compensated for temperature variations using the Arrhenius equation:
$$\text{temp\_effect} = \exp\left(1500.0 \cdot \left(\frac{1}{T_{\text{meas}} + 273.15} - \frac{1}{298.15}\right)\right)$$
- **Transient Observer**: Triggered on current step transients ($|\Delta I| > 0.2\text{ A}$):
  $$R_{0,\text{calc}} = \frac{|\Delta V|}{|\Delta I| \cdot \text{temp\_effect}}$$
- **Static Observer**: Triggered on steady-state loads ($|I| > 0.2\text{ A}$) after a 30-second convergence delay:
  $$R_{0,\text{static}} = \frac{|OCV(SOC) + V_1 + V_2 - V_t|}{|I| \cdot \text{temp\_effect}}$$
- **SOH Projection**: SOH capacity is mapped from resistance growth:
  $$SOH = 1.0 - \frac{(R_0 / R_{0,\text{nom}}) - 1.0}{1.5}$$

### 4. Echo State Network (ESN)
Input features are mapped into a high-dimensional recurrent reservoir, updating reservoir states:
$$\mathbf{x}(t) = (1 - \alpha)\mathbf{x}(t-1) + \alpha \tanh\left(\mathbf{W}_{\text{in}} [1; \mathbf{u}(t)] + \mathbf{W}_{\text{res}} \mathbf{x}(t-1)\right)$$
- **ESN SOH Hybridisation**: SOH changes very slowly compared to the simulation step size. To bridge the training time-resolution mismatch, SOH prediction is hybridized:
  $$\text{esn\_soh\_pred} = 0.02 \cdot \text{esn\_soh\_pred\_raw} + 0.98 \cdot \text{trad\_soh}$$
  This ensures the ESN Remaining Useful Life (RUL) cycle output tracks degradation accurately.

---

## 🚀 Running the System

1. **Install Requirements**:
   ```bash
   pip install -r software/visualiser/requirements.txt
   ```
2. **Start MongoDB** locally on Port 27017 (Optional; falls back to in-memory storage if offline).
3. **Run Simulator Server**:
   ```bash
   python software/simulator/app.py
   ```
4. **Run Visualizer comparisons**:
   ```bash
   python software/visualiser/app.py
   ```
   Navigate to `http://localhost:5000/` to open the evaluation dashboard.
