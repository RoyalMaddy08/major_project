# STM32 Edge Battery State Classifier (Hardware Component)

This directory contains the embedded C firmware, offline Python training pipelines, and data generation assets to run a real-time, optimized **Echo State Network (ESN)** state classifier on an edge STM32 microcontroller (ARM Cortex-M core).

---

## 📂 Hardware Directory File Guide

* **[main.c](file:///d:/_Deployed_Projects_Vercel/major_project/hardware/main.c)**: STM32 microcontroller firmware entry point. It runs the ESN inference loop on static test telemetry data, outputs classifications (Normal/Warning/Critical) over USART, and switches warning LEDs.
* **[train_classifier.py](file:///d:/_Deployed_Projects_Vercel/major_project/hardware/train_classifier.py)**: Offline Python training script that loads raw EV drive cycle data, trains the ESN readout matrix, converts the sparse reservoir to **Compressed Sparse Row (CSR)** arrays, and outputs the C header `esn_classifier_weights.h`.
* **[export_weights.py](file:///d:/_Deployed_Projects_Vercel/major_project/hardware/export_weights.py)**: Helper script to export the trained dashboard visualizer weights (`model_rc.pkl`) into C header formats (`esn_estimator_weights.h`) for microcontroller porting.
* **[train.py](file:///d:/_Deployed_Projects_Vercel/major_project/hardware/train.py)**: Core Python class implementing the sparse reservoir computing initialization, Ridge Regression fitting, and quantization simulation.
* **[data_set.m](file:///d:/_Deployed_Projects_Vercel/major_project/hardware/data_set.m)**: MATLAB script that models simple EV cell voltage, current, and thermal convective dynamics to generate synthetic battery dataset files.
* **[original_ev_battery_dataset_multiclass.csv](file:///d:/_Deployed_Projects_Vercel/major_project/hardware/original_ev_battery_dataset_multiclass.csv)**: Pre-generated synthetic EV dataset containing time-series columns for Voltage, Current, Temperature, and multiclass state labels.

---

## 🧮 Embedded ESN Architecture & Dynamics

The firmware implements a recurrent reservoir state update followed by a linear readout classification:

### 1. Feature Normalization
Input features `[Voltage, Current, Temperature]` ($u$) are scaled using pre-calculated training dataset averages to handle variations in magnitude:
$$u_{\text{scaled}, i} = \frac{u_i - \text{means}_i}{\text{stds}_i}$$

### 2. Sparse Reservoir Update
Project normalized features into a 50-node recurrent state reservoir ($x$). The state transition equation at tick $t$ is:
$$\tilde{x}_t = \tanh\left(\mathbf{W}_{\text{in}} [1; u_{\text{scaled}}] + \mathbf{W}_{\text{res}} x_{t-1}\right)$$
$$x_t = (1 - \alpha) x_{t-1} + \alpha \tilde{x}_t$$
Where:
* $\alpha$ is the leak rate (configured to `0.3`).
* $[1; u_{\text{scaled}}]$ is the input vector with a bias term.
* $\mathbf{W}_{\text{res}}$ is the sparse recurrent weight matrix ($50 \times 50$, initialized with $85\%$ sparsity).

### 3. Readout Classification
Compute output classes $y$ using the trained linear readout matrix:
$$y_t = \mathbf{W}_{\text{out}} [1; u_{\text{scaled}}; x_t]$$
The final predicted state (Normal, Warning, or Critical) is determined via:
$$\text{state} = \operatorname{argmax}(y_{t, 0}, y_{t, 1}, y_{t, 2})$$

---

## ⚡ Embedded Optimizations

### 1. Compressed Sparse Row (CSR) SpMV
Because the reservoir weight matrix $\mathbf{W}_{\text{res}}$ is $85\%$ sparse, the standard matrix-vector multiplication is optimized into a Compressed Sparse Row (CSR) format:
* Only the **non-zero elements** are stored in Flash memory (`esn_W_res_val`).
* Columns are indexed by `esn_W_res_col`, and row pointers are tracked by `esn_W_res_row_ptr`.
* This reduces calculations from $2,500$ multiplications down to only $375$, providing a **$6.7\times$ speedup** in execution speed and saving ~10 KB of Flash space.

### 2. Dual Q15 Fixed-Point Mode
For low-power microcontrollers lacking a floating-point unit (FPU), the firmware supports a fixed-point path. Toggling `#define ESN_FIXED_POINT 1` in `main.c` compiles the ESN update to run entirely in fixed-point math:
* Inputs are scaled to Q12 format ($\pm 8.0$ dynamic range).
* Weights and reservoir states are tracked in Q15 format ($[-1.0, 1.0)$).
* Tanh activation is approximated via a high-speed Q15 activation mapping.
* Only the final readout prediction (3 channels) is cast back to float to maintain high classification boundary accuracy.

---

## 🚀 Quick Start Guide

### Step 1: Retrain the Classifier
To train the classifier on the raw EV dataset and regenerate the weights header:
```bash
python train_classifier.py
```
This writes the sparse/dense weights directly to `esn_classifier_weights.h`.

### Step 2: Build & Flash STM32
1. Open the project inside your STM32 compiler toolchain (e.g. STM32CubeIDE).
2. Ensure `esn_classifier_weights.h` is present in your compiler's include search paths.
3. Configure the execution mode in `main.c` line 50:
   * `#define ESN_FIXED_POINT 0` for high-precision standard float math.
   * `#define ESN_FIXED_POINT 1` for integer-optimized low-power fixed-point math.
4. Compile the project and flash the binary to your STM32 development board.

### Step 3: Monitor Telemetry
Open a serial terminal (e.g. PuTTY or screen) connected to the ST-Link Virtual COM Port:
* **Baud Rate**: `115200`
* **Settings**: 8 Data Bits, 1 Stop Bit, No Parity
The console outputs live telemetry along with true states vs predicted states and classification accuracy:
```text
--- Starting ESN Inference Loop (N=500) ---
[  0] True=NORMAL   Pred=NORMAL   | V=11 I=2 T=32
[  1] True=NORMAL   Pred=NORMAL   | V=11 I=2 T=32
...
[170] True=WARNING  Pred=WARNING  | V=10 I=5 T=35
...
[285] True=CRITICAL Pred=CRITICAL | V=10 I=6 T=45
--- Loop Complete. Accuracy: 98.40% ---
```