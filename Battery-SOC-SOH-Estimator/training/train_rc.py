import sys
import os
import pandas as pd
import numpy as np
import pickle

# Add base directory to path to support config imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from config import Config

try:
    from feature_engineering import extract_features_df
except ImportError:
    from training.feature_engineering import extract_features_df

class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=0.95, leak_rate=0.3, input_scaling=1.0, ridge_param=1e-4, sparsity=0.85):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.ridge_param = ridge_param
        self.sparsity = sparsity
        
        # Initialize input weights
        np.random.seed(42)  # For reproducible weights
        self.W_in = (np.random.rand(n_reservoir, 1 + n_inputs) - 0.5) * 2.0 * input_scaling
        
        # Initialize reservoir weights
        W = np.random.randn(n_reservoir, n_reservoir)
        
        # Apply sparsity (zero out random elements)
        if sparsity > 0.0:
            mask = np.random.rand(*W.shape) < sparsity
            W[mask] = 0.0
            
        # Scale reservoir weights to have desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eigenval = np.max(np.abs(eigenvalues))
        if max_eigenval > 0:
            self.W_res = W * (spectral_radius / max_eigenval)
        else:
            self.W_res = W
            
        # Readout weights
        self.W_out = None
        
        # Reservoir state vector
        self.x = np.zeros((n_reservoir, 1))

    def reset_state(self, state_vector=None):
        if state_vector is not None:
            self.x = np.array(state_vector).reshape(self.n_reservoir, 1)
        else:
            self.x = np.zeros((self.n_reservoir, 1))

    def get_state(self):
        return self.x.flatten().tolist()

    def _update(self, u):
        # u is shape (n_inputs, 1)
        u_biased = np.vstack(([1.0], u))
        # Reservoir state update:
        # x(t) = (1 - alpha)*x(t-1) + alpha * tanh(W_in * u_biased + W_res * x(t-1))
        arg = np.dot(self.W_in, u_biased) + np.dot(self.W_res, self.x)
        self.x = (1.0 - self.leak_rate) * self.x + self.leak_rate * np.tanh(arg)
        return self.x

    def train(self, U, Y):
        """
        Train the readout weights W_out using Ridge Regression.
        :param U: input sequence, shape (n_samples, n_inputs)
        :param Y: target sequence, shape (n_samples, n_outputs)
        """
        n_samples = U.shape[0]
        self.reset_state()
        
        states = []
        for t in range(n_samples):
            u_t = U[t].reshape(-1, 1)
            x_t = self._update(u_t)
            state_vec = np.vstack(([1.0], u_t, x_t))
            states.append(state_vec.flatten())
            
        # Design matrix X: (1 + n_inputs + n_reservoir, n_samples)
        X = np.array(states).T
        
        # Target matrix Y_target: (n_outputs, n_samples)
        Y_target = Y.reshape(n_samples, self.n_outputs).T
        
        # Ridge Regression: W_out = Y_target * X^T * (X * X^T + lambda * I)^-1
        X_XT = np.dot(X, X.T)
        reg_matrix = self.ridge_param * np.eye(X.shape[0])
        self.W_out = np.dot(np.dot(Y_target, X.T), np.linalg.inv(X_XT + reg_matrix))
        
    def predict_step(self, u):
        """
        Advance ESN state by one step and make prediction.
        :param u: input vector of shape (n_inputs,)
        """
        u_t = np.array(u).reshape(-1, 1)
        x_t = self._update(u_t)
        state_vec = np.vstack(([1.0], u_t, x_t))
        y_pred = np.dot(self.W_out, state_vec)
        return y_pred.flatten()

    def predict(self, U):
        """
        Predict output sequence for a series of inputs U.
        :param U: shape (n_samples, n_inputs)
        """
        n_samples = U.shape[0]
        self.reset_state()
        predictions = []
        for t in range(n_samples):
            y_pred = self.predict_step(U[t])
            predictions.append(y_pred)
        return np.array(predictions)

def main():
    csv_path = Config.CSV_PATH
    model_save_path = Config.MODEL_PATH

    print(f"Loading completed dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        print("Error: Dataset not found. Please verify Config.CSV_PATH.")
        return
        
    df = pd.read_csv(csv_path)

    # 1. Feature Engineering
    print("Performing feature engineering...")
    U_raw = extract_features_df(df)
    n_features = U_raw.shape[1]
    print(f"Engineered {n_features} features: Voltage, Current, Temperature, Voltage_grad, Current_ma, Temp_ma")
    
    # Target values
    Y_soc = df[['SOC']].values
    Y_soh = df[['SOH']].values

    # 2. Scale inputs (Zero-mean, Unit-variance scaling)
    print("Normalizing input features...")
    input_means = U_raw.mean(axis=0)
    input_stds = U_raw.std(axis=0)
    input_stds[input_stds == 0.0] = 1.0
    
    U_scaled = (U_raw - input_means) / input_stds

    # 3. Instantiate and Train SOC ESN
    # SOC changes rapidly: spectral_radius=0.95, leak_rate=0.3, sparsity=0.85
    print("Initializing and training SOC Echo State Network...")
    esn_soc = EchoStateNetwork(
        n_inputs=n_features,
        n_reservoir=200,
        n_outputs=1,
        spectral_radius=0.95,
        leak_rate=0.3,
        input_scaling=0.5,
        ridge_param=1e-4,
        sparsity=0.85
    )
    esn_soc.train(U_scaled, Y_soc)
    
    # Evaluate SOC ESN
    pred_soc = esn_soc.predict(U_scaled)
    soc_rmse = np.sqrt(np.mean((Y_soc - pred_soc) ** 2))
    print(f"Training SOC RMSE: {soc_rmse:.6f}")

    # 4. Instantiate and Train SOH ESN
    # SOH changes slowly: spectral_radius=0.6, leak_rate=0.05, sparsity=0.85, ridge_param=1e-2
    print("Initializing and training SOH Echo State Network...")
    esn_soh = EchoStateNetwork(
        n_inputs=n_features,
        n_reservoir=150,
        n_outputs=1,
        spectral_radius=0.60,
        leak_rate=0.05,
        input_scaling=0.2,
        ridge_param=1e-2,
        sparsity=0.85
    )
    esn_soh.train(U_scaled, Y_soh)
    
    # Evaluate SOH ESN
    pred_soh = esn_soh.predict(U_scaled)
    soh_rmse = np.sqrt(np.mean((Y_soh - pred_soh) ** 2))
    print(f"Training SOH RMSE: {soh_rmse:.6f}")

    # 5. Save model package
    model_package = {
        'esn_soc': esn_soc,
        'esn_soh': esn_soh,
        'input_means': input_means,
        'input_stds': input_stds
    }
    
    print(f"Saving trained ESN models to {model_save_path}...")
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_package, f)
        
    print("Model package saved successfully!")

if __name__ == "__main__":
    main()
