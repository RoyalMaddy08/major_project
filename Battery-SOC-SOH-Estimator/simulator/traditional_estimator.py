import numpy as np

try:
    from battery_chemistry import get_chemistry
except ImportError:
    from simulator.battery_chemistry import get_chemistry

class ExtendedKalmanFilter:
    def __init__(self, chemistry_name="li_ion", mismatch=1.0):
        self.chemistry_name = chemistry_name
        self.chemistry = get_chemistry(chemistry_name)
        
        # Load nominal parameters from chemistry scaled by mismatch factor
        self.Cn = self.chemistry.nominal_capacity  # Ah
        self.R0 = self.chemistry.R0_nom * mismatch  # Ohms
        self.R1 = self.chemistry.R1_nom * mismatch  # Ohms
        self.C1 = self.chemistry.C1_nom * mismatch  # Farads
        self.R2 = self.chemistry.R2_nom * mismatch  # Ohms
        self.C2 = self.chemistry.C2_nom * mismatch  # Farads

        # Process noise covariance Q (states: SOC, V1, V2)
        self.Q = np.diag([1e-7, 1e-6, 1e-6])
        
        # Measurement noise covariance R
        self.R_meas = 0.01
        
    def step(self, soc, v1, v2, P, I_meas, V_meas, dt):
        """
        Runs one prediction-correction 2RC EKF step.
        Note: Current I_meas is positive for charge, negative for discharge.
        :param P: 3x3 numpy covariance matrix
        :returns: updated (soc, v1, v2, P_updated_3x3)
        """
        x = np.array([[soc], [v1], [v2]])

        # 1. State Transition Matrices for 2RC
        tau1 = self.R1 * self.C1
        tau2 = self.R2 * self.C2
        
        a1 = np.exp(-dt / tau1) if tau1 > 0 else 0.0
        b1 = self.R1 * (1.0 - a1)
        
        a2 = np.exp(-dt / tau2) if tau2 > 0 else 0.0
        b2 = self.R2 * (1.0 - a2)
        
        F = np.array([[1.0, 0.0, 0.0],
                      [0.0, a1, 0.0],
                      [0.0, 0.0, a2]])
        
        # 2. Prediction Step
        # SOC prediction (Coulomb Counting)
        soc_pred = soc + (I_meas * dt) / (self.Cn * 3600.0)
        soc_pred = np.clip(soc_pred, 0.0, 1.0)
        
        # Polarization voltages predictions
        v1_pred = a1 * v1 + b1 * I_meas
        v2_pred = a2 * v2 + b2 * I_meas
        
        x_pred = np.array([[soc_pred], [v1_pred], [v2_pred]])
        P_pred = np.dot(np.dot(F, P), F.T) + self.Q
        
        # 3. Measurement Prediction
        ocv = self.chemistry.lookup_ocv(soc_pred)
        # Predicted terminal voltage: Vt = OCV(SOC) + I * R0 + V1 + V2
        V_pred = ocv + I_meas * self.R0 + v1_pred + v2_pred
        
        # 4. Measurement Jacobian H = [dOCV/dSOC, 1, 1]
        eps = 0.001
        ocv_plus = self.chemistry.lookup_ocv(soc_pred + eps)
        ocv_minus = self.chemistry.lookup_ocv(soc_pred - eps)
        dOCV = (ocv_plus - ocv_minus) / (2.0 * eps)
        
        H = np.array([[dOCV, 1.0, 1.0]])
        
        # 5. Innovation / Correction Step
        residual = V_meas - V_pred
        S = np.dot(np.dot(H, P_pred), H.T) + self.R_meas
        K = np.dot(P_pred, H.T) / S[0, 0]
        
        x_updated = x_pred + K * residual
        soc_updated = float(np.clip(x_updated[0, 0], 0.0, 1.0))
        v1_updated = float(x_updated[1, 0])
        v2_updated = float(x_updated[2, 0])
        
        I_mat = np.eye(3)
        P_updated = np.dot((I_mat - np.dot(K, H)), P_pred)
        
        return soc_updated, v1_updated, v2_updated, P_updated

class ResistanceSOH:
    def __init__(self, chemistry_name="li_ion", alpha=0.05):
        self.chemistry_name = chemistry_name
        self.chemistry = get_chemistry(chemistry_name)
        self.R0_nom = self.chemistry.R0_nom
        self.alpha = alpha  # low-pass filter coefficient
        
    def step(self, current_r0, prev_v, prev_i, V_meas, I_meas):
        """
        Estimates SOH based on dynamic internal resistance calculations.
        Returns: updated (current_r0, soh_estimate)
        """
        # Calculate transients
        dI = I_meas - prev_i
        dV = V_meas - prev_v
        
        r0_est = current_r0
        if abs(dI) > 0.5:
            # Dynamic resistance calculation: R = dV / dI
            # R = |dV| / |dI| is a robust estimator.
            r0_calc = abs(dV) / abs(dI)
            
            # Bound the calculated R0 value to physically reasonable limits
            # Resistance growth can grow up to 2.5x
            if 0.5 * self.R0_nom < r0_calc < 3.0 * self.R0_nom:
                # Apply low-pass filter
                r0_est = (1.0 - self.alpha) * current_r0 + self.alpha * r0_calc
                
        # Invert the resistance growth formula: R = R0_nom * (1 + 1.5 * (1 - SOH))
        soh_est = 1.0 - ((r0_est / self.R0_nom) - 1.0) / 1.5
        soh_est = np.clip(soh_est, 0.2, 1.0)
        
        return float(r0_est), float(soh_est)
