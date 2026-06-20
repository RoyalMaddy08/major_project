import sys
import os
import unittest
import numpy as np

# Add parent directory to path to allow imports of local packages
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'simulator'))
sys.path.append(os.path.join(base_dir, 'training'))

from battery_chemistry import get_chemistry
from battery_simulator import BatterySimulator
from traditional_estimator import ExtendedKalmanFilter, ResistanceSOH
from feature_engineering import extract_features_step, extract_features_df
from train_rc import EchoStateNetwork

class TestBMSPlatform(unittest.TestCase):
    def test_chemistries(self):
        """Test chemistry lookups and OCV bounding"""
        for name in ["nmc", "lfp", "lead_acid", "li_ion"]:
            chem = get_chemistry(name)
            self.assertIsNotNone(chem)
            self.assertTrue(chem.nominal_capacity > 0)
            
            # OCV at 100% and 0% SOC should be ordered
            ocv_full = chem.lookup_ocv(1.0)
            ocv_empty = chem.lookup_ocv(0.0)
            self.assertTrue(ocv_full > ocv_empty)

    def test_simulator_2rc(self):
        """Test simulator 2RC step update dynamics"""
        for name in ["nmc", "lfp", "lead_acid", "li_ion"]:
            sim = BatterySimulator(name)
            self.assertEqual(sim.soc, 1.0)
            self.assertEqual(sim.soh, 1.0)
            self.assertEqual(sim.V1, 0.0)
            self.assertEqual(sim.V2, 0.0)
            
            # Discharge step
            res = sim.step(current=-2.0, dt=1.0)
            self.assertTrue(res['voltage'] > 0.0)
            self.assertTrue(res['true_soc'] < 1.0)
            self.assertTrue(res['temperature'] >= 25.0)

    def test_ekf_2rc(self):
        """Test EKF 3-state (2RC) estimator updates"""
        ekf = ExtendedKalmanFilter("li_ion")
        soc = 1.0
        v1 = 0.0
        v2 = 0.0
        P = np.eye(3) * 0.01
        
        # Take a step
        soc_up, v1_up, v2_up, P_up = ekf.step(
            soc=soc, v1=v1, v2=v2, P=P,
            I_meas=-1.0, V_meas=11.5, dt=1.0
        )
        
        self.assertTrue(0.0 <= soc_up <= 1.0)
        self.assertEqual(P_up.shape, (3, 3))
        # Covariance must remain positive semi-definite
        self.assertTrue(np.all(np.diag(P_up) >= 0.0))

    def test_resistance_soh(self):
        """Test dynamic resistance tracking SOH"""
        soh_tracker = ResistanceSOH("nmc")
        r0 = soh_tracker.R0_nom
        
        # Test step update on transient current draw
        r0_next, soh_next = soh_tracker.step(
            current_r0=r0,
            prev_v=12.6, prev_i=0.0,
            V_meas=12.2, I_meas=-2.0
        )
        # R = |dV|/|dI| = 0.4 / 2.0 = 0.2
        # Which is higher than nominal R0 (0.025 * 3 = 0.075)
        # Resistance grows, SOH degrades
        self.assertTrue(r0_next > r0)
        self.assertTrue(soh_next < 1.0)

    def test_feature_engineering(self):
        """Test feature extraction formatting"""
        history = [
            {'voltage': 12.0, 'current': -1.0, 'temperature': 25.0},
            {'voltage': 11.9, 'current': -1.5, 'temperature': 25.1},
        ]
        features = extract_features_step(
            V_current=11.8, I_current=-2.0, T_current=25.2,
            history=history
        )
        
        self.assertEqual(features.shape, (6,))
        # Features should represent [V, I, T, V_grad, I_ma, T_ma]
        self.assertAlmostEqual(features[3], -0.1) # 11.8 - 11.9
        self.assertAlmostEqual(features[4], np.mean([-1.0, -1.5, -2.0]))

    def test_esn_sparsity(self):
        """Test ESN sparse reservoir properties"""
        esn = EchoStateNetwork(
            n_inputs=3, n_reservoir=100, n_outputs=1,
            spectral_radius=0.9, leak_rate=0.3, sparsity=0.8
        )
        
        # Check that approximately 80% of recurrent weights are zeroed
        zeros_count = np.sum(esn.W_res == 0.0)
        total_elements = esn.n_reservoir ** 2
        sparsity_ratio = zeros_count / total_elements
        self.assertAlmostEqual(sparsity_ratio, 0.8, delta=0.08)

if __name__ == '__main__':
    unittest.main()
