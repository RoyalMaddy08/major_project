import numpy as np
import pandas as pd

def extract_features_df(df):
    """
    Offline feature engineering on a pandas DataFrame.
    Returns: U_engineered (numpy array of shape (N, n_features))
    """
    # Make a copy to avoid modifications
    df = df.copy()
    
    # Calculate gradient of Voltage (dV)
    df['Voltage_grad'] = df['Voltage'].diff().fillna(0.0)
    
    # Calculate rolling averages of Current and Temperature (window=5)
    df['Current_ma'] = df['Current'].rolling(window=5, min_periods=1).mean()
    df['Temp_ma'] = df['Temperature'].rolling(window=5, min_periods=1).mean()
    
    feature_cols = ['Voltage', 'Current', 'Temperature', 'Voltage_grad', 'Current_ma', 'Temp_ma']
    return df[feature_cols].values

def extract_features_step(V_current, I_current, T_current, history):
    """
    Online feature engineering step-by-step at runtime.
    :param history: List of dictionaries of past readings, e.g. [{'voltage': v, 'current': i, 'temperature': t}, ...]
    :returns: numpy array of shape (n_features,)
    """
    # If history is empty, initialize defaults
    if len(history) == 0:
        V_prev = V_current
        V_history = [V_current]
        I_history = [I_current]
        T_history = [T_current]
    else:
        V_prev = history[-1]['voltage']
        V_history = [r['voltage'] for r in history[-4:]] + [V_current]
        I_history = [r['current'] for r in history[-4:]] + [I_current]
        T_history = [r['temperature'] for r in history[-4:]] + [T_current]
        
    V_grad = V_current - V_prev
    I_ma = np.mean(I_history)
    T_ma = np.mean(T_history)
    
    return np.array([V_current, I_current, T_current, V_grad, I_ma, T_ma])
