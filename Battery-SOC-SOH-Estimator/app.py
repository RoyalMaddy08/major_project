import sys
import os
import time
from datetime import datetime
import pickle
import numpy as np

# Add subdirectories to path to support clean imports and robust pickle unpickling
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'simulator'))
sys.path.append(os.path.join(base_dir, 'training'))

from flask import Flask, jsonify, request, render_template
from pymongo import MongoClient
from config import Config

# Dynamic imports from local modules
from battery_simulator import BatterySimulator, DriveCycles
from battery_chemistry import get_chemistry
from traditional_estimator import ExtendedKalmanFilter, ResistanceSOH
from train_rc import EchoStateNetwork
from feature_engineering import extract_features_step

# System resource monitor fallback
try:
    import psutil
    def get_system_metrics():
        process = psutil.Process(os.getpid())
        # Return CPU percentage and RAM in MB
        return process.cpu_percent(), process.memory_info().rss / (1024 * 1024)
except Exception:
    def get_system_metrics():
        # Fallback values for serverless/constrained envs
        return 1.2, 48.5

app = Flask(__name__)

mongodb_uri = Config.MONGODB_URI
db_client = None
db = None
mongodb_connected = False
telemetry_fallback = []  # Fallback in-memory list if MongoDB is offline

# Default state parameters for the stateless simulator
DEFAULT_SIM_STATE = {
    'chemistry': 'li_ion',
    'time': 0.0,
    'soc': 1.0,
    'soh': 1.0,
    'V1': 0.0,
    'V2': 0.0,
    'temperature': 25.0,
    'internal_resistance_growth': 1.0,
    'sim_running': False,
    'active_cycle': 'udds',
    'accelerated_aging': False,
    'last_real_time': None,
    'ekf_mismatch': 1.0,
    'quantize_mode': 'float32',
    # Traditional EKF+CC states
    'cc_soc': 1.0,
    'ekf_soc': 1.0,
    'ekf_v1': 0.0,
    'ekf_v2': 0.0,
    'ekf_p': [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
    # Traditional SOH states
    'trad_soh': 1.0,
    'trad_r0': 0.075,
    'prev_voltage': 3.7 * 3,
    'prev_current': 0.0,
    # ESN Stateful estimation variables
    'esn_soc_state': None,
    'esn_soh_state': None,
    'esn_soc_pred': 1.0,
    'esn_soh_pred': 1.0,
    'rolling_history': []
}
global_sim_state = DEFAULT_SIM_STATE.copy()

simulator = BatterySimulator()

SIMULATION_STEP_DELAY = Config.SIMULATION_STEP_DELAY

# Load trained Reservoir Computing model
model_loaded = False
esn_soc = None
esn_soh = None
input_means = None
input_stds = None

model_path = Config.MODEL_PATH

model_last_modified = 0.0

def load_ml_model():
    global esn_soc, esn_soh, input_means, input_stds, model_loaded, model_last_modified
    if os.path.exists(model_path):
        try:
            mtime = os.path.getmtime(model_path)
            with open(model_path, 'rb') as f:
                package = pickle.load(f)
                esn_soc = package['esn_soc']
                esn_soh = package['esn_soh']
                input_means = package['input_means']
                input_stds = package['input_stds']
                model_loaded = True
                model_last_modified = mtime
                print("Echo State Networks loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model_loaded = False
            model_last_modified = 0.0
    else:
        print(f"Warning: Model file not found at {model_path}. Please run train_rc.py first.")
        model_loaded = False
        model_last_modified = 0.0

# Connect to MongoDB
try:
    db_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=2000)
    db_client.server_info()
    db = db_client[Config.MONGODB_DB_NAME]
    mongodb_connected = True
    db[Config.MONGODB_READINGS_COLLECTION].create_index([("time", 1)])
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Warning: Could not connect to MongoDB ({e}). Falling back to in-memory store.")
    mongodb_connected = False
    telemetry_fallback = []

def save_readings_bulk(readings_list):
    """
    Save list of readings to MongoDB or fallback in-memory store in bulk
    """
    global telemetry_fallback
    for r in readings_list:
        r['timestamp'] = datetime.utcnow().isoformat()
        
    if mongodb_connected:
        try:
            db[Config.MONGODB_READINGS_COLLECTION].insert_many(readings_list)
        except Exception as e:
            print(f"Database write error: {e}")
            for r in readings_list:
                r['_id'] = len(telemetry_fallback)
                telemetry_fallback.append(r)
    else:
        for r in readings_list:
            r['_id'] = len(telemetry_fallback)
            telemetry_fallback.append(r)

    # Cap local fallback array to prevent memory exhaustion in prolonged outages
    if len(telemetry_fallback) > Config.TELEMETRY_FALLBACK_LIMIT:
        telemetry_fallback = telemetry_fallback[-Config.TELEMETRY_FALLBACK_LIMIT:]

def get_all_readings(limit=None):
    if limit is None:
        limit = Config.TELEMETRY_RESPONSE_LIMIT
    if mongodb_connected:
        try:
            cursor = db[Config.MONGODB_READINGS_COLLECTION].find({}, {'_id': False}).sort('time', -1).limit(limit)
            readings = list(cursor)
            readings.reverse()  # Restore chronological order
            return readings
        except Exception as e:
            print(f"Database read error: {e}")
            return telemetry_fallback[-limit:]
    else:
        return telemetry_fallback[-limit:]

def clear_all_readings():
    global telemetry_fallback
    if mongodb_connected:
        try:
            db[Config.MONGODB_READINGS_COLLECTION].delete_many({})
        except Exception as e:
            print(f"Database clear error: {e}")
    telemetry_fallback = []

def load_sim_state():
    global global_sim_state
    if mongodb_connected:
        try:
            state = db[Config.MONGODB_STATE_COLLECTION].find_one({'_id': 'singleton'})
            if state is not None:
                # Remove internal _id key to keep structure clean
                state.pop('_id', None)
                return state
        except Exception as e:
            print(f"Error loading state from MongoDB: {e}")
    return global_sim_state

def save_sim_state(state):
    global global_sim_state
    if mongodb_connected:
        try:
            state_copy = state.copy()
            state_copy['_id'] = 'singleton'
            db[Config.MONGODB_STATE_COLLECTION].replace_one({'_id': 'singleton'}, state_copy, upsert=True)
            return
        except Exception as e:
            print(f"Error saving state to MongoDB: {e}")
    global_sim_state = state

def sync_simulator():
    state = load_sim_state()
    if not state.get('sim_running', False):
        state['last_real_time'] = time.time()
        save_sim_state(state)
        return

    last_real_time = state.get('last_real_time')
    now = time.time()
    
    if last_real_time is None:
        state['last_real_time'] = now
        save_sim_state(state)
        return

    elapsed_real = now - last_real_time
    
    # Mitigate catch-up lag storms on cold start or after long inactivity.
    # If lag exceeds Config.SIM_CATCHUP_THRESHOLD steps worth of real time,
    # truncate to Config.SIM_TRUNCATE_STEPS so the catch-up loop never floods the server.
    if elapsed_real > Config.SIM_CATCHUP_THRESHOLD * SIMULATION_STEP_DELAY:
        last_real_time = now - Config.SIM_TRUNCATE_STEPS * SIMULATION_STEP_DELAY
        elapsed_real = Config.SIM_TRUNCATE_STEPS * SIMULATION_STEP_DELAY

    steps_to_run = int(elapsed_real // SIMULATION_STEP_DELAY)
    
    if steps_to_run <= 0:
        return

    # Cap steps to avoid serverless function timeouts (Config.SIM_MAX_CATCHUP_STEPS)
    steps_to_run = min(steps_to_run, Config.SIM_MAX_CATCHUP_STEPS)

    # Hydrate battery simulator from state
    chemistry_name = state.get('chemistry', 'li_ion')
    simulator.reset(chemistry_name)
    simulator.time = state['time']
    simulator.soc = state['soc']
    simulator.soh = state['soh']
    simulator.V1 = state['V1']
    simulator.V2 = state['V2']
    simulator.temperature = state['temperature']
    simulator.internal_resistance_growth = state['internal_resistance_growth']

    active_cycle = state['active_cycle']
    accelerated_aging = state['accelerated_aging']
    ekf_mismatch = state.get('ekf_mismatch', 1.0)
    quantize_mode = state.get('quantize_mode', 'float32')

    # Hydrate estimators
    ekf = ExtendedKalmanFilter(chemistry_name, mismatch=ekf_mismatch)
    soh_tracker = ResistanceSOH(chemistry_name)
    
    cc_soc = state.get('cc_soc', 1.0)
    ekf_soc = state.get('ekf_soc', 1.0)
    ekf_v1 = state.get('ekf_v1', 0.0)
    ekf_v2 = state.get('ekf_v2', 0.0)
    ekf_p = np.array(state.get('ekf_p', [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]))
    
    trad_soh = state.get('trad_soh', 1.0)
    trad_r0 = state.get('trad_r0', simulator.chemistry.R0_nom)
    prev_voltage = state.get('prev_voltage', simulator.chemistry.lookup_ocv(1.0))
    prev_current = state.get('prev_current', 0.0)

    # Hydrate ESN reservoir state vectors
    esn_soc_state = state.get('esn_soc_state')
    esn_soh_state = state.get('esn_soh_state')
    esn_soc_pred = state.get('esn_soc_pred', 1.0)
    esn_soh_pred = state.get('esn_soh_pred', 1.0)
    rolling_history = state.get('rolling_history', [])

    if model_loaded:
        if esn_soc_state is None or len(esn_soc_state) != esn_soc.n_reservoir:
            esn_soc_state = [0.0] * esn_soc.n_reservoir
        if esn_soh_state is None or len(esn_soh_state) != esn_soh.n_reservoir:
            esn_soh_state = [0.0] * esn_soh.n_reservoir

    # Reservoir priming: when the ESN is fresh (all-zero reservoir state after reset), warm up
    # the recurrent reservoir by feeding the initial battery OCV conditions for Config.ESN_PRIMING_STEPS
    # steps. This prevents the "cold start" convergence lag where predictions drift for 30-50s
    # as the reservoir stabilises from its zero-initialised state.
    if model_loaded and all(v == 0.0 for v in esn_soc_state[:10]):
        V_prime = simulator.chemistry.lookup_ocv(simulator.soc)
        I_prime = 0.0
        T_prime = simulator.temperature
        prime_history = []
        prime_u_raw = extract_features_step(V_prime, I_prime, T_prime, prime_history)
        prime_u_raw[3] = prime_u_raw[3] * (Config.DATASET_TIME_STEP / SIMULATION_STEP_DELAY)
        prime_u_selected = prime_u_raw[Config.ESN_SELECTED_FEATURE_INDICES]
        prime_u_scaled = (prime_u_selected - input_means) / input_stds

        esn_soc.reset_state()
        esn_soh.reset_state()
        for _ in range(Config.ESN_PRIMING_STEPS):
            # Drive reservoir state without making a prediction
            esn_soc._update(prime_u_scaled.reshape(-1, 1))
            esn_soh._update(prime_u_scaled.reshape(-1, 1))
        esn_soc_state = esn_soc.get_state()
        esn_soh_state = esn_soh.get_state()
        print(f"[ESN] Reservoir primed ({Config.ESN_PRIMING_STEPS} steps). V_prime={V_prime:.3f}V, SOC={simulator.soc:.3f}")



    readings_to_save = []

    for _ in range(steps_to_run):
        t = simulator.time
        # Get cycle current
        if active_cycle == "udds":
            I = DriveCycles.udds(t)
        elif active_cycle == "hwfet":
            I = DriveCycles.hwfet(t)
        elif active_cycle == "us06":
            I = DriveCycles.us06(t)
        elif active_cycle == "constant":
            I = DriveCycles.constant_discharge(t)
        elif active_cycle == "charge":
            I = DriveCycles.cccv_charge(t, simulator.soc)
        else:
            I = 0.0

        # Step simulator dynamics
        sim_output = simulator.step(I, SIMULATION_STEP_DELAY, accelerated_aging=accelerated_aging)
        noisy_state = simulator.add_sensor_noise(sim_output)

        V_meas = noisy_state['voltage']
        I_meas = noisy_state['current']  # positive = charge, negative = discharge

        # Update traditional Coulomb Counter
        cc_soc = cc_soc + (I_meas * SIMULATION_STEP_DELAY) / (simulator.nominal_capacity * 3600.0)
        cc_soc = max(0.0, min(1.0, cc_soc))

        # Update traditional 2RC EKF
        t0 = time.perf_counter()
        ekf_soc, ekf_v1, ekf_v2, ekf_p = ekf.step(
            ekf_soc, ekf_v1, ekf_v2, ekf_p,
            I_meas, V_meas, SIMULATION_STEP_DELAY
        )
        ekf_time = (time.perf_counter() - t0) * 1000.0  # ms

        # Update traditional SOH
        trad_r0, trad_soh = soh_tracker.step(
            trad_r0, prev_voltage, prev_current, V_meas, I_meas
        )

        prev_voltage = V_meas
        prev_current = I_meas

        # Stateful ESN inference
        esn_time = 0.0
        if model_loaded:
            t0 = time.perf_counter()
            # Online feature extraction
            u_raw = extract_features_step(V_meas, -I_meas, noisy_state['temperature'], rolling_history)
            # Normalise voltage gradient to match training dataset's time resolution vs simulation interval
            u_raw[3] = u_raw[3] * (Config.DATASET_TIME_STEP / SIMULATION_STEP_DELAY)
            
            # Select robust electrical features via Config.ESN_SELECTED_FEATURE_INDICES
            u_raw_selected = u_raw[Config.ESN_SELECTED_FEATURE_INDICES]
            
            # Scale features
            u_scaled = (u_raw_selected - input_means) / input_stds
            
            # Reset state to last step state vector
            esn_soc.reset_state(esn_soc_state)
            esn_soh.reset_state(esn_soh_state)
            
            # Predict step-wise
            pred_soc_val = esn_soc.predict_step(u_scaled, quantize_mode=quantize_mode)
            pred_soh_val = esn_soh.predict_step(u_scaled, quantize_mode=quantize_mode)
            
            # Print debug info once every 5 simulation seconds
            if int(simulator.time) % 5 == 0:
                print(f"[ESN Debug] Time: {simulator.time}s")
                print(f"  u_raw_selected: {u_raw_selected.tolist()}")
                print(f"  u_scaled: {u_scaled.tolist()}")
                print(f"  Pred SOC: {pred_soc_val[0]:.4f}, Pred SOH: {pred_soh_val[0]:.4f}")
            
            # Capture new state vectors
            esn_soc_state = esn_soc.get_state()
            esn_soh_state = esn_soh.get_state()
            
            esn_soc_pred = float(np.clip(pred_soc_val[0], 0.0, 1.0))
            esn_soh_pred = float(np.clip(pred_soh_val[0], 0.0, 1.0))
            
            # Update rolling history (capped to Config.FEATURE_ROLLING_WINDOW - 1 past entries)
            rolling_history.append({'voltage': V_meas, 'current': -I_meas, 'temperature': noisy_state['temperature']})
            if len(rolling_history) > Config.FEATURE_ROLLING_WINDOW - 1:
                rolling_history.pop(0)
                
            esn_time = (time.perf_counter() - t0) * 1000.0  # ms

        cpu_usage, mem_usage = get_system_metrics()

        # Save record
        record = {
            'time': sim_output['time'],
            'voltage': V_meas,
            'current': -I_meas,  # discharge positive to match charts
            'temperature': noisy_state['temperature'],
            'cc_soc': cc_soc,
            'ekf_soc': ekf_soc,
            'ekf_soh': trad_soh,
            'esn_soc': esn_soc_pred,
            'esn_soh': esn_soh_pred,
            'trad_soh': trad_soh,
            'true_soc': sim_output['true_soc'],
            'true_soh': sim_output['true_soh'],
            # Benchmark telemetry
            'ekf_time': ekf_time,
            'esn_time': esn_time,
            'cpu_usage': cpu_usage,
            'mem_usage': mem_usage
        }
        readings_to_save.append(record)

    if readings_to_save:
        save_readings_bulk(readings_to_save)

    # Save updated state
    new_state = {
        'chemistry': chemistry_name,
        'time': simulator.time,
        'soc': simulator.soc,
        'soh': simulator.soh,
        'V1': simulator.V1,
        'V2': simulator.V2,
        'temperature': simulator.temperature,
        'internal_resistance_growth': simulator.internal_resistance_growth,
        'sim_running': True,
        'active_cycle': active_cycle,
        'accelerated_aging': accelerated_aging,
        'last_real_time': last_real_time + (steps_to_run * SIMULATION_STEP_DELAY),
        'ekf_mismatch': ekf_mismatch,
        'quantize_mode': quantize_mode,
        'cc_soc': cc_soc,
        'ekf_soc': ekf_soc,
        'ekf_v1': ekf_v1,
        'ekf_v2': ekf_v2,
        'ekf_p': ekf_p.tolist(),
        'trad_soh': trad_soh,
        'trad_r0': trad_r0,
        'prev_voltage': prev_voltage,
        'prev_current': prev_current,
        'esn_soc_state': esn_soc_state,
        'esn_soh_state': esn_soh_state,
        'esn_soc_pred': esn_soc_pred,
        'esn_soh_pred': esn_soh_pred,
        'rolling_history': rolling_history
    }
    save_sim_state(new_state)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        global model_loaded, model_last_modified
        if os.path.exists(model_path):
            mtime = os.path.getmtime(model_path)
            if mtime > model_last_modified:
                load_ml_model()
        elif not model_loaded:
            load_ml_model()
            
        sync_simulator()
        state = load_sim_state()
        
        return jsonify({
            'sim_running': state['sim_running'],
            'active_cycle': state['active_cycle'],
            'accelerated_aging': state['accelerated_aging'],
            'model_loaded': model_loaded,
            'mongodb_connected': mongodb_connected,
            'battery_time': state['time'],
            'chemistry': state.get('chemistry', 'li_ion'),
            'ekf_mismatch': state.get('ekf_mismatch', 1.0),
            'quantize_mode': state.get('quantize_mode', 'float32')
        })
    except Exception as e:
        print(f"Error in /api/status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/control', methods=['POST'])
def control_simulation():
    try:
        data = request.json or {}
        command = data.get('command')
        chemistry = data.get('chemistry')
        
        sync_simulator()
        state = load_sim_state()
        
        # Initialize chemistry transition
        if chemistry is not None:
            state['chemistry'] = chemistry
            command = 'reset'  # Force reset when switching chemistry

        if command == 'start':
            state['sim_running'] = True
            state['last_real_time'] = time.time()
        elif command == 'stop':
            state['sim_running'] = False
        elif command == 'reset':
            active_chem = state.get('chemistry', 'li_ion')
            chem_obj = get_chemistry(active_chem)
            
            state['sim_running'] = False
            state['time'] = 0.0
            state['soc'] = 1.0
            state['soh'] = 1.0
            state['V1'] = 0.0
            state['V2'] = 0.0
            state['temperature'] = 25.0
            state['internal_resistance_growth'] = 1.0
            state['last_real_time'] = None
            state['cc_soc'] = 1.0
            state['ekf_soc'] = 1.0
            state['ekf_v1'] = 0.0
            state['ekf_v2'] = 0.0
            state['ekf_p'] = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
            state['trad_soh'] = 1.0
            state['trad_r0'] = chem_obj.R0_nom
            state['prev_voltage'] = chem_obj.lookup_ocv(1.0)
            state['prev_current'] = 0.0
            state['esn_soc_state'] = None
            state['esn_soh_state'] = None
            state['esn_soc_pred'] = 1.0
            state['esn_soh_pred'] = 1.0
            state['rolling_history'] = []
            clear_all_readings()
            
        if 'cycle_type' in data:
            state['active_cycle'] = data['cycle_type']
            
        if 'accelerated_aging' in data:
            state['accelerated_aging'] = bool(data['accelerated_aging'])

        if 'ekf_mismatch' in data:
            state['ekf_mismatch'] = float(data['ekf_mismatch'])

        if 'quantize_mode' in data:
            state['quantize_mode'] = str(data['quantize_mode'])
            
        save_sim_state(state)
        return jsonify({
            'status': 'ok',
            'sim_running': state['sim_running'],
            'active_cycle': state['active_cycle'],
            'accelerated_aging': state['accelerated_aging'],
            'chemistry': state.get('chemistry', 'li_ion'),
            'ekf_mismatch': state.get('ekf_mismatch', 1.0),
            'quantize_mode': state.get('quantize_mode', 'float32')
        })
    except Exception as e:
        print(f"Error in /api/control: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/telemetry', methods=['GET'])
def get_telemetry():
    try:
        sync_simulator()
        readings = get_all_readings()
        
        telemetry_data = []
        for r in readings:
            telemetry_data.append({
                'time': r.get('time', 0.0),
                'voltage': r.get('voltage', 0.0),
                'current': r.get('current', 0.0),
                'temperature': r.get('temperature', 25.0),
                'ekf_soc': r.get('ekf_soc', 1.0),
                'ekf_soh': r.get('ekf_soh', 1.0),
                'esn_soc': r.get('esn_soc', 1.0),
                'esn_soh': r.get('esn_soh', 1.0),
                'cc_soc': r.get('cc_soc', r.get('ekf_soc', 1.0)),
                'trad_soh': r.get('trad_soh', r.get('ekf_soh', 1.0)),
                'true_soc': r.get('true_soc', r.get('ekf_soc', 1.0)),
                'true_soh': r.get('true_soh', r.get('ekf_soh', 1.0)),
                # Benchmarking
                'ekf_time': r.get('ekf_time', 0.0),
                'esn_time': r.get('esn_time', 0.0),
                'cpu_usage': r.get('cpu_usage', 0.0),
                'mem_usage': r.get('mem_usage', 0.0)
            })
            
        return jsonify({
            'model_loaded': model_loaded,
            'data': telemetry_data
        })
    except Exception as e:
        print(f"Error in /api/telemetry: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

load_ml_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.FLASK_DEBUG)
