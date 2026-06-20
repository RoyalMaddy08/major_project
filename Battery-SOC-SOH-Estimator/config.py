import os
from dotenv import load_dotenv

# Load environmental variables from .env file
load_dotenv()

class Config:
    # MongoDB configuration (Atlas URI in production, Localhost in development)
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "battery_estimation_db")
    MONGODB_READINGS_COLLECTION = os.environ.get("MONGODB_READINGS_COLLECTION", "readings")
    MONGODB_STATE_COLLECTION = os.environ.get("MONGODB_STATE_COLLECTION", "sim_state")
    
    # Flask port and debug settings
    PORT = int(os.environ.get("PORT", 5000))
    FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t", "yes")
    
    # Directory path mappings
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_env = os.environ.get("MODEL_PATH", "model_rc.pkl")
    MODEL_PATH = model_env if os.path.isabs(model_env) else os.path.join(BASE_DIR, model_env)
    
    csv_env = os.environ.get("CSV_PATH", os.path.join("datasets", "training_ev_battery_dataset_multiclass.csv"))
    CSV_PATH = csv_env if os.path.isabs(csv_env) else os.path.join(BASE_DIR, csv_env)
    
    # Physical battery simulator parameters (defines write interval and step size in seconds)
    SIMULATION_STEP_DELAY = float(os.environ.get("SIMULATION_STEP_DELAY", 1.0))

