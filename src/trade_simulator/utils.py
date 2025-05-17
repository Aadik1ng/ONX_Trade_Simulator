import time
import logging
import configparser

# --- Latency Measurement ---
def measure_latency(start_time):
    """Return elapsed time in milliseconds since start_time (perf_counter)."""
    return (time.perf_counter() - start_time) * 1000

# --- Logging Setup ---
def setup_logging(name="trade_simulator"):
    """Set up and return a logger."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# --- Config Loading ---
def load_api_keys(filepath="api_key"):
    """
    Load API keys and secrets from a config file.
    Returns a dict with keys: apikey, secretkey, etc.
    """
    config = {}
    parser = configparser.ConfigParser()
    with open(filepath, "r") as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                config[k.strip()] = v.strip().strip('"')
    return config

# --- Fee Tier Table (example) ---
FEE_TIERS = {
    "Tier 1": {"maker": 0.0008, "taker": 0.0010},
    "Tier 2": {"maker": 0.0006, "taker": 0.0008},
    "Tier 3": {"maker": 0.0004, "taker": 0.0006},
} 