import json
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("history.json")

def load_history():
    """Load project history from JSON file"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_history_cached():
    """Load history directly from file without caching"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history_entry(method, input_data, output_data):
    """Save a new project entry to history"""
    history = load_history()
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "input": input_data,
        "output": output_data
    }
    history.append(new_entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)