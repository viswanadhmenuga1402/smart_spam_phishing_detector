from pathlib import Path
from datetime import datetime
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = BASE_DIR / "data" / "scan_history.csv"
COLUMNS = ["timestamp", "message", "ml_prediction", "predicted_class_score", "rule_risk", "risk_indicators"]

def save_scan(message, label, score, risk_level, flags):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "message": message.strip().replace("\n", " ")[:1000],
        "ml_prediction": label,
        "predicted_class_score": round(score, 4),
        "rule_risk": risk_level,
        "risk_indicators": "; ".join(flags),
    }
    frame = pd.DataFrame([row], columns=COLUMNS)
    frame.to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)

def read_history():
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        frame = pd.read_csv(HISTORY_PATH)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    for column in ("message", "risk_indicators"):
        if column in frame.columns:
            frame[column] = frame[column].astype(object).fillna("").astype(str)
    return frame

def clear_history():
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
