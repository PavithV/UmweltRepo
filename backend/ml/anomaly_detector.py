import os
import time
import joblib
import pandas as pd
import psycopg2
from sklearn.ensemble import IsolationForest
from database import get_conn

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/anomaly_iforest.joblib")
SENSOR_ID = os.getenv("ML_SENSOR_ID")  # fokussierter Sensor
TRAIN_LIMIT = int(os.getenv("TRAIN_LIMIT", "10000"))

# Warte auf Datenbank-Verbindung
def wait_for_db(max_retries=10, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            conn = get_conn()
            conn.close()
            print("DB-Verbindung erfolgreich")
            return
        except psycopg2.OperationalError as e:
            print(f"Warte auf DB (Versuch {retries+1}/{max_retries}): {e}")
            time.sleep(delay)
            retries += 1
    raise RuntimeError("Konnte nach mehreren Versuchen keine Verbindung zur Datenbank aufbauen.")

def load_data():
    # Holt die letzten TRAIN_LIMIT Messungen für SENSOR_ID
    query = """
        SELECT EXTRACT(EPOCH FROM time) AS t, value
        FROM measurements
        WHERE sensor_id = %s
        ORDER BY time DESC
        LIMIT %s;
    """
    # get_conn() öffnet Verbindung zu DB
    with get_conn() as conn:
        df = pd.read_sql(query, conn, params=[SENSOR_ID, TRAIN_LIMIT])
    # Sortiere aufsteigend nach Zeit
    return df.sort_values("t")

def train():
    df = load_data()
    if df.empty:
        print("Keine Daten für Training gefunden. Beende.")
        return
    # Nur Werte spalten
    model = IsolationForest(contamination=0.01, random_state=0)
    model.fit(df[["value"]])
    # Verzeichnis sicherstellen
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
        except Exception as e:
            print(f"Fehler beim Anlegen des Modell-Ordners: {e}")
    joblib.dump(model, MODEL_PATH)
    print("Model trained & saved →", MODEL_PATH)
    # Einfache Evaluation:
    try:
        preds = model.predict(df[["value"]])
        n_anom = int((preds == -1).sum())
        print(f"Anomalien im Trainingsset: {n_anom} von {len(df)}")
    except Exception:
        pass

def predict(values: pd.Series):
    """values: pd.Series von float sensor readings"""
    # Lädt Modell
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelldatei nicht gefunden unter {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    arr = values.to_numpy().reshape(-1, 1)
    return model.predict(arr)  # -1 (Anomalie) / 1 (normal)

if __name__ == "__main__":
    if not SENSOR_ID:
        print("Setze ML_SENSOR_ID-Umgebungsvariable, um Modell zu trainieren.")
    else:
        # Warte auf DB (maximal retry-Versuche)
        try:
            wait_for_db(max_retries=12, delay=5)
        except RuntimeError as e:
            print("Fehler: ", e)
            exit(1)
        # Trainiere Modell
        train()
        # Falls du periodisch neu trainieren willst, kannst du hier schedule verwenden:
        # import schedule
        # schedule.every(24).hours.do(train)
        # while True:
        #     schedule.run_pending()
        #     time.sleep(60)
