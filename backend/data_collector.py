import os, time, requests, pandas as pd, sys
from datetime import datetime, timezone
from database import insert_measurements, get_conn
import psycopg2

BASE_URL = "https://api.opensensemap.org/boxes/{box_id}"
BOX_ID = os.getenv("SENSEBOX_ID") or "5e7e9b94946d0c001b6e64b4"
POLL_SEC = int(os.getenv("POLL_SECONDS", 30))

def wait_for_db(max_retries=12, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            conn = get_conn()
            conn.close()
            print("DB-Verbindung erfolgreich", flush=True)
            return
        except psycopg2.OperationalError as e:
            print(f"Warte auf DB (Versuch {retries+1}/{max_retries}): {e}", flush=True)
            time.sleep(delay)
            retries += 1
    print("Konnte keine Verbindung zur DB aufbauen, beende Collector.", flush=True)
    sys.exit(1)

def fetch_latest_measurements():
    url = BASE_URL.format(box_id=BOX_ID)
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    sensors = r.json().get("sensors", [])
    rows = []
    for s in sensors:
        val = s.get("lastMeasurement")
        if not val:
            continue
        ts = datetime.fromisoformat(val["createdAt"].replace("Z", "+00:00")).astimezone(timezone.utc)
        rows.append((ts, s["_id"],s.get("title"),s.get("unit"), float(val["value"])))
    return rows

def main():
    # Warte initial auf DB
    wait_for_db()
    print("Collector gestartet", flush=True)
    while True:
        try:
            print(f"Abruf für SenseBox {BOX_ID} ...", flush=True)
            rows = fetch_latest_measurements()
            print(f"{len(rows)} Messwerte gefunden", flush=True)
            if rows:
                insert_measurements(rows)
        except Exception as e:
            print("Collector error:", e, flush=True)
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
