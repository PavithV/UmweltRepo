import sys
sys.path.append('/app/backend')
import os, dash, pandas as pd
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import psycopg2
from backend.ml.anomaly_detector import predict

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "sensebox"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}
SENSOR_ID = os.getenv("ML_SENSOR_ID")
MAX_POINTS = 1000  # points shown in the graph
MODEL_PATH = "/app/model/anomaly_iforest.joblib"  # Pfad im Container


def fetch_data(sensor_id):
    q = """
        SELECT time, value
        FROM measurements
        WHERE sensor_id = %s
        ORDER BY time DESC
        LIMIT %s;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(q, conn, params=[sensor_id, MAX_POINTS])
    return df.sort_values("time")


def layout():
    return html.Div(
        [
            html.H2("SenseBox Live Dashboard"),
            dcc.Dropdown(id="sensor-select", placeholder="Choose sensor"),
            dcc.Graph(id="live-graph"),
            dcc.Interval(id="update", interval=30_000, n_intervals=0),
        ],
        style={"padding": "2rem"},
    )


app = dash.Dash(__name__)
app.layout = layout


@app.callback(
    Output("sensor-select", "options"),
    Input("update", "n_intervals"),
)
def update_sensors(_):
    q = """
        SELECT DISTINCT sensor_id, sensor_name, unit
        FROM measurements
        WHERE sensor_name IS NOT NULL
        ORDER BY sensor_name
        LIMIT 50;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(q, conn)
    options = []
    for _, row in df.iterrows():
        name = row["sensor_name"] or row["sensor_id"]
        unit = row.get("unit") or ""
        label = f"{name}" + (f" [{unit}]" if unit else "")
        options.append({"label": label, "value": row["sensor_id"]})
    return options



@app.callback(
    Output("live-graph", "figure"),
    Input("update", "n_intervals"),
    Input("sensor-select", "value"),
)
def update_graph(_, sensor_id):
    if not sensor_id:
        return go.Figure()
    df = fetch_data(sensor_id)
    fig = go.Figure()
    fig.add_scatter(x=df["time"], y=df["value"], mode="lines", name="value")

    # If this is the ML sensor, mark anomalies
    if SENSOR_ID and sensor_id == SENSOR_ID and len(df) > 20:
        preds = predict(df["value"])
        anomalies = df[preds == -1]
        fig.add_scatter(
            x=anomalies["time"],
            y=anomalies["value"],
            mode="markers",
            marker_symbol="x",
            marker_size=10,
            name="anomaly",
        )

    fig.update_layout(yaxis_title="Sensor Value", xaxis_title="Time")
    return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
