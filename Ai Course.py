# -------------------------------
# Smart Industrial Fridge – Multi-Shelf AI Monitor (Streamlit)
# -------------------------------
# Features:
# - Shelves: Dairy, Meat, Vegetables
# - 5 products per shelf; initial weights random 20–40 kg (per run)
# - Simulation: normal & abnormal consumption, random restocks
# - Anomaly detection: Isolation Forest (per product)
# - Forecast depletion: Linear Regression (per product), 1 step = 0.5 days
# - Overstock alert if weight > initial * 1.8
# - All alert emails include current weight
# - Email via SMTP_* env vars (same as your PyCharm setup)
# -------------------------------

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.text import MIMEText

# Time mapping: each simulation step equals 0.5 day
TIME_PER_STEP_DAYS = 0.5
OVERSTOCK_FACTOR = 1.8

# Define shelves and products (names only; weights are randomized at runtime)
PRODUCTS = {
    "Dairy":       ["Milk", "Cheese", "Yogurt", "Butter", "Cream"],
    "Meat":        ["Chicken", "Beef", "Fish", "Sausage", "Ham"],
    "Vegetables":  ["Carrots", "Potatoes", "Onions", "Tomatoes", "Lettuce"],
}

# -------------------------------
# Notifier (Email + Streamlit UX)
# -------------------------------

class Notifier:
    def __init__(self):
        # Read from Streamlit secrets if present, else from environment
        def get_secret(name, default=None):
            try:
                return st.secrets.get(name, None) if hasattr(st, "secrets") else None
            except Exception:
                return None

        self.smtp_host = get_secret("SMTP_HOST") or os.getenv("SMTP_HOST")
        self.smtp_port = get_secret("SMTP_PORT") or os.getenv("SMTP_PORT")
        self.smtp_user = get_secret("SMTP_USER") or os.getenv("SMTP_USER")
        self.smtp_pass = get_secret("SMTP_PASS") or os.getenv("SMTP_PASS")
        self.email_to  = get_secret("EMAIL_TO")  or os.getenv("EMAIL_TO")

    @property
    def configured(self) -> bool:
        return all([self.smtp_host, self.smtp_port, self.smtp_user, self.smtp_pass, self.email_to])

    def send_email(self, subject: str, body: str) -> bool:
        if not self.configured:
            st.warning("Email not configured; skipping email send.")
            return False
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"]   = self.email_to
            with smtplib.SMTP(self.smtp_host, int(self.smtp_port)) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            st.toast("Email sent")
            return True
        except Exception as e:
            st.error(f"Email failed: {e}")
            return False

# -------------------------------
# Models per product
# -------------------------------

def train_models(history_df: pd.DataFrame):
    """Train Isolation Forest & Linear Regression for one product's history."""
    X = pd.DataFrame()
    X["delta"] = history_df["weight"].diff().fillna(0.0)
    X["abs_delta"] = X["delta"].abs()

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X)

    lr = LinearRegression()
    t_idx = np.arange(len(history_df)).reshape(-1, 1)
    w = history_df["weight"].values
    lr.fit(t_idx, w)

    return iso, lr

# -------------------------------
# State initialization helpers
# -------------------------------

def initialize_weights() -> dict:
    """Randomize initial weights 20–40 kg for each product."""
    return {
        shelf: {prod: random.uniform(20.0, 40.0) for prod in PRODUCTS[shelf]}
        for shelf in PRODUCTS
    }

def deep_copy_fridge(fridge: dict) -> dict:
    return {shelf: dict(items) for shelf, items in fridge.items()}

# -------------------------------
# Simulation step
# -------------------------------

def simulate_step(fridge_state: dict,
                  histories: dict,
                  initial_weights: dict,
                  alerts: list,
                  notifier: Notifier):
    """
    Advance the simulation by one step for all products on all shelves.
    Generates consumption, possible restock, then AI checks & alerts.
    """
    for shelf, products in fridge_state.items():
        for product, curr_weight in products.items():

            # 1) Random restock chance (5%): add 3–10 kg
            did_restock = False
            if random.random() < 0.05:
                restock_amt = random.uniform(3.0, 10.0)
                new_weight = curr_weight + restock_amt
                did_restock = True
                msg = (f"Restock: {shelf} – {product} restocked (+{restock_amt:.1f} kg). "
                       f"Current weight: {new_weight:.2f} kg")
                alerts.append(msg)
                notifier.send_email("Fridge Restock", msg)
            else:
                # 2) Consumption: normal 0.1–0.3 kg; 10% chance of extra abnormal 2–5 kg
                normal_drop = random.uniform(0.1, 0.3)
                if random.random() < 0.10:
                    abnormal_drop = random.uniform(2.0, 5.0)
                    total_drop = normal_drop + abnormal_drop
                    event = "abnormal"
                else:
                    total_drop = normal_drop
                    event = "normal"
                new_weight = max(curr_weight - total_drop, 0.0)

            # 3) Overstock check (after restock)
            if did_restock:
                over_limit = initial_weights[shelf][product] * OVERSTOCK_FACTOR
                if new_weight > over_limit:
                    msg = (f"Overload: {shelf} – {product} exceeded safe limit "
                           f"({new_weight:.2f} kg > {over_limit:.2f} kg).")
                    alerts.append(msg)
                    notifier.send_email("Fridge Overstock Alert", msg)

            # 4) Update state & history
            fridge_state[shelf][product] = new_weight

            if product not in histories[shelf]:
                histories[shelf][product] = pd.DataFrame(columns=["time", "weight", "event"])

            histories[shelf][product].loc[len(histories[shelf][product])] = [
                datetime.utcnow(),
                new_weight,
                "restock" if did_restock else event
            ]

            # 5) AI checks (need history length)
            hist_df = histories[shelf][product]
            if len(hist_df) > 10:
                iso, lr = train_models(hist_df)

                # Current feature vector
                delta = hist_df["weight"].diff().fillna(0.0).iloc[-1]
                X_curr = pd.DataFrame([[delta, abs(delta)]], columns=["delta", "abs_delta"])
                pred = iso.predict(X_curr)[0]  # 1 normal, -1 anomaly
                is_anomaly = (pred == -1)

                # Forecast time to empty (steps -> days)
                slope = lr.coef_[0]
                intercept = lr.intercept_
                eta_days = None
                if slope < 0:
                    t_empty = -intercept / slope
                    steps_remaining = t_empty - (len(hist_df) - 1)
                    if steps_remaining >= 0:
                        eta_days = steps_remaining * TIME_PER_STEP_DAYS

                # Alerts
                if is_anomaly and not did_restock:
                    msg = (f"Anomaly: {shelf} – {product} drop unusual (Δ={delta:.2f} kg). "
                           f"Current weight: {new_weight:.2f} kg")
                    alerts.append(msg)
                    notifier.send_email("Fridge Anomaly Alert", msg)

                if eta_days is not None and eta_days < 7.0:
                    msg = (f"Depletion Soon: {shelf} – {product} may empty in "
                           f"{eta_days:.1f} days. Current weight: {new_weight:.2f} kg")
                    alerts.append(msg)
                    notifier.send_email("Fridge Depletion Warning", msg)

    return fridge_state, histories, alerts

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Smart Fridge AI", layout="wide")
st.title("Smart Industrial Fridge — Multi-Shelf AI Monitor")

with st.sidebar:
    st.subheader("Simulation Controls")
    steps = st.slider("Simulation steps", 10, 300, 80, 1)
    st.caption("Time mapping: 1 step = 0.5 days")
    st.subheader("Email")
    st.caption("Uses SMTP_* secrets or environment variables. Email is optional.")

run_btn = st.button("Run Simulation")
reset_btn = st.button("Reset")

# Session state: initial weights, current weights, histories, alerts
if "initial_weights" not in st.session_state:
    st.session_state.initial_weights = initialize_weights()
if "fridge_state" not in st.session_state:
    st.session_state.fridge_state = deep_copy_fridge(st.session_state.initial_weights)
if "histories" not in st.session_state:
    st.session_state.histories = {shelf: {} for shelf in PRODUCTS}
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "notifier" not in st.session_state:
    st.session_state.notifier = Notifier()

if reset_btn:
    st.session_state.initial_weights = initialize_weights()
    st.session_state.fridge_state = deep_copy_fridge(st.session_state.initial_weights)
    st.session_state.histories = {shelf: {} for shelf in PRODUCTS}
    st.session_state.alerts = []
    st.success("Simulation reset with new randomized initial weights (20–40 kg).")

# Show current initial weights (for transparency & overstock thresholds)
with st.expander("Initial weights (randomized this run) & overstock thresholds"):
    rows = []
    for shelf, prods in st.session_state.initial_weights.items():
        for prod, w0 in prods.items():
            rows.append({
                "Shelf": shelf,
                "Product": prod,
                "Initial (kg)": round(w0, 2),
                "Overstock if > (kg)": round(w0 * OVERSTOCK_FACTOR, 2)
            })
    st.dataframe(pd.DataFrame(rows).sort_values(["Shelf", "Product"]), use_container_width=True, height=260)

# Placeholders for charts and alerts
chart_cols = st.columns(3)
chart_placeholders = {
    "Dairy": chart_cols[0].container(),
    "Meat": chart_cols[1].container(),
    "Vegetables": chart_cols[2].container(),
}
alerts_placeholder = st.container()

def render_shelf_charts(histories: dict):
    for shelf, container in chart_placeholders.items():
        with container:
            st.subheader(f"{shelf} shelf")
            # Combine product histories for this shelf
            combined = []
            for prod, df in histories[shelf].items():
                if not df.empty:
                    tmp = df.copy()
                    tmp["product"] = prod
                    combined.append(tmp)
            if not combined:
                st.info("No data yet. Run the simulation.")
                continue
            dfc = pd.concat(combined)
            # Pivot for multi-line chart
            pivot = dfc.pivot(index="time", columns="product", values="weight")
            st.line_chart(pivot, height=300)

def render_alerts(alerts: list):
    st.subheader("Alerts")
    if not alerts:
        st.info("No alerts yet.")
    else:
        for a in alerts[-15:]:
            st.write(a)

if run_btn:
    for _ in range(steps):
        st.session_state.fridge_state, st.session_state.histories, st.session_state.alerts = simulate_step(
            st.session_state.fridge_state,
            st.session_state.histories,
            st.session_state.initial_weights,
            st.session_state.alerts,
            st.session_state.notifier
        )
        # Update UI incrementally
        render_shelf_charts(st.session_state.histories)
        render_alerts(st.session_state.alerts)

    st.success("Simulation finished.")
else:
    # Render current state if not running
    render_shelf_charts(st.session_state.histories)
    render_alerts(st.session_state.alerts)

