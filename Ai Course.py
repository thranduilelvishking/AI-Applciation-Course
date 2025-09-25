# -------------------------------
# Smart Industrial Fridge AI Monitor (Streamlit Dashboard Version)
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

# -------------------------------
# STEP 1: Simulate shelf weights
# -------------------------------

def simulate_weight(current_weight):
    normal_drop = random.uniform(0.1, 0.3)
    if random.random() < 0.1:  # 10% chance abnormal
        abnormal_drop = random.uniform(2, 5)
        drop = normal_drop + abnormal_drop
        event = "abnormal"
    else:
        drop = normal_drop
        event = "normal"
    new_weight = max(current_weight - drop, 0)
    return new_weight, event


# -------------------------------
# STEP 2: Train AI models
# -------------------------------

def train_models(history):
    X = pd.DataFrame()
    X["delta"] = history["weight"].diff().fillna(0)
    X["abs_delta"] = X["delta"].abs()

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X)

    lr = LinearRegression()
    t = np.arange(len(history)).reshape(-1, 1)
    w = history["weight"].values
    lr.fit(t, w)

    return iso, lr


# -------------------------------
# STEP 3: Notification System (Email only)
# -------------------------------

class Notifier:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = os.getenv("SMTP_PORT")
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.email_to = os.getenv("EMAIL_TO")

    def send_email(self, subject: str, body: str):
        if not all([self.smtp_host, self.smtp_port, self.smtp_user, self.smtp_pass, self.email_to]):
            st.warning("⚠️ Email not configured, skipping email alert.")
            return False
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.email_to
            with smtplib.SMTP(self.smtp_host, int(self.smtp_port)) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            st.success("📧 Email sent successfully.")
            return True
        except Exception as e:
            st.error(f"Email failed: {e}")
            return False


# -------------------------------
# STEP 4: Dashboard UI
# -------------------------------

st.set_page_config(page_title="Smart Fridge AI Monitor", layout="wide")
st.title("🥶 Smart Industrial Fridge AI Monitor")

# Sidebar controls
seconds = st.sidebar.slider("Simulation steps", 10, 200, 50)
start_weight = st.sidebar.number_input("Starting weight (kg)", 10, 200, 50)
run_button = st.sidebar.button("Start Simulation")

# Placeholders for dynamic updates
chart_placeholder = st.empty()
alerts_placeholder = st.empty()

if run_button:
    history = pd.DataFrame(columns=["time", "weight", "event"])
    weight = start_weight
    notifier = Notifier()
    alerts = []

    for t in range(seconds):
        weight, event = simulate_weight(weight)
        history.loc[len(history)] = [datetime.now(), weight, event]

        if len(history) > 10:
            iso, lr = train_models(history)
            delta = history["weight"].diff().fillna(0).iloc[-1]
            abs_delta = abs(delta)
            X_curr = pd.DataFrame([[delta, abs_delta]], columns=["delta", "abs_delta"])
            pred = iso.predict(X_curr)[0]
            anomaly = (pred == -1)

            slope = lr.coef_[0]
            intercept = lr.intercept_
            eta = None
            if slope < 0:
                t_empty = -intercept / slope
                eta = max(0, t_empty - len(history))

            # Handle alerts
            if anomaly:
                msg = f"⚠️ ALERT: Unusual drop detected at step {t} (Δ={delta:.2f}kg)"
                alerts.append(msg)
                notifier.send_email("Fridge Alert", msg)

            if eta and eta < 10:
                msg = f"📦 Reminder: Shelf will be empty soon (ETA ≈ {eta:.1f} steps)"
                alerts.append(msg)
                notifier.send_email("Fridge Reminder", msg)

        # Update chart
        chart_placeholder.line_chart(history.set_index("time")["weight"])

        # Update alerts
        if alerts:
            alerts_placeholder.markdown("### Alerts")
            for a in alerts[-5:]:
                alerts_placeholder.write(a)

    st.success("✅ Simulation finished.")

