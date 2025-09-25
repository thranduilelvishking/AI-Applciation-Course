# Smart Industrial Fridge Monitoring (Daily Summary Email Version)

import random
import time
import os
import smtplib
import numpy as np
import pandas as pd
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# -------------------------------
# Email Notifier
# -------------------------------

class EmailNotifier:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.email_to = os.getenv("EMAIL_TO")

        # collect alerts during the day
        self.alerts_today = []

    def add_alert(self, message: str):
        """Store alert message for later summary"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alerts_today.append(f"[{timestamp}] {message}")

    def send_daily_summary(self, day: str):
        """Send one email with all alerts of the day"""
        if not self.alerts_today:
            body = f"No unusual events detected on {day}."
        else:
            body = "\n".join(self.alerts_today)

        msg = MIMEText(body)
        msg["Subject"] = f"Smart Fridge Daily Report – {day}"
        msg["From"] = self.smtp_user
        msg["To"] = self.email_to

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            print(f"📧 Daily summary email sent for {day}")
        except Exception as e:
            print("❌ Email failed:", e)

        # reset alerts for the next day
        self.alerts_today = []


# -------------------------------
# Simulation Setup
# -------------------------------

SHELVES = {
    "Dairy": ["Milk", "Cheese", "Yogurt", "Butter", "Cream"],
    "Meat": ["Chicken", "Beef", "Pork", "Fish", "Lamb"],
    "Vegetables": ["Carrots", "Potatoes", "Onions", "Tomatoes", "Cabbage"],
}

# assign random initial weights (20–40 kg each)
INITIAL_STOCK = {shelf: {p: random.randint(20, 40) for p in products}
                 for shelf, products in SHELVES.items()}


# -------------------------------
# Functions
# -------------------------------

def simulate_weight_change(current_weight):
    """Simulate product weight change (normal use, anomaly, or restock)."""
    event = "normal"

    # 10% chance of abnormal big drop
    if random.random() < 0.1:
        drop = random.uniform(2, 5)
        event = "abnormal"
    else:
        drop = random.uniform(0.1, 0.3)

    # 5% chance of restock
    if random.random() < 0.05:
        restock = random.uniform(5, 10)
        new_weight = current_weight + restock
        event = "restock"
    else:
        new_weight = max(current_weight - drop, 0)

    return new_weight, event


def train_models(history_df):
    """Train Isolation Forest and Linear Regression."""
    X = pd.DataFrame()
    X["delta"] = history_df["weight"].diff().fillna(0)
    X["abs_delta"] = X["delta"].abs()

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X)

    lr = LinearRegression()
    t = np.arange(len(history_df)).reshape(-1, 1)
    w = history_df["weight"].values
    lr.fit(t, w)

    return iso, lr


# -------------------------------
# Main Simulation
# -------------------------------

def run_simulation(days=7):
    """
    Run fridge monitoring simulation.
    Each 'day' in simulation = 1 loop step.
    """
    notifier = EmailNotifier()

    # copy initial stock
    stock = {s: p.copy() for s, p in INITIAL_STOCK.items()}
    histories = {shelf: {prod: pd.DataFrame(columns=["time", "weight", "event"])
                         for prod in products}
                 for shelf, products in SHELVES.items()}

    current_day = datetime(2025, 7, 1)

    for d in range(days):
        print(f"\n=== Day {d+1} ({current_day.strftime('%Y-%m-%d')}) ===")

        for shelf, products in stock.items():
            for product, weight in products.items():
                # simulate new weight
                new_weight, event = simulate_weight_change(weight)
                histories[shelf][product].loc[len(histories[shelf][product])] = [
                    current_day, new_weight, event
                ]
                stock[shelf][product] = new_weight

                # train models if enough history
                if len(histories[shelf][product]) > 10:
                    iso, lr = train_models(histories[shelf][product])

                    # anomaly detection
                    delta = histories[shelf][product]["weight"].diff().iloc[-1]
                    abs_delta = abs(delta)
                    pred = iso.predict([[delta, abs_delta]])[0]
                    anomaly = (pred == -1)

                    # forecasting
                    slope = lr.coef_[0]
                    intercept = lr.intercept_
                    eta = None
                    if slope < 0:
                        t_empty = -intercept / slope
                        eta = max(0, t_empty - len(histories[shelf][product]))

                    # live console output
                    print(f"{shelf} – {product}: {new_weight:.2f}kg "
                          f"({event}, anomaly={anomaly}, eta={eta if eta else '-'})")

                    # add to alerts for email summary
                    if anomaly:
                        notifier.add_alert(
                            f"⚠️ {shelf} – {product}: Unusual drop "
                            f"(Δ={delta:.2f}kg, now {new_weight:.2f}kg)"
                        )
                    if eta and eta < 10:
                        notifier.add_alert(
                            f"📦 {shelf} – {product}: Predicted to empty in {int(eta)} days"
                        )
                    if event == "restock" and new_weight > 1.5 * INITIAL_STOCK[shelf][product]:
                        notifier.add_alert(
                            f"⚠️ {shelf} – {product}: Overstock detected ({new_weight:.2f}kg)"
                        )
                else:
                    print(f"{shelf} – {product}: {new_weight:.2f}kg (training...)")

        # send summary email at the end of the day
        notifier.send_daily_summary(current_day.strftime("%Y-%m-%d"))

        # next simulated day
        current_day += timedelta(days=1)
        time.sleep(1)  # short pause for realism


# -------------------------------
# Run Simulation
# -------------------------------

if __name__ == "__main__":
    run_simulation(days=7)
