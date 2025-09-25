# app.py
import os
import smtplib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from email.mime.text import MIMEText
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

    def send_email(self, subject: str, body: str):
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = self.email_to
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            print(f"📧 Email sent: {subject}")
        except Exception as e:
            print("❌ Email failed:", e)

# -------------------------------
# Helpers
# -------------------------------
def train_models(group):
    X = pd.DataFrame()
    X["delta"] = group["weight"].diff().fillna(0)
    X["abs_delta"] = X["delta"].abs()
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X)
    lr = LinearRegression()
    t = np.arange(len(group)).reshape(-1, 1)
    w = group["weight"].values
    lr.fit(t, w)
    return iso, lr

def get_eta(lr, group):
    slope = lr.coef_[0]
    intercept = lr.intercept_
    if slope < 0:
        t_empty = -intercept / slope
        return max(0, t_empty - len(group))
    return None

def weekly_report(df, start_date, end_date):
    subset = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    lines = [f"Weekly Report ({start_date} → {end_date})"]
    for (shelf, product), group in subset.groupby(["shelf", "product"]):
        drops = (group["event"] == "abnormal").sum()
        refills = (group["event"] == "restock").sum()
        initial = group.iloc[0]["weight"]
        current = group.iloc[-1]["weight"]
        _, lr = train_models(group)
        eta = get_eta(lr, group)
        eta_str = f"{int(eta)} days" if eta else "-"
        lines.append(
            f"{shelf} – {product}: Drops={drops}, Refills={refills}, "
            f"Initial={initial:.2f}kg, Current={current:.2f}kg, ETA={eta_str}"
        )
    return "\n".join(lines)

def monthly_report(df, start_date, end_date):
    subset = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    lines = [f"Monthly Report ({start_date} → {end_date})"]
    for (shelf, product), group in subset.groupby(["shelf", "product"]):
        drops = (group["event"] == "abnormal").sum()
        refills = (group["event"] == "restock").sum()
        initial = group.iloc[0]["weight"]
        current = group.iloc[-1]["weight"]
        _, lr = train_models(group)
        eta = get_eta(lr, group)
        eta_str = f"{int(eta)} days" if eta else "-"
        lines.append(
            f"{shelf} – {product}: Drops={drops}, Refills={refills}, "
            f"Initial={initial:.2f}kg, Current={current:.2f}kg, ETA={eta_str}"
        )
    return "\n".join(lines)

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("Smart Fridge Monitoring – Commercial Prototype")
    st.markdown("Twice-daily reports, plus weekly and monthly summaries.")

    df = pd.read_excel("fridge_data.xlsx")
    df["date"] = pd.to_datetime(df["date"])

    notifier = EmailNotifier()
    start_date = df["date"].min()
    end_date = df["date"].max()

    current_date = start_date
    half_day_flag = True

    while current_date <= end_date:
        st.subheader(f"📅 Day: {current_date.strftime('%Y-%m-%d')}")
        day_data = df[df["date"] == current_date]
        st.write(day_data[["shelf", "product", "weight", "event"]])

        # Half-day report
        if half_day_flag:
            notifier.send_email(
                f"Half-Day Report – {current_date.strftime('%Y-%m-%d')}",
                f"Half-day snapshot:\n\n{day_data.to_string(index=False)}"
            )
            half_day_flag = False
        else:
            notifier.send_email(
                f"Full-Day Report – {current_date.strftime('%Y-%m-%d')}",
                f"Full-day snapshot:\n\n{day_data.to_string(index=False)}"
            )
            half_day_flag = True

        # Weekly summary
        if (current_date - start_date).days % 7 == 6:
            report = weekly_report(df, current_date - timedelta(days=6), current_date)
            notifier.send_email(f"Weekly Summary – {current_date}", report)

        # Monthly summary
        if current_date == end_date:
            report = monthly_report(df, start_date, end_date)
            notifier.send_email(f"Monthly Summary – {end_date}", report)

        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
