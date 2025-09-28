# app.py
import os
import smtplib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from huggingface_hub import InferenceClient
from sklearn.linear_model import LinearRegression

# -------------------------------
# Hugging Face Setup
# -------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
forecast_model = "time-series-foundation-models/Lag-Llama"   # for forecasting
text_model = "gpt2"  # small text generator
client = InferenceClient(token=HF_TOKEN)

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
            st.success(f"Email sent: {subject}")
        except Exception as e:
            st.error(f"Email failed: {e}")

# -------------------------------
# AI Helpers
# -------------------------------
def forecast_eta(weights):
    """Try Hugging Face Lag-Llama. Fallback to Linear Regression."""
    try:
        input_str = ",".join([str(w) for w in weights[-30:]])
        resp = client.text_generation(
            model=forecast_model,
            prompt=f"Forecast remaining days until stock reaches ~0. Data: {input_str}",
            max_new_tokens=50
        )
        return resp.strip()
    except Exception:
        # fallback Linear Regression
        try:
            X = np.arange(len(weights)).reshape(-1, 1)
            y = np.array(weights)
            lr = LinearRegression().fit(X, y)
            slope = lr.coef_[0]
            intercept = lr.intercept_
            if slope < 0:
                t_empty = -intercept / slope
                eta = max(0, t_empty - len(weights))
                return f"{int(eta)} days (LR)"
            return "Stable (LR)"
        except Exception as e:
            return f"(forecast failed: {e})"

def llm_rephrase(text):
    """Use GPT2 to make reports more natural."""
    try:
        resp = client.text_generation(
            model=text_model,
            prompt=f"Rephrase this fridge monitoring report in a clear, human style:\n{text}",
            max_new_tokens=120
        )
        return resp.strip()
    except Exception:
        return text  # fallback raw text

# -------------------------------
# Reporting
# -------------------------------
def weekly_report(df, start_date, end_date):
    subset = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    lines = [f"Weekly Report ({start_date.date()} → {end_date.date()})"]
    for (shelf, product), group in subset.groupby(["shelf", "product"]):
        weights = group["weight"].tolist()
        drops = (group["event"] == "abnormal").sum()
        refills = (group["event"] == "restock").sum()
        initial = group.iloc[0]["weight"]
        current = group.iloc[-1]["weight"]
        eta = forecast_eta(weights)
        lines.append(
            f"{shelf} – {product}: Drops={drops}, Refills={refills}, "
            f"Initial={initial:.2f}kg, Current={current:.2f}kg, ETA={eta}"
        )
    return llm_rephrase("\n".join(lines))

def monthly_report(df, start_date, end_date):
    subset = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    lines = [f"Monthly Report ({start_date.date()} → {end_date.date()})"]
    for (shelf, product), group in subset.groupby(["shelf", "product"]):
        weights = group["weight"].tolist()
        drops = (group["event"] == "abnormal").sum()
        refills = (group["event"] == "restock").sum()
        initial = group.iloc[0]["weight"]
        current = group.iloc[-1]["weight"]
        eta = forecast_eta(weights)
        lines.append(
            f"{shelf} – {product}: Drops={drops}, Refills={refills}, "
            f"Initial={initial:.2f}kg, Current={current:.2f}kg, ETA={eta}"
        )
    return llm_rephrase("\n".join(lines))

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("Smart Industrial Fridge Monitor")
    st.markdown("Hugging Face (Lag-Llama + GPT2) + Email Reports")

    df = pd.read_excel("fridge_data.xlsx")
    df["date"] = pd.to_datetime(df["date"])

    notifier = EmailNotifier()

    # Date selector
    start_date = st.date_input("Start date", df["date"].min().date())
    end_date = st.date_input("End date", df["date"].max().date())

    if st.button("Run Monitoring"):
        current_date = start_date
        half_day_flag = True

        while current_date <= end_date:
            day_data = df[df["date"] == pd.to_datetime(current_date)]

            # Zero-stock alerts
            zero_stock = day_data[day_data["weight"] <= 0]
            for _, row in zero_stock.iterrows():
                notifier.send_email(
                    f"Zero Stock Alert – {row['product']} on {current_date}",
                    f"{row['shelf']} – {row['product']} has reached zero stock on {current_date}."
                )

            # Half-day vs full-day report
            if half_day_flag:
                notifier.send_email(
                    f"Half-Day Report – {current_date}",
                    f"Snapshot for {current_date}\n\n{day_data.to_string(index=False)}"
                )
                half_day_flag = False
            else:
                notifier.send_email(
                    f"Full-Day Report – {current_date}",
                    f"Snapshot for {current_date}\n\n{day_data.to_string(index=False)}"
                )
                half_day_flag = True

            # Weekly summary
            if (pd.to_datetime(current_date) - df["date"].min()).days % 7 == 6:
                report = weekly_report(df, pd.to_datetime(current_date) - timedelta(days=6), pd.to_datetime(current_date))
                notifier.send_email(f"Weekly Summary – {current_date}", report)

            # Monthly summary
            if pd.to_datetime(current_date) == df["date"].max():
                report = monthly_report(df, df["date"].min(), df["date"].max())
                notifier.send_email(f"Monthly Summary – {current_date}", report)

            current_date += timedelta(days=1)

    if st.button("Reset"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
