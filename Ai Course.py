# app.py
import os
import smtplib
import pandas as pd
import streamlit as st
from datetime import timedelta
from email.mime.text import MIMEText

from sklearn.ensemble import IsolationForest
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Hugging Face Chronos setup
# -------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # put in Streamlit secrets
chronos_model = "amazon/chronos-t5-small"

tokenizer = AutoTokenizer.from_pretrained(chronos_model, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(chronos_model, use_auth_token=HF_TOKEN)


def forecast_eta(weights):
    """Forecast using Hugging Face Chronos model."""
    try:
        # Last 30 values (string input)
        input_str = " ".join([str(w) for w in weights[-30:]])
        inputs = tokenizer(input_str, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=10)
        forecast_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return f"Forecast: {forecast_text}"
    except Exception as e:
        return f"(forecast failed: {e})"


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
            st.success(f"📧 Email sent: {subject}")
        except Exception as e:
            st.error(f"❌ Email failed: {e}")


# -------------------------------
# Reporting helpers
# -------------------------------
def detect_anomalies(weights):
    """Detect anomalies using IsolationForest."""
    try:
        if len(weights) < 5:
            return "Not enough data"
        iso = IsolationForest(contamination=0.1, random_state=42)
        X = pd.DataFrame(weights, columns=["w"])
        preds = iso.fit_predict(X)
        anomalies = (preds == -1).sum()
        return f"{anomalies} anomalies detected"
    except Exception as e:
        return f"(anomaly check failed: {e})"


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
        anomaly_flag = detect_anomalies(weights)
        lines.append(
            f"{shelf} – {product}: Drops={drops}, Refills={refills}, "
            f"Initial={initial:.2f}kg, Current={current:.2f}kg, ETA={eta}, Anomaly={anomaly_flag}"
        )
    return "\n".join(lines)


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
        anomaly_flag = detect_anomalies(weights)
        lines.append(
            f"{shelf} – {product}: Drops={drops}, Refills={refills}, "
            f"Initial={initial:.2f}kg, Current={current:.2f}kg, ETA={eta}, Anomaly={anomaly_flag}"
        )
    return "\n".join(lines)


# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("Smart Industrial Fridge Monitor ❄️")
    st.markdown("Prototype using Hugging Face Chronos + IsolationForest + Email Reports")

    # Load dataset
    df = pd.read_excel("fridge_data.xlsx")
    df["date"] = pd.to_datetime(df["date"])

    notifier = EmailNotifier()

    # Date selector
    start_date = st.date_input("Start date", df["date"].min().date())
    end_date = st.date_input("End date", df["date"].max().date())

    if st.button("▶️ Run Monitoring"):
        current_date = start_date
        half_day_flag = True

        while current_date <= end_date:
            day_data = df[df["date"] == pd.to_datetime(current_date)]

            # Half-day vs full-day
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

    if st.button("🔄 Reset"):
        st.experimental_rerun()


if __name__ == "__main__":
    main()
