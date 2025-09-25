# -------------------------------
# Smart Industrial Fridge — AI + GPT Monitoring (Calendar + Reports)
# -------------------------------

import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

import smtplib
from email.mime.text import MIMEText

# --- GPT (optional) ---
try:
    # Works with openai>=1.0 or legacy. We'll try modern first.
    from openai import OpenAI
    _OPENAI_KEY = (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    gpt_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
    GPT_READY = gpt_client is not None
except Exception:
    GPT_READY = False
    gpt_client = None

# -------------------------------
# Config
# -------------------------------

TIME_PER_STEP_DAYS = 0.5        # 1 simulation step = 0.5 day
OVERSTOCK_FACTOR = 1.8          # overload threshold multiplier
ANOMALY_PROB = 0.10             # probability of a big drop event
RESTOCK_PROB = 0.05             # probability of a restock event
IDLE_PROB = 0.05                # probability of "no change" for a product in a step

PRODUCTS = {
    "Dairy":       ["Milk", "Cheese", "Yogurt", "Butter", "Cream"],
    "Meat":        ["Chicken", "Beef", "Fish", "Sausage", "Ham"],
    "Vegetables":  ["Carrots", "Potatoes", "Onions", "Tomatoes", "Lettuce"],
}

# -------------------------------
# Email notifier
# -------------------------------

class Notifier:
    def __init__(self):
        def secret(name):  # secrets first, fallback env
            try:
                return st.secrets.get(name, None) if hasattr(st, "secrets") else None
            except Exception:
                return None
        self.smtp_host = secret("SMTP_HOST") or os.getenv("SMTP_HOST")
        self.smtp_port = secret("SMTP_PORT") or os.getenv("SMTP_PORT")
        self.smtp_user = secret("SMTP_USER") or os.getenv("SMTP_USER")
        self.smtp_pass = secret("SMTP_PASS") or os.getenv("SMTP_PASS")
        self.email_to  = secret("EMAIL_TO")  or os.getenv("EMAIL_TO")

    @property
    def configured(self) -> bool:
        return all([self.smtp_host, self.smtp_port, self.smtp_user, self.smtp_pass, self.email_to])

    def send_email(self, subject: str, body: str) -> bool:
        if not self.configured:
            st.warning("⚠️ Email not configured; skipping email send.")
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
            st.toast("📧 Email sent", icon="✉️")
            return True
        except Exception as e:
            st.error(f"Email failed: {e}")
            return False

# -------------------------------
# GPT helpers
# -------------------------------

def gpt_generate_report(alert_lines: List[str], title: str, no_change: bool) -> str:
    """
    Build a clear report with GPT. If no_change is True, ask whether they were closed and include CTA.
    """
    if not GPT_READY:
        if no_change:
            return (f"**{title}**\nNo stock changes recorded.\n"
                    "Were you closed? If not, possible causes include sensor malfunction, misloading, or abnormal inactivity. "
                    "Please visit the dashboard to confirm and investigate.")
        else:
            return (f"**{title}**\n" +
                    ("No alerts in this period." if not alert_lines else "\n".join(alert_lines)))

    if no_change:
        prompt = (
            f"Create a concise professional monitoring note titled '{title}'. "
            "No stock changes were recorded in this period. "
            "Ask if the business was closed. Include a clear call-to-action telling the owner to visit the app to answer. "
            "If they were not closed, list likely causes (sensor issue, misloading, spoilage, abnormal inactivity) "
            "and urge investigation. Keep to 4–6 sentences."
        )
    else:
        joined = "\n".join(alert_lines)
        prompt = (
            f"Write a professional monitoring report titled '{title}'. "
            "Summarize the following alerts, prioritize urgent items (e.g., <3 days to empty), and add specific recommendations. "
            "Keep it short and actionable (6–8 sentences). Alerts:\n"
            f"{joined}"
        )

    try:
        resp = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT error: {e}]"

def gpt_followup_no_change(owner_answer: str) -> str:
    """
    If owner said 'no' (not closed), generate likely causes and next steps.
    """
    if not GPT_READY:
        return ("Owner reported they were not closed. Possible causes: sensor malfunction, misloaded items, "
                "spoilage, or operational inactivity. Action: inspect sensors and load cells, check logs and camera "
                "(if available), and verify handling procedures.")
    prompt = (
        "Owner confirmed they were NOT closed today, yet no stock changes were recorded. "
        "Write a concise action plan: likely causes (sensor malfunction, misloading, spoilage, abnormal inactivity), "
        "priority checks to run, and who should be notified. Keep to 5 bullet points."
    )
    try:
        resp = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT error: {e}]"

# -------------------------------
# Model training
# -------------------------------

def train_models(history_df: pd.DataFrame) -> Tuple[IsolationForest, LinearRegression]:
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
# Simulation helpers
# -------------------------------

def init_initial_weights() -> Dict[str, Dict[str, float]]:
    return {shelf: {prod: float(random.randint(20, 40)) for prod in PRODUCTS[shelf]} for shelf in PRODUCTS}

def deep_copy_fridge(weights: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {s: dict(p) for s, p in weights.items()}

def step_datetime(start_date: datetime, step_index: int) -> datetime:
    return start_date + timedelta(days=step_index * TIME_PER_STEP_DAYS)

def simulate_one_step(
    fridge_state: Dict[str, Dict[str, float]],
    histories: Dict[str, Dict[str, pd.DataFrame]],
    initial_weights: Dict[str, Dict[str, float]],
    alerts: List[Dict],
    notifier: Notifier,
    now_dt: datetime
) -> None:
    """
    Simulate consumption/restock for all products, update histories, run AI, and push alerts (with timestamps).
    """
    for shelf, products in fridge_state.items():
        for product, curr_weight in products.items():
            # Idle possibility: intentionally no change (to enable no-change days)
            if random.random() < IDLE_PROB:
                new_weight = curr_weight
                event = "no_change"
            else:
                # Consumption
                normal_drop = random.uniform(0.1, 0.3)
                if random.random() < ANOMALY_PROB:
                    abnormal_drop = random.uniform(2.0, 5.0)
                    total_drop = normal_drop + abnormal_drop
                    event = "abnormal"
                else:
                    total_drop = normal_drop
                    event = "normal"

                # Restock possibility
                if random.random() < RESTOCK_PROB:
                    restock_amt = random.uniform(5.0, 10.0)
                    new_weight = curr_weight + restock_amt
                    # Restock alert
                    msg = (f"🔄 Restock | {shelf} – {product} +{restock_amt:.1f} kg | "
                           f"Current: {new_weight:.2f} kg | {now_dt:%Y-%m-%d}")
                    alerts.append({"time": now_dt, "type": "restock", "text": msg})
                    notifier.send_email("Fridge Restock", msg)

                    # Overstock check
                    over_limit = initial_weights[shelf][product] * OVERSTOCK_FACTOR
                    if new_weight > over_limit:
                        msg2 = (f"⚡ Overload | {shelf} – {product} exceeded safe limit "
                                f"({new_weight:.2f} kg > {over_limit:.2f} kg) | {now_dt:%Y-%m-%d}")
                        alerts.append({"time": now_dt, "type": "overload", "text": msg2})
                        notifier.send_email("Fridge Overstock Alert", msg2)
                else:
                    new_weight = max(curr_weight - total_drop, 0.0)

            # Update state
            fridge_state[shelf][product] = new_weight

            # History init + append
            if product not in histories[shelf]:
                histories[shelf][product] = pd.DataFrame(columns=["time", "weight", "event"])
            histories[shelf][product].loc[len(histories[shelf][product])] = [now_dt, new_weight, event]

            # AI checks
            hist_df = histories[shelf][product]
            if len(hist_df) > 10:
                iso, lr = train_models(hist_df)

                delta = hist_df["weight"].diff().fillna(0.0).iloc[-1]
                X_curr = pd.DataFrame([[delta, abs(delta)]], columns=["delta", "abs_delta"])
                pred = iso.predict(X_curr)[0]  # 1 normal, -1 anomaly
                is_anomaly = (pred == -1)

                # Forecast to empty
                slope = float(lr.coef_[0])
                intercept = float(lr.intercept_)
                eta_days = None
                if slope < 0:
                    t_zero = -intercept / slope
                    steps_remaining = t_zero - (len(hist_df) - 1)
                    if steps_remaining >= 0:
                        eta_days = steps_remaining * TIME_PER_STEP_DAYS

                # Alerts (with current weight)
                if is_anomaly and event != "no_change":
                    amsg = (f"⚠️ Anomaly | {shelf} – {product} unusual drop (Δ={delta:.2f} kg) | "
                            f"Current: {new_weight:.2f} kg | {now_dt:%Y-%m-%d}")
                    alerts.append({"time": now_dt, "type": "anomaly", "text": amsg})
                    notifier.send_email("Fridge Anomaly", amsg)

                if eta_days is not None and eta_days < 7.0:
                    dmsg = (f"📦 Depletion Soon | {shelf} – {product} may empty in {eta_days:.1f} days | "
                            f"Current: {new_weight:.2f} kg | {now_dt:%Y-%m-%d}")
                    alerts.append({"time": now_dt, "type": "depletion", "text": dmsg})
                    notifier.send_email("Fridge Depletion Warning", dmsg)

# -------------------------------
# Reporting helpers
# -------------------------------

def filter_alerts_by_range(alerts: List[Dict], start_dt: datetime, end_dt: datetime) -> List[str]:
    return [a["text"] for a in alerts if start_dt <= a["time"] < end_dt]

def make_reports(alerts: List[Dict], day_dt: datetime, notifier: Notifier) -> Dict[str, str]:
    """
    Build daily/weekly/monthly reports with GPT (or fallback) and send "no-change" email with CTA.
    """
    # Ranges
    day_start = datetime(day_dt.year, day_dt.month, day_dt.day)
    day_end   = day_start + timedelta(days=1)

    week_start = day_start - timedelta(days=6)
    week_end   = day_end

    month_start = datetime(day_dt.year, day_dt.month, 1)
    # next month start:
    if day_dt.month == 12:
        month_end = datetime(day_dt.year + 1, 1, 1)
    else:
        month_end = datetime(day_dt.year, day_dt.month + 1, 1)

    # Filter texts
    daily_lines   = filter_alerts_by_range(alerts, day_start, day_end)
    weekly_lines  = filter_alerts_by_range(alerts, week_start, week_end)
    monthly_lines = filter_alerts_by_range(alerts, month_start, month_end)

    # Titles with dates
    title_daily   = f"📅 Daily Report — {day_start:%d %B %Y}"
    title_weekly  = f"📅 Weekly Report — {week_start:%d %b}–{(week_end - timedelta(days=1)):%d %b %Y}"
    title_monthly = f"📅 Monthly Report — {month_start:%B %Y}"

    # Build texts via GPT (or fallback)
    daily_text   = gpt_generate_report(daily_lines, title_daily,  no_change=(len(daily_lines) == 0))
    weekly_text  = gpt_generate_report(weekly_lines, title_weekly, no_change=(len(weekly_lines) == 0))
    monthly_text = gpt_generate_report(monthly_lines, title_monthly, no_change=(len(monthly_lines) == 0))

    # If daily had no changes, email a CTA asking them to visit app and answer
    if len(daily_lines) == 0:
        cta_email = (
            f"{title_daily}\n"
            "No stock changes were recorded today.\n"
            "Were you closed today? Please visit the Smart Fridge dashboard to answer.\n"
            "If you were not closed, possible causes include sensor malfunction, misloading, spoilage, or abnormal inactivity. "
            "Please investigate."
        )
        notifier.send_email("Daily Status: No Changes Detected", cta_email)

    return {"daily": daily_text, "weekly": weekly_text, "monthly": monthly_text}

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Smart Fridge AI (Calendar Reports)", layout="wide")
st.title("🥶 Smart Industrial Fridge — AI + GPT (Calendar & Reports)")

with st.sidebar:
    st.subheader("Simulation Window")
    start_date = st.date_input("Start date", value=datetime(2025, 7, 1).date())
    end_date   = st.date_input("End date",   value=datetime(2025, 8, 1).date())
    st.caption("Tip: 1 step = 0.5 days, so a 31-day month ≈ 62 steps.")
    st.subheader("Email & GPT")
    st.caption(f"Email: {'✅ configured' if Notifier().configured else '⚪ not set'} • GPT: {'✅' if GPT_READY else '⚪'}")

run_btn = st.button("▶️ Run Simulation")
reset_btn = st.button("🔄 Reset")

# Session state
def reset_state():
    st.session_state.initial_weights = init_initial_weights()
    st.session_state.fridge_state = deep_copy_fridge(st.session_state.initial_weights)
    st.session_state.histories = {shelf: {} for shelf in PRODUCTS}
    st.session_state.alerts = []
    st.session_state.last_day_dt = None
    st.session_state.notifier = Notifier()

if "initial_weights" not in st.session_state:
    reset_state()

if reset_btn:
    reset_state()
    st.success("Simulation state reset. New randomized initial weights (20–40 kg).")

# Show initial weights & thresholds
with st.expander("Initial weights & overstock thresholds"):
    rows = []
    for shelf, prods in st.session_state.initial_weights.items():
        for prod, w0 in prods.items():
            rows.append({"Shelf": shelf, "Product": prod, "Initial (kg)": round(w0, 2),
                         "Overstock if > (kg)": round(w0 * OVERSTOCK_FACTOR, 2)})
    df_init = pd.DataFrame(rows).sort_values(["Shelf", "Product"])
    st.dataframe(df_init, use_container_width=True, height=260)

# Run simulation
if run_btn:
    sd = datetime.combine(start_date, datetime.min.time())
    ed = datetime.combine(end_date, datetime.min.time())

    total_days = (ed - sd).days
    total_steps = max(1, int(total_days / TIME_PER_STEP_DAYS))  # e.g., 31 days / 0.5 = 62 steps

    for s in range(total_steps):
        current_dt = step_datetime(sd, s)
        simulate_one_step(
            st.session_state.fridge_state,
            st.session_state.histories,
            st.session_state.initial_weights,
            st.session_state.alerts,
            st.session_state.notifier,
            current_dt
        )

    st.session_state.last_day_dt = step_datetime(sd, total_steps - 1)
    st.success(f"✅ Simulation finished: {sd:%d %b %Y} → {st.session_state.last_day_dt:%d %b %Y}")

# Charts per shelf
cols = st.columns(3)
for i, shelf in enumerate(PRODUCTS.keys()):
    with cols[i]:
        st.subheader(f"{shelf} shelf")
        combined = []
        for prod, dfh in st.session_state.histories[shelf].items():
            if not dfh.empty:
                t = dfh.copy()
                t["product"] = prod
                combined.append(t)
        if combined:
            dfc = pd.concat(combined)
            pivot = dfc.pivot(index="time", columns="product", values="weight")
            st.line_chart(pivot, height=280)
        else:
            st.info("No data yet. Run the simulation.")

# Alerts feed
st.subheader("Alerts")
if not st.session_state.alerts:
    st.info("No alerts yet.")
else:
    for a in st.session_state.alerts[-20:]:
        st.write(a["text"])

# Reports (Daily / Weekly / Monthly)
st.subheader("📊 GPT Reports")
if st.session_state.last_day_dt:
    reports = make_reports(st.session_state.alerts, st.session_state.last_day_dt, st.session_state.notifier)
    st.markdown(reports["daily"])
    st.markdown("---")
    st.markdown(reports["weekly"])
    st.markdown("---")
    st.markdown(reports["monthly"])
else:
    st.info("Run the simulation to generate reports.")

# Owner follow-up for NO-CHANGE days
st.subheader("Owner Follow-Up (No-Change Days)")
st.caption("If the daily report showed no changes, confirm whether you were closed. Your answer will drive GPT next steps.")
owner_answer = st.text_input("Were you closed today? (yes/no)", value="")
if owner_answer.strip().lower() in {"yes", "no"} and st.session_state.last_day_dt:
    if owner_answer.strip().lower() == "no":
        follow = gpt_followup_no_change(owner_answer)
        st.error(follow)
        # email the follow-up
        Notifier().send_email(
            f"Follow-Up: No-change day not closed — {st.session_state.last_day_dt:%d %b %Y}",
            follow
        )
    else:
        st.success("Thanks. No action needed if you were closed.")
