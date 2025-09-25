# Smart Industrial Fridge — AI (IF + LR) + GPT Reports + Daily Email Summary
# Streamlit app (ready for Streamlit Community Cloud)

import os
import random
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.text import MIMEText

# --- GPT (optional but recommended) ---
# Works with openai>=1.0.0
try:
    from openai import OpenAI
    _OPENAI_KEY = (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    gpt_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
    GPT_READY = gpt_client is not None
except Exception:
    gpt_client = None
    GPT_READY = False

# -------------------------------
# Config / Constants
# -------------------------------

TIME_PER_STEP_DAYS = 0.5      # 1 step = 0.5 day
OVERSTOCK_FACTOR = 1.8        # overload if current > initial * 1.8
ANOMALY_PROB = 0.10           # chance of abnormal big drop
RESTOCK_PROB = 0.05           # chance of restock
IDLE_PROB = 0.05              # chance of no-change for a product

PRODUCTS = {
    "Dairy":       ["Milk", "Cheese", "Yogurt", "Butter", "Cream"],
    "Meat":        ["Chicken", "Beef", "Fish", "Sausage", "Ham"],
    "Vegetables":  ["Carrots", "Potatoes", "Onions", "Tomatoes", "Lettuce"],
}

# -------------------------------
# Email Notifier
# -------------------------------

class EmailNotifier:
    def __init__(self):
        def secret(name):
            try:
                return st.secrets.get(name, None) if hasattr(st, "secrets") else None
            except Exception:
                return None

        self.smtp_host = secret("SMTP_HOST") or os.getenv("SMTP_HOST")
        self.smtp_port = int(secret("SMTP_PORT") or os.getenv("SMTP_PORT") or 587)
        self.smtp_user = secret("SMTP_USER") or os.getenv("SMTP_USER")
        self.smtp_pass = secret("SMTP_PASS") or os.getenv("SMTP_PASS")
        self.email_to  = secret("EMAIL_TO")  or os.getenv("EMAIL_TO")

        self.alerts_today: List[str] = []  # raw alert lines for the day

    @property
    def configured(self) -> bool:
        return all([self.smtp_host, self.smtp_port, self.smtp_user, self.smtp_pass, self.email_to])

    def add_alert(self, message: str):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.alerts_today.append(f"[{ts}] {message}")

    def _gpt_daily_report(self, day: str) -> str:
        """Convert raw alerts to a professional daily report via GPT (or fallback)."""
        if not self.alerts_today:
            if GPT_READY:
                prompt = (
                    f"Create a concise professional daily fridge report for {day}. "
                    "No stock changes were recorded. Ask if the business was closed. "
                    "Include a clear call-to-action telling the owner to visit the app to answer. "
                    "If they were not closed, list likely causes (sensor issue, misloading, spoilage, inactivity) "
                    "and urge investigation. Keep to 4–6 sentences."
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
                    return f"Daily Report – {day}\nNo stock changes recorded.\n[GPT error: {e}]"
            else:
                return (f"Daily Report – {day}\n"
                        "No stock changes recorded today.\n"
                        "Were you closed? Please visit the Smart Fridge dashboard to answer.\n"
                        "If not closed, possible causes: sensor malfunction, misloading, spoilage, abnormal inactivity.\n"
                        "Please investigate.")

        # We have raw alerts → summarize via GPT if possible
        joined = "\n".join(self.alerts_today)
        if GPT_READY:
            prompt = (
                f"Write a professional daily fridge report for {day}. "
                "Summarize these alerts, prioritize urgent items (<3 days to empty), "
                "and include specific recommendations. Keep it short and actionable (6–8 sentences).\n"
                f"Alerts:\n{joined}"
            )
            try:
                resp = gpt_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=450,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return f"Daily Report – {day}\n{joined}\n[GPT error: {e}]"
        else:
            return f"Daily Report – {day}\n" + joined

    def send_daily_summary(self, day_str: str):
        """Send one GPT-enhanced email per day (fallback to raw if GPT not available)."""
        if not self.configured:
            st.warning("⚠️ Email not configured; daily email skipped.")
            self.alerts_today = []
            return

        body = self._gpt_daily_report(day_str)
        msg = MIMEText(body)
        msg["Subject"] = f"Smart Fridge Daily Report – {day_str}"
        msg["From"] = self.smtp_user
        msg["To"] = self.email_to

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            st.toast(f"📧 Daily report emailed for {day_str}", icon="✉️")
        except Exception as e:
            st.error(f"Email failed: {e}")

        # reset for next day
        self.alerts_today = []

# -------------------------------
# GPT Helpers for UI Reports
# -------------------------------

def gpt_period_report(alert_lines: List[str], title: str, no_change: bool) -> str:
    if no_change:
        if GPT_READY:
            prompt = (
                f"Create a concise professional monitoring note titled '{title}'. "
                "No stock changes were recorded. Ask if the business was closed and include a CTA to visit the app to answer. "
                "If not closed, list likely causes (sensor issue, misloading, spoilage, inactivity) and urge investigation. "
                "4–6 sentences."
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
                return f"**{title}**\nNo stock changes recorded.\n[GPT error: {e}]"
        else:
            return (f"**{title}**\nNo stock changes recorded.\n"
                    "Were you closed? Please visit the app to answer.\n"
                    "If not, possible causes include sensor malfunction, misloading, spoilage, or abnormal inactivity.")

    if not alert_lines:
        return f"**{title}**\nNo alerts in this period."

    joined = "\n".join(alert_lines)
    if GPT_READY:
        prompt = (
            f"Write a professional report titled '{title}'. "
            "Summarize these alerts, prioritize urgent items (<3 days to empty), "
            "and recommend next actions. Keep it concise.\n"
            f"Alerts:\n{joined}"
        )
        try:
            resp = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"**{title}**\n{joined}\n[GPT error: {e}]"
    else:
        return f"**{title}**\n{joined}"

def gpt_no_change_followup() -> str:
    if GPT_READY:
        prompt = (
            "Owner confirmed they were NOT closed on a no-change day. "
            "Write a concise action plan: likely causes (sensor malfunction, misloading, spoilage, abnormal inactivity), "
            "priority checks, and who to notify. 5 bullet points."
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
    else:
        return ("- Inspect load sensors and wiring immediately\n"
                "- Verify recent handling procedures and staff logs\n"
                "- Check for spoilage or temperature incidents\n"
                "- Review camera/logs if available for inactivity\n"
                "- Notify maintenance and shift supervisor")

# -------------------------------
# ML Models
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
# Simulation
# -------------------------------

def init_initial_weights() -> Dict[str, Dict[str, float]]:
    return {shelf: {prod: float(random.randint(20, 40)) for prod in PRODUCTS[shelf]} for shelf in PRODUCTS}

def deep_copy(d: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {k: dict(v) for k, v in d.items()}

def step_dt(start_dt: datetime, step_i: int) -> datetime:
    return start_dt + timedelta(days=step_i * TIME_PER_STEP_DAYS)

def simulate_step(
    fridge: Dict[str, Dict[str, float]],
    histories: Dict[str, Dict[str, pd.DataFrame]],
    initial: Dict[str, Dict[str, float]],
    alerts: List[Dict],
    notifier: EmailNotifier,
    now: datetime
) -> None:
    for shelf, prods in fridge.items():
        for product, w in prods.items():
            # Possibility of no change
            if random.random() < IDLE_PROB:
                new_w = w
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
                    restock = random.uniform(5.0, 10.0)
                    new_w = w + restock
                    txt = f"🔄 Restock | {shelf} – {product} +{restock:.1f} kg | Current: {new_w:.2f} kg | {now:%Y-%m-%d}"
                    alerts.append({"time": now, "type": "restock", "text": txt})
                    notifier.add_alert(txt)

                    over_limit = initial[shelf][product] * OVERSTOCK_FACTOR
                    if new_w > over_limit:
                        txt2 = (f"⚡ Overload | {shelf} – {product} exceeded safe limit "
                                f"({new_w:.2f} kg > {over_limit:.2f} kg) | {now:%Y-%m-%d}")
                        alerts.append({"time": now, "type": "overload", "text": txt2})
                        notifier.add_alert(txt2)
                else:
                    new_w = max(w - total_drop, 0.0)

            fridge[shelf][product] = new_w

            # History append
            if product not in histories[shelf]:
                histories[shelf][product] = pd.DataFrame(columns=["time", "weight", "event"])
            histories[shelf][product].loc[len(histories[shelf][product])] = [now, new_w, event]

            # AI checks (after enough points)
            h = histories[shelf][product]
            if len(h) > 10:
                iso, lr = train_models(h)

                delta = h["weight"].diff().fillna(0.0).iloc[-1]
                X_curr = pd.DataFrame([[delta, abs(delta)]], columns=["delta", "abs_delta"])
                pred = iso.predict(X_curr)[0]
                is_anom = (pred == -1)

                slope = float(lr.coef_[0])
                intercept = float(lr.intercept_)
                eta_days = None
                if slope < 0:
                    t_zero = -intercept / slope
                    steps_left = t_zero - (len(h) - 1)
                    if steps_left >= 0:
                        eta_days = steps_left * TIME_PER_STEP_DAYS

                if is_anom and event != "no_change":
                    txt = (f"⚠️ Anomaly | {shelf} – {product} unusual drop (Δ={delta:.2f} kg) | "
                           f"Current: {new_w:.2f} kg | {now:%Y-%m-%d}")
                    alerts.append({"time": now, "type": "anomaly", "text": txt})
                    notifier.add_alert(txt)

                if eta_days is not None and eta_days < 7.0:
                    txt = (f"📦 Depletion Soon | {shelf} – {product} may empty in {eta_days:.1f} days | "
                           f"Current: {new_w:.2f} kg | {now:%Y-%m-%d}")
                    alerts.append({"time": now, "type": "depletion", "text": txt})
                    notifier.add_alert(txt)

# -------------------------------
# Reporting helpers
# -------------------------------

def filter_alerts(alerts: List[Dict], start_dt: datetime, end_dt: datetime) -> List[str]:
    return [a["text"] for a in alerts if start_dt <= a["time"] < end_dt]

def build_period_reports(alerts: List[Dict], day_dt: datetime, notifier: EmailNotifier) -> Dict[str, str]:
    day_start = datetime(day_dt.year, day_dt.month, day_dt.day)
    day_end = day_start + timedelta(days=1)

    week_start = day_start - timedelta(days=6)
    week_end = day_end

    month_start = datetime(day_dt.year, day_dt.month, 1)
    if day_dt.month == 12:
        month_end = datetime(day_dt.year + 1, 1, 1)
    else:
        month_end = datetime(day_dt.year, day_dt.month + 1, 1)

    d_lines = filter_alerts(alerts, day_start, day_end)
    w_lines = filter_alerts(alerts, week_start, week_end)
    m_lines = filter_alerts(alerts, month_start, month_end)

    t_daily = f"📅 Daily Report — {day_start:%d %B %Y}"
    t_week  = f"📅 Weekly Report — {week_start:%d %b}–{(week_end - timedelta(days=1)):%d %b %Y}"
    t_month = f"📅 Monthly Report — {month_start:%B %Y}"

    daily = gpt_period_report(d_lines, t_daily, no_change=(len(d_lines) == 0))
    weekly = gpt_period_report(w_lines, t_week, no_change=(len(w_lines) == 0))
    monthly = gpt_period_report(m_lines, t_month, no_change=(len(m_lines) == 0))

    # If daily had no changes → send a CTA email (visit app to answer)
    if len(d_lines) == 0:
        if notifier.configured:
            cta = (f"{t_daily}\nNo stock changes recorded.\n"
                   "Were you closed today? Please visit the Smart Fridge dashboard to answer.\n"
                   "If not closed, possible causes: sensor malfunction, misloading, spoilage, or abnormal inactivity. "
                   "Please investigate.")
            try:
                with smtplib.SMTP(notifier.smtp_host, notifier.smtp_port) as server:
                    server.starttls()
                    server.login(notifier.smtp_user, notifier.smtp_pass)
                    msg = MIMEText(cta)
                    msg["Subject"] = "Daily Status: No Changes Detected"
                    msg["From"] = notifier.smtp_user
                    msg["To"] = notifier.email_to
                    server.send_message(msg)
                st.toast("✉️ No-change CTA email sent", icon="✉️")
            except Exception as e:
                st.error(f"CTA email failed: {e}")

    return {"daily": daily, "weekly": weekly, "monthly": monthly}

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Smart Fridge AI (Reports + Email)", layout="wide")
st.title("🥶 Smart Industrial Fridge — AI + GPT Reports (Daily Email)")

with st.sidebar:
    st.subheader("Simulation Window")
    start_date = st.date_input("Start date", value=datetime(2025, 7, 1).date())
    end_date = st.date_input("End date", value=datetime(2025, 8, 1).date())
    st.caption("1 step = 0.5 days • A 31-day span ≈ 62 steps")

    st.subheader("Status")
    email_status = "✅ configured" if EmailNotifier().configured else "⚪ not set"
    st.caption(f"Email: {email_status} • GPT: {'✅' if GPT_READY else '⚪'}")

run_btn = st.button("▶️ Run Simulation")
reset_btn = st.button("🔄 Reset")

# Session state init/reset
def reset_state():
    st.session_state.initial = init_initial_weights()
    st.session_state.fridge = deep_copy(st.session_state.initial)
    st.session_state.histories = {shelf: {} for shelf in PRODUCTS}
    st.session_state.alerts: List[Dict] = []
    st.session_state.notifier = EmailNotifier()
    st.session_state.last_day_dt = None

if "initial" not in st.session_state:
    reset_state()

if reset_btn:
    reset_state()
    st.success("State reset. New randomized initial weights (20–40 kg).")

# Show initial weights & overstock thresholds
with st.expander("Initial weights & overstock thresholds"):
    rows = []
    for shelf, prods in st.session_state.initial.items():
        for prod, w0 in prods.items():
            rows.append({
                "Shelf": shelf,
                "Product": prod,
                "Initial (kg)": round(w0, 2),
                "Overstock if > (kg)": round(w0 * OVERSTOCK_FACTOR, 2)
            })
    st.dataframe(pd.DataFrame(rows).sort_values(["Shelf", "Product"]),
                 use_container_width=True, height=260)

# Run simulation
if run_btn:
    sd = datetime.combine(start_date, datetime.min.time())
    ed = datetime.combine(end_date, datetime.min.time())
    days = max(1, (ed - sd).days)
    steps = max(1, int(days / TIME_PER_STEP_DAYS))  # e.g., 31 / 0.5 = 62

    for s in range(steps):
        now = step_dt(sd, s)
        simulate_step(
            st.session_state.fridge,
            st.session_state.histories,
            st.session_state.initial,
            st.session_state.alerts,
            st.session_state.notifier,
            now
        )

        # If we've crossed a whole day boundary, send the daily email
        if (now.hour, now.minute) == (0, 0):  # step_dt uses whole days, so this aligns at midnight
            st.session_state.notifier.send_daily_summary(now.strftime("%Y-%m-%d"))

    st.session_state.last_day_dt = step_dt(sd, steps - 1)
    st.success(f"✅ Simulation completed: {sd:%d %b %Y} → {st.session_state.last_day_dt:%d %b %Y}")

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
st.subheader("Recent Alerts")
if not st.session_state.alerts:
    st.info("No alerts yet.")
else:
    for a in st.session_state.alerts[-20:]:
        st.write(a["text"])

# Reports
st.subheader("📊 GPT Reports")
if st.session_state.last_day_dt:
    reports = build_period_reports(st.session_state.alerts, st.session_state.last_day_dt, st.session_state.notifier)
    st.markdown(reports["daily"])
    st.markdown("---")
    st.markdown(reports["weekly"])
    st.markdown("---")
    st.markdown(reports["monthly"])
else:
    st.info("Run the simulation to generate reports.")

# Owner follow-up for no-change days
st.subheader("Owner Follow-Up (no-change days)")
st.caption("If the daily report showed no changes, confirm whether you were closed. Your answer triggers GPT next steps.")
answer = st.text_input("Were you closed today? (yes/no)", value="")
if answer.strip().lower() in {"yes", "no"} and st.session_state.last_day_dt:
    if answer.strip().lower() == "no":
        follow = gpt_no_change_followup()
        st.error(follow)

        # email the follow-up
        notifier = EmailNotifier()
        if notifier.configured:
            try:
                with smtplib.SMTP(notifier.smtp_host, notifier.smtp_port) as server:
                    server.starttls()
                    server.login(notifier.smtp_user, notifier.smtp_pass)
                    msg = MIMEText(follow)
                    msg["Subject"] = f"Follow-Up: No-change day not closed — {st.session_state.last_day_dt:%d %b %Y}"
                    msg["From"] = notifier.smtp_user
                    msg["To"] = notifier.email_to
                    server.send_message(msg)
                st.toast("✉️ Follow-up emailed", icon="✉️")
            except Exception as e:
                st.error(f"Follow-up email failed: {e}")
    else:
        st.success("Thanks — no action needed if the site was closed.")
