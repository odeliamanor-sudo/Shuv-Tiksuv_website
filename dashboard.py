import plotly
st.write("Plotly version:", plotly.__version__)


st.write("Python:", sys.version)
import numpy as np
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sim_core import run_replications, run_one_week_with_snapshots

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Shuv-Tikshuv Control Room",
    page_icon="🛰️",
    layout="wide",
)

# ----------------------------
# Minimal neon CSS
# ----------------------------
st.markdown("""
<style>
.block-container{padding-top:1.2rem; padding-bottom:2rem;}
h1, h2, h3 {letter-spacing:0.2px;}
.glow {
  background: linear-gradient(135deg, rgba(122,162,255,0.20), rgba(255,122,198,0.10));
  border: 1px solid rgba(122,162,255,0.25);
  box-shadow: 0 0 25px rgba(122,162,255,0.12);
  border-radius: 18px;
  padding: 16px 18px;
}
.kpi {
  background: rgba(13,18,34,0.9);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 14px;
}
.kpi .label {opacity:0.8; font-size:0.9rem;}
.kpi .value {font-size:1.6rem; font-weight:700; margin-top:4px;}
.small {opacity:0.75; font-size:0.85rem;}
hr {border-color: rgba(255,255,255,0.08);}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="glow">
  <h1 style="margin:0;">🛰️ Shuv-Tikshuv Control Room</h1>
  <div class="small">Dashboard + Animation Playback of your simulation (fault / train+join / senior) — built for the bonus ✨</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ----------------------------
# Sidebar controls (What-If)
# ----------------------------
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    reps = st.slider("Replications (weeks)", 10, 120, 50, 5)
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=2025, step=1)

    st.subheader("Channel + Behavior")
    p_whatsapp = st.slider("P(WhatsApp)", 0.0, 1.0, 0.40, 0.01)
    p_wa_fail = st.slider("P(WA→Phone fail)", 0.0, 1.0, 0.20, 0.01)
    p_escalate = st.slider("P(Transfer to Senior)", 0.0, 1.0, 0.20, 0.01)

    st.subheader("Playback")
    snap_every = st.selectbox("Snapshot every (minutes)", [5, 10, 15, 20], index=1)

    st.divider()
    run_btn = st.button("▶️ Run / Refresh", type="primary")

params = dict(p_whatsapp=p_whatsapp, p_wa_fail=p_wa_fail, p_escalate=p_escalate)

# ----------------------------
# Cached compute
# ----------------------------
@st.cache_data(show_spinner=True)
def cached_replications(reps_, seed_, params_):
    return run_replications(reps_, seed0=int(seed_), **params_)

@st.cache_data(show_spinner=True)
def cached_one_week(seed_, snap_every_, params_):
    res, snap_df = run_one_week_with_snapshots(seed=int(seed_), snapshot_every_min=int(snap_every_), **params_)
    return res, snap_df

if run_btn:
    cached_replications.clear()
    cached_one_week.clear()

agg = cached_replications(reps, seed, params)
one_week_res, snap_df = cached_one_week(seed, snap_every, params)

# ----------------------------
# KPI strip
# ----------------------------
ab = agg["Abandon_hour_avg_per_day"]
idle_hour = agg["idle_hour_avg"]
idle_group = agg["idle_group_avg"]

kpi1 = float(np.mean(ab))
kpi2 = float(np.max(ab))
kpi3 = float(np.mean(idle_hour))
kpi4 = float(np.max(idle_hour))

c1, c2, c3, c4 = st.columns(4)
for col, label, val, suf in [
    (c1, "Avg abandonments / hour (per day)", kpi1, ""),
    (c2, "Peak abandonments hour", kpi2, ""),
    (c3, "Avg idle % (all hours)", kpi3, "%"),
    (c4, "Peak idle % (worst hour)", kpi4, "%"),
]:
    col.markdown(f"""
    <div class="kpi">
      <div class="label">{label}</div>
      <div class="value">{val:,.2f}{suf}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Small tech charts row
# ----------------------------
colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])

# Abandonments hourly
fig_ab = go.Figure()
fig_ab.add_trace(go.Scatter(
    x=list(range(24)), y=ab, mode="lines+markers",
    line=dict(width=3),
    marker=dict(size=7),
    name="Abandon"
))
fig_ab.update_layout(
    title="Abandonments by Hour (avg/day)",
    height=260, margin=dict(l=10, r=10, t=45, b=20),
    xaxis=dict(dtick=2), yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
)
colA.plotly_chart(fig_ab, use_container_width=True)

# Idle by hour
fig_idle = go.Figure()
fig_idle.add_trace(go.Bar(x=list(range(24)), y=idle_hour, name="Idle %"))
fig_idle.update_layout(
    title="Idle % by Hour",
    height=260, margin=dict(l=10, r=10, t=45, b=20),
    xaxis=dict(dtick=2), yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.08)")
)
colB.plotly_chart(fig_idle, use_container_width=True)

# Idle by group
fig_group = go.Figure()
labels = ["fault", "train_join", "senior"]
fig_group.add_trace(go.Bar(x=labels, y=list(idle_group)))
fig_group.update_layout(
    title="Idle % by Group",
    height=260, margin=dict(l=10, r=10, t=45, b=20),
    yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.08)")
)
colC.plotly_chart(fig_group, use_container_width=True)

# Time-in-system distribution (mini)
order = ["fault", "train", "join", "disconnect"]
sys = agg["system_time_all"]
df_sys = pd.DataFrame(
    [(k, v) for k in order for v in sys[k]],
    columns=["type", "minutes"]
)
fig_box = px.box(df_sys, x="type", y="minutes", points=False)
fig_box.update_layout(
    title="Time in System (box, no outliers)",
    height=260, margin=dict(l=10, r=10, t=45, b=20),
    xaxis_title="", yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
)
colD.plotly_chart(fig_box, use_container_width=True)

st.write("")
st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Playback Animation (snapshots)
# ----------------------------
st.subheader("🎞️ Playback — system state over time")

if snap_df.empty:
    st.warning("No snapshots produced. Try a smaller snapshot interval.")
else:
    # Slider chooses snapshot index
    max_i = len(snap_df) - 1
    idx = st.slider("Time index", 0, max_i, 0, 1)
    row = snap_df.iloc[idx]

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("t (min)", f"{row['t']:.0f}")
    top2.metric("Queues (fault / tj / senior)", f"{int(row['Q_fault'])} / {int(row['Q_train_join'])} / {int(row['Q_senior'])}")
    top3.metric("Agents (fault / tj / senior)", f"{int(row['agents_fault'])} / {int(row['agents_tj'])} / {int(row['agents_sen'])}")
    top4.metric("Abandoned (total)", f"{int(row['abandoned_total'])}")

    # Techy "radar-like" gauge for load (queue pressure)
    q_total = int(row["Q_fault"] + row["Q_train_join"] + row["Q_senior"])
    cap = int((row["agents_fault"] + row["agents_tj"] + row["agents_sen"]) * 2)  # rough capacity units
    pressure = 0 if cap == 0 else min(100, 100 * q_total / max(1, cap))

    g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pressure,
        number={'suffix': "%"},
        title={"text": "Queue Pressure (approx)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "steps": [
                {"range": [0, 40]},
                {"range": [40, 70]},
                {"range": [70, 100]},
            ],
        },
    ))
    g.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    st.plotly_chart(g, use_container_width=True)

    # Timeline plot (moving dot)
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=snap_df["t"], y=snap_df["Q_fault"], mode="lines", name="Q fault"))
    fig_tl.add_trace(go.Scatter(x=snap_df["t"], y=snap_df["Q_train_join"], mode="lines", name="Q train_join"))
    fig_tl.add_trace(go.Scatter(x=snap_df["t"], y=snap_df["Q_senior"], mode="lines", name="Q senior"))
    fig_tl.add_trace(go.Scatter(
        x=[row["t"]], y=[row["Q_fault"]], mode="markers", name="NOW",
        marker=dict(size=12, symbol="circle-open")
    ))
    fig_tl.update_layout(
        title="Queue lengths over time (playback cursor)",
        height=340,
        margin=dict(l=10, r=10, t=60, b=30),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    with st.expander("🔍 Snapshot table (debug / appendix)"):
        st.dataframe(snap_df, use_container_width=True)

st.write("")
st.caption("Ayelet+Adi+Odelia")



