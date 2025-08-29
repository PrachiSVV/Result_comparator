# app.py
import hashlib
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# ==============================
# ---------- CONFIG ------------
# ==============================
st.set_page_config(
    page_title="Results vs Expectations",
    page_icon="📊",
    layout="wide",
)

REQUIRED_COLS = [
    "co_code","nsesymbol","broker_name","sales","pat","ebitda","picked_type",
    "expected_sales","expected_ebitda","expected_pat",
    "ebitda_margin_percent","pat_margin_percent",
    "sales_beat","pat_beat","ebitda_beat",
    "sales_flag","pat_flag","ebitda_flag","overall_flag",
]

NUMERIC_COLS = [
    "sales","pat","ebitda",
    "expected_sales","expected_ebitda","expected_pat",
    "ebitda_margin_percent","pat_margin_percent",
    "sales_beat","pat_beat","ebitda_beat",
]

STATUS_ORDER = ["Beat", "Inline", "Miss"]

# ==============================
# --------- AUTH ---------------
# ==============================
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def check_login(username: str, password: str) -> bool:
    users = st.secrets.get("auth", {}).get("users", {})
    if not username or username not in users:
        return False
    return _sha256(password) == users[username]

def login_gate():
    if "auth_user" in st.session_state:
        return True

    st.markdown(
        """
        <div style="padding:24px;border-radius:16px;background:linear-gradient(135deg,#0ea5e9, #3b82f6);color:white;">
          <h2 style="margin:0;">🔐 Login to Results Dashboard</h2>
          <p style="margin:6px 0 0 0;opacity:.9;">Secure access required</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login"):
        c1, c2, c3 = st.columns([3,3,1.2])
        username = c1.text_input("Username")
        password = c2.text_input("Password", type="password")
        submit   = c3.form_submit_button("Log in")
    if submit:
        if check_login(username, password):
            st.session_state.auth_user = username
            st.success("Logged in successfully.")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")
            time.sleep(0.6)
    st.stop()

# Gate the app
login_gate()

# Optional logout in sidebar
if st.sidebar.button("🚪 Log out"):
    st.session_state.pop("auth_user", None)
    st.rerun()

# ==============================
# ---------- DATA --------------
# ==============================
@st.cache_data(show_spinner=False)
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def validate_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

def first_non_null(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else None

def build_flag_percent_df(cmp: pd.DataFrame) -> pd.DataFrame:
    """
    Returns long DF with columns: metric (sales/ebitda/pat), status (Beat/Inline/Miss), percent, count
    across all broker rows for the selected company.
    """
    rows = []
    for flag_col, label in [("sales_flag","sales"), ("ebitda_flag","ebitda"), ("pat_flag","pat")]:
        col = cmp[flag_col].dropna()
        total = len(col)
        for status in STATUS_ORDER:
            count = int((col == status).sum()) if total > 0 else 0
            pct = (count / total * 100.0) if total > 0 else 0.0
            rows.append({"metric": label, "status": status, "percent": pct, "count": count, "total": total})
    out = pd.DataFrame(rows)
    out["metric"] = pd.Categorical(out["metric"], categories=["sales","ebitda","pat"], ordered=True)
    out["status"] = pd.Categorical(out["status"], categories=STATUS_ORDER, ordered=True)
    return out.sort_values(["metric","status"])

# ==============================
# ---------- HERO --------------
# ==============================
st.markdown(
    f"""
    <div style="padding:22px 22px 14px;border-radius:16px;
                background:linear-gradient(135deg,#0ea5e9, #3b82f6);color:#fff;margin-bottom:8px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <h1 style="margin:0;padding:0;">📊 Results vs Broker Expectations</h1>
          <p style="margin:6px 0 0 0;opacity:.95;">
            One view to compare <b>Actual</b> vs <b>Expected</b> by metric, plus Beat/Inline/Miss distribution across brokers.
          </p>
        </div>
        <div style="text-align:right;opacity:.9;">
          <div>Signed in as <b>{st.session_state['auth_user']}</b></div>
          <div style="font-size:12px;">{datetime.now().strftime('%d %b %Y, %I:%M %p')}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# ---------- INPUT -------------
# ==============================
st.caption("Upload your CSV (schema validated).")
file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
if file is None:
    with st.expander("ℹ️ CSV schema & tips", expanded=False):
        st.code(", ".join(REQUIRED_COLS), language="text")
        st.info("Numeric columns will be auto-coerced. Invalid values become NaN.")
    st.stop()

df = read_csv(file)
validate_schema(df)
df = coerce_numeric(df)

# ==============================
# --------- FILTERS ------------
# ==============================
with st.sidebar:
    st.subheader("Filters")
    symbols = sorted(df["nsesymbol"].dropna().unique())
    sel_symbol = st.selectbox("Company (nsesymbol)", options=symbols, index=0, help="Pick a company to visualize")

    all_brokers = sorted(df["broker_name"].dropna().unique())
    sel_brokers = st.multiselect("Brokers", options=all_brokers, default=all_brokers)

    all_picked = sorted(df["picked_type"].dropna().unique())
    sel_picked = st.multiselect("picked_type", options=all_picked, default=all_picked)

    flag_vals = ["Beat","Miss","Inline"]
    sel_sales_flag   = st.multiselect("sales_flag",  options=flag_vals, default=flag_vals)
    sel_pat_flag     = st.multiselect("pat_flag",    options=flag_vals, default=flag_vals)
    sel_ebitda_flag  = st.multiselect("ebitda_flag", options=flag_vals, default=flag_vals)
    sel_overall_flag = st.multiselect("overall_flag",options=flag_vals, default=flag_vals)

# Apply filters
f = df[
    (df["broker_name"].isin(sel_brokers)) &
    (df["picked_type"].isin(sel_picked)) &
    (df["sales_flag"].isin(sel_sales_flag)) &
    (df["pat_flag"].isin(sel_pat_flag)) &
    (df["ebitda_flag"].isin(sel_ebitda_flag)) &
    (df["overall_flag"].isin(sel_overall_flag))
].copy()

cmp = f[f["nsesymbol"] == sel_symbol].copy()
if cmp.empty:
    st.warning("No rows for selected company with current filters.")
    st.stop()

# ==============================
# --------- METRICS ------------
# ==============================
actual_sales  = first_non_null(cmp["sales"])
actual_ebitda = first_non_null(cmp["ebitda"])
actual_pat    = first_non_null(cmp["pat"])

avg_exp_sales  = cmp["expected_sales"].mean()
avg_exp_ebitda = cmp["expected_ebitda"].mean()
avg_exp_pat    = cmp["expected_pat"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Actual Sales", f"{actual_sales:,.2f}" if pd.notna(actual_sales) else "—")
m2.metric("Avg Expected Sales", f"{avg_exp_sales:,.2f}" if pd.notna(avg_exp_sales) else "—")
m3.metric("Avg Expected EBITDA", f"{avg_exp_ebitda:,.2f}" if pd.notna(avg_exp_ebitda) else "—")
m4.metric("Avg Expected PAT", f"{avg_exp_pat:,.2f}" if pd.notna(avg_exp_pat) else "—")

# Warn if actuals differ across broker rows (shouldn't)
if cmp["sales"].nunique(dropna=True) > 1 or cmp["pat"].nunique(dropna=True) > 1 or cmp["ebitda"].nunique(dropna=True) > 1:
    st.info("ℹ️ Multiple differing actuals detected across rows; using the first non-null per metric for charting.")

# ==============================
# --------- CHARTS -------------
# ==============================
st.markdown("### 📈 Actual vs Brokers’ Expected — grouped by **metric**")

# Expected per broker (mean if duplicates)
exp_by_broker = (
    cmp.groupby("broker_name", as_index=False)[
        ["expected_sales","expected_ebitda","expected_pat"]
    ].mean()
    .sort_values("broker_name")
)

# Build long DF for a single grouped-by-metric chart: Actual + Expected per broker
rows = [
    {"metric": "sales",  "series": "Actual", "value": actual_sales},
    {"metric": "ebitda", "series": "Actual", "value": actual_ebitda},
    {"metric": "pat",    "series": "Actual", "value": actual_pat},
]
for _, r in exp_by_broker.iterrows():
    rows.append({"metric": "sales",  "series": f"Expected · {r['broker_name']}", "value": r["expected_sales"]})
    rows.append({"metric": "ebitda", "series": f"Expected · {r['broker_name']}", "value": r["expected_ebitda"]})
    rows.append({"metric": "pat",    "series": f"Expected · {r['broker_name']}", "value": r["expected_pat"]})

plot_df = pd.DataFrame(rows)
fig_actual_vs_expected = px.bar(
    plot_df,
    x="metric",
    y="value",
    color="series",
    barmode="group",
    title=f"{sel_symbol}: Actual vs Brokers' Expected (Sales / EBITDA / PAT)",
    category_orders={"metric": ["sales","ebitda","pat"]},
    hover_data={"metric": True, "value": ":.2f", "series": True},
)
fig_actual_vs_expected.update_layout(
    xaxis_title="Metric",
    yaxis_title="Value",
    legend_title="",
    margin=dict(t=60, r=10, l=10, b=10),
)
st.plotly_chart(fig_actual_vs_expected, use_container_width=True)

# --- Beat values by metric (as given) ---
# ==============================
st.markdown("### 📊 Beat values (as % vs expectation) — grouped by metric")

beat_long = cmp.melt(
    id_vars=["broker_name"],
    value_vars=["sales_beat", "ebitda_beat", "pat_beat"],
    var_name="metric",
    value_name="percent_value"
)

fig_beats = px.bar(
    beat_long,
    x="metric",
    y="percent_value",
    color="broker_name",
    barmode="group",
    title=f"{sel_symbol}: Beat values (already % in data)",
    category_orders={"metric": ["sales_beat","ebitda_beat","pat_beat"]},
    hover_data={"broker_name":True,"percent_value":":.2f"}
)
fig_beats.update_layout(
    xaxis_title="Metric",
    yaxis_title="Beat value (%)",
    legend_title="Broker"
)
st.plotly_chart(fig_beats, use_container_width=True)


# ==============================
# --------- TABS ---------------
# ==============================
tab1, tab2 = st.tabs(["📋 Selected Company Table", "📄 Full Filtered Data"])

with tab1:
    st.dataframe(
        cmp.sort_values("broker_name"),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    def highlight_beats(row: pd.Series):
        any_beat = (
            row.get("sales_flag") == "Beat"
            or row.get("pat_flag") == "Beat"
            or row.get("ebitda_flag") == "Beat"
            or row.get("overall_flag") == "Beat"
        )
        return ['background-color: #e6ffe6'] * len(row) if any_beat else [''] * len(row)

    st.dataframe(
        f.style.apply(highlight_beats, axis=1),
        use_container_width=True,
    )

    csv = f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv,
        file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# Footer
st.caption("Built for clarity: Actual vs expected, plus Beat/Inline/Miss distribution, with login & schema validation.")

