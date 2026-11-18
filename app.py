
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title = "ATM Withdrawal Forecast",
    page_icon  = "🏧",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

ATM_COL  = "ATM_ID"
DATE_COL = "TRAN_DATE"
W_XGB    = 0.509
W_LGB    = 0.491

EGYPT_HOLIDAYS = {
    pd.Timestamp("2025-01-07"), pd.Timestamp("2025-04-25"),
    pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-30"),
    pd.Timestamp("2025-07-23"), pd.Timestamp("2025-10-06"),
    pd.Timestamp("2026-01-07"), pd.Timestamp("2026-04-25"),
    pd.Timestamp("2026-05-01"),
}

@st.cache_resource
def load_models():
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model("model_xgboost_v2.json")
    lgb_m = lgb.Booster(model_file="model_lightgbm_v2.txt")
    return xgb_m, lgb_m

@st.cache_data
def load_data():
    train = pd.read_parquet("train_small.parquet")
    test  = pd.read_parquet("test_small.parquet")

    train[DATE_COL] = pd.to_datetime(train[DATE_COL])
    test[DATE_COL]  = pd.to_datetime(test[DATE_COL])

    full = pd.concat([train, test]).sort_values(
        [ATM_COL, DATE_COL]
    ).reset_index(drop=True)

    with open("feature_cols_v2.pkl", "rb") as f:
        features = pickle.load(f)

    with open("global_mean.pkl", "rb") as f:
        g_mean = pickle.load(f)

    atm_stats  = pd.read_parquet("atm_stats.parquet")
    bank_stats = pd.read_parquet("bank_stats.parquet")
    atm_enc    = pd.read_parquet("atm_encoding.parquet")["ENC"]

    return train, test, full, features, g_mean, atm_stats, bank_stats, atm_enc

try:
    xgb_model, lgb_model = load_models()
    (train_df, test_df, full_df,
     FEATURE_COLS, global_mean,
     atm_stats, bank_stats, atm_enc) = load_data()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

def get_lag_value(history_df, target_date, lag_days):
    lookup = target_date - pd.Timedelta(days=lag_days)
    row    = history_df[history_df[DATE_COL] == lookup]
    if len(row) > 0:
        return row["DAILY_WITHDRAWAL"].values[-1]
    return history_df["DAILY_WITHDRAWAL"].mean()

def predict_weekly_st(atm_id, start_date):
    start_date = pd.Timestamp(start_date)

    atm_hist = (
        full_df[full_df[ATM_COL] == atm_id]
        [[ATM_COL, DATE_COL, "DAILY_WITHDRAWAL"]
         + [c for c in FEATURE_COLS if c in full_df.columns]]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    if len(atm_hist) == 0:
        return None

    static_row   = atm_hist.iloc[-1]
    history_copy = atm_hist.copy()
    future_dates = pd.date_range(start=start_date, periods=7, freq="D")
    predictions  = []

    atm_hist_mean   = atm_hist["DAILY_WITHDRAWAL"].mean()
    atm_hist_std    = atm_hist["DAILY_WITHDRAWAL"].std()
    atm_hist_median = atm_hist["DAILY_WITHDRAWAL"].median()
    atm_hist_max    = atm_hist["DAILY_WITHDRAWAL"].max()

    atm_id_enc = atm_enc.get(atm_id, global_mean)

    bank_id   = static_row["BANK_ID"] if "BANK_ID" in static_row.index else None
    bank_mean = bank_stats.loc[bank_id, "BANK_MEAN"] if bank_id in bank_stats.index else global_mean
    bank_std  = bank_stats.loc[bank_id, "BANK_STD"]  if bank_id in bank_stats.index else 0

    for pred_date in future_dates:
        dom     = pred_date.day
        dow     = pred_date.dayofweek
        is_wknd = dow in [4, 5]

        if is_wknd:
            pred_val = 0
        else:
            lags = {}
            for lag in [7, 14, 21, 28]:
                lags[f"LAG_{lag}"] = get_lag_value(history_copy, pred_date, lag)

            last_7  = history_copy.tail(7)["DAILY_WITHDRAWAL"]
            last_14 = history_copy.tail(14)["DAILY_WITHDRAWAL"]
            last_28 = history_copy.tail(28)["DAILY_WITHDRAWAL"]

            rolling_7  = last_7.mean()
            rolling_14 = last_14.mean()
            rolling_28 = last_28.mean()
            rolling_std = last_7.std()
            rolling_max = last_7.max()

            features = {
                "IS_PRIVATE_SALARY"  : int(1 <= dom <= 3),
                "IS_GOV_SALARY"      : int(23 <= dom <= 25),
                "IS_PENSION"         : int(29 <= dom <= 31),
                "IS_PRE_WEEKEND"     : int(dow == 3),
                "IS_MONDAY"          : int(dow == 0),
                "IS_MID_MONTH"       : int(15 <= dom <= 21),
                "IS_HOLIDAY"         : int(pred_date in EGYPT_HOLIDAYS),
                "DAY_OF_WEEK"        : dow,
                "DAY_OF_MONTH"       : dom,
                "MONTH"              : pred_date.month,
                "QUARTER"            : (pred_date.month - 1) // 3 + 1,
                "IS_END_OF_MONTH"    : int(dom >= 26),
                "IS_FIRST_WEEK"      : int(dom <= 7),
                "IS_SECOND_WEEK"     : int(8 <= dom <= 14),
                "IS_THIRD_WEEK"      : int(15 <= dom <= 21),
                "IS_FOURTH_WEEK"     : int(dom >= 22),
                "WEEK_OF_MONTH"      : (dom - 1) // 7 + 1,
                "LAG_7"              : lags["LAG_7"],
                "LAG_14"             : lags["LAG_14"],
                "LAG_21"             : lags["LAG_21"],
                "LAG_28"             : lags["LAG_28"],
                "ROLLING_MEAN_7"     : rolling_7,
                "ROLLING_MEAN_14"    : rolling_14,
                "ROLLING_MEAN_28"    : rolling_28,
                "ROLLING_STD_7"      : rolling_std,
                "ROLLING_MAX_7"      : rolling_max,
                "LOG_LAG_7"          : np.log1p(lags["LAG_7"]),
                "LOG_LAG_14"         : np.log1p(lags["LAG_14"]),
                "LOG_LAG_21"         : np.log1p(lags["LAG_21"]),
                "LOG_LAG_28"         : np.log1p(lags["LAG_28"]),
                "LOG_ROLLING_MEAN_7" : np.log1p(rolling_7),
                "LOG_ROLLING_MEAN_14": np.log1p(rolling_14),
                "LOG_ROLLING_MEAN_28": np.log1p(rolling_28),
                "ATM_ID_ENC"         : atm_id_enc,
                "ATM_HIST_MEAN"      : atm_hist_mean,
                "ATM_HIST_STD"       : atm_hist_std,
                "ATM_HIST_MEDIAN"    : atm_hist_median,
                "ATM_HIST_MAX"       : atm_hist_max,
                "RATIO_TO_ATM_MEAN"  : rolling_7 / (atm_hist_mean + 1),
                "SALARY_X_PREWKND"   : int(1 <= dom <= 3) * int(dow == 3),
                "SALARY_X_MONDAY"    : int(1 <= dom <= 3) * int(dow == 0),
                "GOV_X_PREWKND"      : int(23 <= dom <= 25) * int(dow == 3),
                "LAG7_X_PREWKND"     : lags["LAG_7"] * int(dow == 3),
                "ROLLING7_X_SALARY"  : rolling_7 * int(1 <= dom <= 3),
                "BANK_MEAN"          : bank_mean,
                "BANK_STD"           : bank_std,
            }

            for col in FEATURE_COLS:
                if col not in features:
                    features[col] = static_row[col] if col in static_row.index else 0

            X_pred   = pd.DataFrame([features])[FEATURE_COLS]
            X_pred   = X_pred.apply(pd.to_numeric, errors="coerce").fillna(0)
            p_xgb    = float(xgb_model.predict(X_pred)[0])
            p_lgb    = float(lgb_model.predict(X_pred.values)[0])
            pred_val = max(0, W_XGB * p_xgb + W_LGB * p_lgb)

        new_row = pd.DataFrame([{
            ATM_COL           : atm_id,
            DATE_COL          : pred_date,
            "DAILY_WITHDRAWAL": pred_val
        }])
        history_copy = pd.concat([history_copy, new_row], ignore_index=True)

        sal = (
            "💰 Private Salary" if 1 <= dom <= 3 else
            "🏛️ Gov Salary"     if 23 <= dom <= 25 else
            "👴 Pension"         if 29 <= dom <= 31 else
            "📅 Mid-Month"       if 15 <= dom <= 21 else
            ""
        )
        predictions.append({
            "Date": pred_date, "Day": pred_date.strftime("%A"),
            "DOM": dom, "Predicted": pred_val,
            "Is_Weekend": is_wknd, "Salary_Flag": sal,
        })

    pred_df = pd.DataFrame(predictions)
    working = pred_df[~pred_df["Is_Weekend"]]

    return {
        "atm_id"    : atm_id,
        "start_date": start_date,
        "predictions": pred_df,
        "total_week": pred_df["Predicted"].sum(),
        "total_work": working["Predicted"].sum(),
        "avg_work"  : working["Predicted"].mean() if len(working) > 0 else 0,
        "atm_mean"  : atm_hist_mean,
        "history"   : atm_hist,
    }

# ════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════
st.markdown(
    "<h1 class='main-header'>🏧 ATM Withdrawal Forecast</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

if not models_loaded:
    st.error("❌ Models not loaded! Check files.")
    st.stop()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    all_atms = sorted(full_df[ATM_COL].unique().tolist())
    selected_atm = st.selectbox(
        "🏧 Select ATM ID",
        options = all_atms,
        help    = f"{len(all_atms):,} ATMs available"
    )

    st.markdown("---")
    st.markdown("### 📅 Forecast Date")

    data_end  = full_df[DATE_COL].max()
    pred_date = st.date_input(
        "Forecast Start Date",
        value = (data_end + pd.Timedelta(days=1)).date(),
        help  = "Choose any date"
    )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("Today", use_container_width=True):
            pred_date = pd.Timestamp.today().date()
    with col_q2:
        if st.button("+7 Days", use_container_width=True):
            pred_date = (pd.Timestamp.today() + pd.Timedelta(days=7)).date()

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.success("🏆 XGBoost v2 + LightGBM v2")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("WMAPE", "36.9%")
        st.metric("R²",    "0.641")
    with col_m2:
        st.metric("MAE", "72K EGP")

    st.markdown("---")
    atm_data = full_df[full_df[ATM_COL] == selected_atm]
    if len(atm_data) > 0:
        st.markdown("### ATM Stats")
        st.metric("Avg Daily", f"{atm_data['DAILY_WITHDRAWAL'].mean():,.0f} EGP")
        st.metric("Max Daily", f"{atm_data['DAILY_WITHDRAWAL'].max():,.0f} EGP")
        st.metric("Days",      f"{len(atm_data):,}")

    st.markdown("---")
    predict_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_key" not in st.session_state:
    st.session_state.last_key = None

current_key = f"{selected_atm}_{pred_date}"

if predict_btn or st.session_state.last_key != current_key:
    with st.spinner("🔄 Generating forecast..."):
        result = predict_weekly_st(selected_atm, pred_date)
    st.session_state.last_result = result
    st.session_state.last_key    = current_key
else:
    result = st.session_state.last_result

if result is None:
    st.info("👈 اختار ATM واضغط Generate Forecast")
    st.stop()

pred_df = result["predictions"]
working = pred_df[~pred_df["Is_Weekend"]]
change  = result["avg_work"] - result["atm_mean"]
change_pct = change / result["atm_mean"] * 100 if result["atm_mean"] > 0 else 0

days_ahead = (pd.Timestamp(pred_date) - data_end).days
if days_ahead > 30:
    st.warning(f"⚠️ Forecasting {days_ahead} days after last data ({data_end.strftime('%d %b %Y')})")
elif days_ahead > 0:
    st.info(f"ℹ️ Forecasting {days_ahead} days ahead")
else:
    st.info("📊 Backtesting Mode")

st.markdown("### 📊 Weekly Summary")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total (Full Week)",   f"{result['total_week']:,.0f} EGP")
with col2: st.metric("Total (Working)",      f"{result['total_work']:,.0f} EGP")
with col3: st.metric("Avg per Day",          f"{result['avg_work']:,.0f} EGP", delta=f"{change:+,.0f}")
with col4: st.metric("vs Historical",        f"{change_pct:+.1f}%")

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Historical + Forecast")
    fig, ax = plt.subplots(figsize=(12, 5))
    hist = result["history"].tail(60)
    ax.plot(hist[DATE_COL], hist["DAILY_WITHDRAWAL"],
            color="steelblue", lw=1.5, alpha=0.8, label="Historical")
    ax.plot(pred_df["Date"], pred_df["Predicted"],
            color="red", lw=2.5, marker="o", markersize=8,
            ls="--", label="Forecast", zorder=5)
    ax.axhline(result["atm_mean"], color="gray", ls=":", lw=1.5,
               alpha=0.7, label=f"Avg: {result['atm_mean']:,.0f}")
    ax.axvline(pd.Timestamp(pred_date), color="green", ls="--", lw=2, alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Withdrawal (EGP)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("📊 Daily Breakdown")
    colors_bar = [
        "#BDC3C7" if r["Is_Weekend"] else
        "#F39C12" if r["Salary_Flag"] != "" else
        "#85C1E9" if 15 <= r["DOM"] <= 21 else
        "#2ECC71"
        for _, r in pred_df.iterrows()
    ]
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bars = ax2.bar(pred_df["Day"], pred_df["Predicted"],
                   color=colors_bar, edgecolor="white")
    for bar, val in zip(bars, pred_df["Predicted"]):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 2000,
                     f"{val/1e3:.0f}K",
                     ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    ax2.axhline(result["atm_mean"], color="red", ls="--", lw=2)
    ax2.set_ylabel("EGP")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")
st.subheader("📋 Forecast Details")

display_df = pred_df.copy()
display_df["Date"]    = display_df["Date"].dt.strftime("%Y-%m-%d")
display_df["Amount"]  = display_df["Predicted"].apply(lambda x: f"{x:,.0f} EGP")
display_df["Weekend"] = display_df["Is_Weekend"].apply(lambda x: "🏖️ Yes" if x else "✅ No")

st.dataframe(
    display_df[["Date", "Day", "DOM", "Amount", "Weekend", "Salary_Flag"]],
    use_container_width=True, hide_index=True
)

sal_days = pred_df[(pred_df["Salary_Flag"] != "") & (~pred_df["Is_Weekend"])]
if len(sal_days) > 0:
    st.markdown("---")
    st.subheader("⚡ Special Days")
    cols_sal = st.columns(len(sal_days))
    for i, (_, row) in enumerate(sal_days.iterrows()):
        with cols_sal[i]:
            st.info(f"{row['Day']}\n{row['Date'].strftime('%d %b')}\n{row['Salary_Flag']}\n{row['Predicted']:,.0f} EGP")

st.markdown("---")
st.subheader("📥 Export")

col_e1, col_e2 = st.columns(2)
with col_e1:
    csv = pred_df.to_csv(index=False)
    st.download_button(
        label="Download CSV", data=csv,
        file_name=f"forecast_{selected_atm}_{pred_date}.csv",
        mime="text/csv", use_container_width=True
    )
with col_e2:
    lines = [
        "ATM Withdrawal Forecast",
        "="*40,
        f"ATM ID   : {selected_atm}",
        f"Week     : {pred_date}",
        f"Total    : {result['total_work']:,.0f} EGP",
        f"Avg/Day  : {result['avg_work']:,.0f} EGP",
        f"vs Hist  : {change:+,.0f} EGP ({change_pct:+.1f}%)",
        "", "Daily:", "-"*40,
    ]
    for _, row in pred_df.iterrows():
        wknd = "[WKD]" if row["Is_Weekend"] else "     "
        lines.append(
            f"{row['Date'].strftime('%Y-%m-%d')} {row['Day']:<12} "
            f"{wknd} {row['Predicted']:>10,.0f} EGP {row['Salary_Flag']}"
        )
    st.download_button(
        label="Download Report", data="\n".join(lines),
        file_name=f"report_{selected_atm}_{pred_date}.txt",
        mime="text/plain", use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.8rem;'>"
    "🏧 ATM Forecast | XGBoost v2 + LightGBM v2 | WMAPE: 36.90% | Egypt ATM Network"
    "</div>",
    unsafe_allow_html=True
)
