import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="EV Forecast", layout="wide")

# Load model
model = joblib.load('forecasting_ev_model.pkl')

# Custom styling
st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
        }
        h1, h2, h3, h4, h5, h6, .stTextInput label, .stSelectbox label, .stMultiSelect label, .stButton>button {
            color: #FFD700;
        }
        .stMarkdown {
            color: #FFD700;
        }
        .stSuccess {
            background-color: #333333 !important;
            color: #FFD700 !important;
            border-left: 5px solid #FFD700;
        }
        .css-1kyxreq, .css-ffhzg2 {
            color: #FFD700 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFD700; margin-top: 20px;'>
        🔮 EV Adoption Forecaster
    </div>
""", unsafe_allow_html=True)

# Subtitle
st.markdown("""
    <div style='text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 20px; color: #AAAAAA;'>
        Forecasting Electric Vehicle (EV) Trends in Washington Counties
    </div>
""", unsafe_allow_html=True)

# Image
st.image("ev-car-factory.jpg", use_container_width=True)

# Instruction
st.markdown("""
    <div style='font-size: 18px; color: #FFD700; padding-top: 10px;'>
        Select a county to view the EV forecast for the next 3 years.
    </div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Dropdown
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# Forecast logic
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()
forecast_horizon = 36
future_rows = []

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0]

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# Combine historical and forecasted
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# Plot
st.subheader(f"📊 Cumulative EV Forecast: {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_facecolor("#000000")
fig.patch.set_facecolor("#000000")
ax.set_title(f"EV Trend (Historical + Forecast)", color="#FFD700")
ax.set_xlabel("Date", color="#FFD700")
ax.set_ylabel("Cumulative EVs", color="#FFD700")
ax.tick_params(colors='#FFD700')
ax.legend()
st.pyplot(fig)

# Forecast interpretation
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    pct_change = ((forecasted_total - historical_total) / historical_total) * 100
    icon = "📈" if pct_change > 0 else "📉"
    st.success(f"EVs in **{county}** are expected to show a **{icon} {pct_change:.2f}% change** in the next 3 years.")
else:
    st.warning("Cannot compute forecast growth due to zero historical data.")

# Prepared for
st.markdown("<br><hr><div style='text-align:center; color: #FFD700;'>Prepared for the <b>AICTE INTERNSHIP</b></div>", unsafe_allow_html=True)
