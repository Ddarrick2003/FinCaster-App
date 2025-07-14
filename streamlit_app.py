
import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go

st.set_page_config(page_title="FinCaster", layout="wide")
st.title("🌞💵 FinCaster: Financial Forecasting App")

uploaded_file = st.file_uploader("📤 Upload your OHLCV CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Preview:", df.head())
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()

    st.success(f"✅ Data Loaded: {df.shape} rows")

    if df.shape[0] < 60:
        st.warning("⚠️ Limited data rows. Some models might not perform optimally.")

    tab1, tab2 = st.tabs(["📈 LSTM Forecast", "📉 GARCH Risk Forecast"])

    with tab1:
        st.subheader("LSTM Forecasting")
        try:
            features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
            X, y = create_sequences(df[features], target_col='Close')
            if len(X) == 0:
                st.warning("⚠️ Not enough data for sequence modeling.")
            else:
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                model.fit(X_train, y_train, epochs=10, batch_size=16,
                          validation_data=(X_test, y_test),
                          callbacks=[EarlyStopping(patience=3)], verbose=0)
                preds = model.predict(X_test).flatten()
                st.line_chart({"Actual": y_test[:100], "Predicted": preds[:100]})
        except Exception as e:
            st.error(f"LSTM failed: {e}")

    with tab2:
        st.subheader("GARCH Risk Forecast")
        try:
            vol_forecast, var_1d = forecast_garch_var(df)
            st.metric(label="1-Day VaR (95%)", value=f"{var_1d:.2f}%")
            st.line_chart(vol_forecast.values)
            st.markdown("### 📘 GARCH Explanation")
            st.info(f'''
            - 📈 **Volatility Forecast** shows future uncertainty in returns.
            - 📉 **VaR** estimates max loss with 95% confidence over 1 day.
            ''')
        except Exception as e:
            st.error(f"GARCH failed: {e}")

else:
    st.info("📥 Please upload a CSV file to begin.")
