import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="FinCaster", layout="wide")
st.title("🌞💵 FinCaster: Financial Forecasting App")

uploaded_file = st.file_uploader("📤 Upload your OHLCV CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Missing required columns. Found: {df.columns}")
            st.stop()

        df = preprocess_data(df)
        st.write("📄 Preview (last 5 rows):", df.tail())

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()

    st.success(f"✅ Data Loaded: {df.shape[0]} rows")

    if df.shape[0] < 60:
        st.warning("⚠️ Data too short. Some models may not perform well.")

    tab1, tab2 = st.tabs(["📈 LSTM Forecast", "📉 GARCH Risk Forecast"])

    with tab1:
        st.subheader("📈 LSTM Forecasting")
        try:
            features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
            X, y = create_sequences(df[features], target_col='Close')

            if len(X) == 0:
                st.warning("⚠️ Not enough sequence data for training.")
            else:
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                model.fit(X_train, y_train, epochs=10, batch_size=16,
                          validation_data=(X_test, y_test),
                          callbacks=[EarlyStopping(patience=3)], verbose=0)

                preds = model.predict(X_test).flatten()

                chart_df = pd.DataFrame({
                    'Actual': y_test[:100],
                    'Predicted': preds[:100]
                })

                st.line_chart(chart_df)
        except Exception as e:
            st.error(f"LSTM forecast error: {e}")

    with tab2:
        st.subheader("📉 GARCH Risk Forecast")
        try:
            vol_forecast, var_1d = forecast_garch_var(df)
            st.metric("1-Day VaR (95%)", f"{var_1d:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=vol_forecast.values, mode="lines", name="Volatility Forecast"))
            fig.update_layout(title="Forecasted Volatility via GARCH")
            st.plotly_chart(fig)

            st.markdown("### 📘 Risk Interpretation")
            st.info(f"""
            - **Volatility Chart**: Predicts near-term return variability.
            - **1-Day Value at Risk**: Loss not expected to exceed **{abs(var_1d):.2f}%** with 95% confidence.
            """)
        except Exception as e:
            st.error(f"GARCH forecast error: {e}")

else:
    st.info("📥 Upload a valid CSV with OHLCV to get started.")
