
from arch import arch_model
import pandas as pd

def forecast_garch_var(df, horizon=30, confidence=0.95):
    returns = df["Returns"].dropna() * 100
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=horizon)
    variance_forecast = forecast.variance.values[-1, :]
    volatility = (variance_forecast ** 0.5) / 100
    var_1d = -res.conditional_volatility.iloc[-1] * 1.65
    return pd.Series(volatility), var_1d
