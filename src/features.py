import numpy as np
import pandas as pd

def fourier_series(t, period, K=3):
    """
    Create Fourier terms up to order K for a given period (in hours).
    t: np.array of indices [0..T-1]
    """
    X = {}
    for k in range(1, K+1):
        X[f'sin_{period}_{k}'] = np.sin(2*np.pi*k*t/period)
        X[f'cos_{period}_{k}'] = np.cos(2*np.pi*k*t/period)
    return pd.DataFrame(X)

def build_feature_frame(df: pd.DataFrame, lags=48, rolling=[24, 168]):
    """
    df: DataFrame with columns ['timestamp','temp_c','demand_mw'] indexed by time ascending
    Returns a new DataFrame with feature columns and target column 'y' (current demand).
    """
    df = df.copy().reset_index(drop=True)
    df = df.sort_values('timestamp')
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dow']  = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month']= pd.to_datetime(df['timestamp']).dt.month

    # Lag features for demand and temperature
    for L in range(1, lags+1):
        df[f"lag_{L}"] = df['demand_mw'].shift(L)
        df[f"temp_lag_{L}"] = df['temp_c'].shift(L)

    # Rolling means
    for r in rolling:
        df[f"roll_mean_{r}"] = df['demand_mw'].rolling(r).mean()

    # Fourier seasonality
    t = np.arange(len(df))
    F_daily = fourier_series(t, period=24, K=3)
    F_week  = fourier_series(t, period=24*7, K=2)
    F_annual= fourier_series(t, period=int(24*365.25), K=2)
    F = pd.concat([F_daily, F_week, F_annual], axis=1)
    df = pd.concat([df, F], axis=1)

    # Target
    df['y'] = df['demand_mw']

    # Drop rows with NaN due to lags/rolling
    df = df.dropna().reset_index(drop=True)
    return df

def make_multi_horizon_sets(df_feat: pd.DataFrame, horizon=48):
    """
    Prepare training matrices for direct multi-horizon forecasting:
    X at time t mapped to y_{t+h} for h in 1..H.
    Returns: X, Y_dict where Y_dict[h] is a Series for horizon h.
    """
    feature_cols = [c for c in df_feat.columns if c not in ['timestamp','demand_mw','y']]
    X = df_feat[feature_cols].copy()
    Y_dict = {}
    for h in range(1, horizon+1):
        Y_dict[h] = df_feat['y'].shift(-h)  # y at t+h
    # Align by dropping last H rows
    X = X.iloc[:-horizon].reset_index(drop=True)
    for h in list(Y_dict.keys()):
        Y_dict[h] = Y_dict[h].iloc[:-horizon].reset_index(drop=True)
    return X, Y_dict, feature_cols
