import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

from .data_simulation import simulate_demand_weather
from .forecaster_lstm import DemandForecasterLSTM, LSTMConfig
from .optimizer import economic_dispatch_lp
from .evaluation import regression_metrics, plot_forecast, plot_dispatch

import argparse

def cli():
    p = argparse.ArgumentParser(description="Forecast demand and optimize dispatch.")
    p.add_argument('--horizon', type=int, default=48, help='Forecast horizon in hours (default: 48)')
    p.add_argument('--fast', action='store_true', help='Use a smaller training window and 24h horizon for a quick run')
    return p.parse_args()

ROOT = Path(__file__).resolve().parents[1]

def ensure_data():
    data_path = ROOT / "data" / "demand_weather.csv"
    if not data_path.exists():
        df = simulate_demand_weather()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
    return df


def train_and_forecast(df, horizon=48, fast=False):
    # LSTM configuration
    seq_len = 72 if fast else 168
    epochs  = 8  if fast else 30
    cfg = LSTMConfig(seq_len=seq_len, horizon=horizon, epochs=epochs, batch_size=128, lr=1e-3,
                     hidden_size=128, num_layers=2, dropout=0.2, device="cpu")

    forecaster = DemandForecasterLSTM(cfg)
    metrics = forecaster.fit(df)

    # Build 48h (or set horizon) forecast from the latest window
    y_hat = forecaster.predict(df)

    # Forecast timestamps
    last_ts = pd.to_datetime(df['timestamp']).max()
    ts_fore = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq='H')
    df_fore = pd.DataFrame({'timestamp': ts_fore, 'y_hat': y_hat})

    # History slice for plotting
    hist_slice_start = last_ts - pd.Timedelta(days=10)
    df_hist = df[pd.to_datetime(df['timestamp']) >= hist_slice_start].copy()

    # Save model
    forecaster.save(str(ROOT / "models"))

    return forecaster, metrics, df_hist, df_fore

def run_dispatch(df_fore):
    demand_forecast = df_fore['y_hat'].values
    res = economic_dispatch_lp(demand_forecast, T=len(demand_forecast))

    df_opt = df_fore.copy()
    df_opt['g_base'] = res['g_base']
    df_opt['g_peak'] = res['g_peak']
    df_opt['charge'] = res['charge']
    df_opt['discharge'] = res['discharge']
    df_opt['soc'] = res['soc']
    df_opt['demand_hat'] = demand_forecast
    return res, df_opt

def main():
    args = cli()
    outdir = ROOT / "reports" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    df = ensure_data()
    h = 24 if args.fast else args.horizon
    forecaster, metrics, df_hist, df_fore = train_and_forecast(df, horizon=h)

    # Validation metrics from LSTM (already aggregated):
    mae = metrics.get("MAE", float("nan"))
    rmse = metrics.get("RMSE", float("nan"))
    print(f"Validation: MAE={mae:.2f} MW, RMSE={rmse:.2f} MW")

    # Plots
    f1 = plot_forecast(df_hist, df_fore, outdir=str(outdir))
    print("Saved", f1)

    # Optimization / dispatch
    res, df_opt = run_dispatch(df_fore)
    df_opt['shed'] = res['shed']
    f2, f3 = plot_dispatch(df_opt, outdir=str(outdir))
    print("Saved", f2, "and", f3)
    print(f"Total dispatch cost over horizon: {res['cost']:.2f}")

    # Save forecasts & dispatch
    df_fore.to_csv(ROOT / "reports" / "forecast_48h.csv", index=False)
    df_opt.to_csv(ROOT / "reports" / "dispatch_48h.csv", index=False)

if __name__ == "__main__":
    main()
