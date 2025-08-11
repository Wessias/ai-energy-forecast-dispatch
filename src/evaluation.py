import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-6, y_true))) * 100.0
    return {'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape}

def plot_forecast(df_hist, df_fore, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.plot(df_hist['timestamp'], df_hist['demand_mw'], label='History')
    plt.plot(df_fore['timestamp'], df_fore['y_hat'], label='Forecast')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("MW")
    plt.title("48-hour Demand Forecast")
    fig_path = os.path.join(outdir, "forecast.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return fig_path

def plot_dispatch(df_fore_opt, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.plot(df_fore_opt['timestamp'], df_fore_opt['demand_hat'], label='Demand forecast')
    plt.plot(df_fore_opt['timestamp'], df_fore_opt['g_base'], label='Baseload')
    plt.plot(df_fore_opt['timestamp'], df_fore_opt['g_peak'], label='Peaker')
    plt.plot(df_fore_opt['timestamp'], df_fore_opt['discharge'] - df_fore_opt['charge'], label='Battery net')
    if 'shed' in df_fore_opt.columns:
        plt.plot(df_fore_opt['timestamp'], df_fore_opt['shed'], label='Load shed')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("MW")
    plt.title("Economic Dispatch (MW)")
    fig_path = os.path.join(outdir, "dispatch.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(df_fore_opt['timestamp'], df_fore_opt['soc'])
    plt.xlabel("Time")
    plt.ylabel("MWh")
    plt.title("Battery State of Charge")
    fig_path2 = os.path.join(outdir, "soc.png")
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=160)
    plt.close()
    return fig_path, fig_path2
