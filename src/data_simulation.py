import numpy as np
import pandas as pd

def simulate_demand_weather(start="2023-01-01", end="2025-06-30", seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="H", tz="UTC")

    # Temperature model (°C): annual + daily cycles + noise
    hours = np.arange(len(idx))
    annual = 10 * np.sin(2*np.pi*hours/(24*365.25))   # amplitude
    daily  = 5  * np.sin(2*np.pi*hours/24)
    temp = 12 + annual + daily + rng.normal(0, 2, len(idx))

    # Demand model (MW): base + sensitivity to heating/cooling + seasonality + noise
    # Higher demand when |temp - 18| is large (heating/cooling degree)
    hdd_cdd = np.abs(temp - 18)
    weekly = 50 * np.sin(2*np.pi*hours/(24*7) + np.pi/6)
    daily_demand = 80 * np.sin(2*np.pi*hours/24 - np.pi/2)  # peak in evening
    trend = 0.003 * hours  # slight growth over time
    base = 500

    demand = base + 6*hdd_cdd + 0.5*weekly + 0.8*daily_demand + trend + rng.normal(0, 10, len(idx))

    df = pd.DataFrame({
        "timestamp": idx,
        "temp_c": temp,
        "demand_mw": demand
    })
    return df

if __name__ == "__main__":
    df = simulate_demand_weather()
    df.to_csv("data/demand_weather.csv", index=False)
    print("Wrote data/demand_weather.csv with", len(df), "rows")
