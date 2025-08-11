# AI Time Series + Optimization: Energy Demand Forecasting & Economic Dispatch

**What it is:** A hybrid project that forecasts short-term electricity demand with ML and then optimizes
a simple generation + battery dispatch to meet that demand at minimum cost.

- **AI/ML**: **PyTorch LSTM** sequence model (direct multi-horizon decoder) trained with engineered lag, rolling, and Fourier features.
- **Time series**: Hourly data with strong daily/weekly/annual seasonality + temperature effects.
- **Optimization**: Linear program (SciPy's `linprog`) for economic dispatch with baseload, peaker, and a battery.
- **Stack**: Python 3, NumPy, Pandas, scikit-learn, SciPy, Matplotlib.

> This is designed to be **portfolio-ready**: clean structure, reproducible, documented, and visual.

---

## Project Structure

```
ai_timeseries_optimization/
├── data/
│   └── demand_weather.csv         # Simulated hourly data (2023-01-01 .. 2025-06-30)
├── models/
│   └── (saved models & scalers)
├── reports/
│   └── figures/                   # Matplotlib output
├── src/
│   ├── data_simulation.py         # Creates the dataset with realistic patterns
│   ├── features.py                # Time-series feature engineering
│   ├── forecaster.py              # ML training & multi-step forecasting
│   ├── optimizer.py               # Economic dispatch LP using SciPy linprog
│   ├── evaluation.py              # Metrics & plotting
│   └── main.py                    # End-to-end pipeline
├── requirements.txt
└── README.md
```

---

## Quickstart

1. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**  
   ```bash
   python -m src.main
   ```

This will:
- Generate or reuse `data/demand_weather.csv`
- Train ML models
- Produce **48-hour** demand forecasts
- Solve the **economic dispatch** LP
- Save plots in `reports/figures` and print summary metrics

---

## Modeling Details

### Forecasting
- **Targets**: Hourly demand (MW)
- **Horizon**: 48 hours ahead, trained as a set of **direct models** (one per horizon step)
- **Models**: PyTorch `nn.LSTM` with a small MLP head to predict the full horizon in one shot
- **Features**:
  - Lags: 1..48 hours
  - Rolling means: 24h, 168h (weekly)
  - Calendar: hour-of-day, day-of-week, month
  - Fourier seasonal terms (daily, weekly, annual)
  - Temperature + lagged temperature

### Optimization
- Decision variables per hour *t* (continuous LP):
  - `g_base_t`  : baseload generation (MW) — cheap, capacity-limited, ramp-limited
  - `g_peak_t`  : peaker generation (MW) — expensive, flexible
  - `ch_t`      : battery charge power (MW)
  - `dis_t`     : battery discharge power (MW)
  - `soc_t`     : battery state of charge (MWh)
- Objective: **minimize cost** = sum(c_base * g_base_t + c_peak * g_peak_t + eps*(ch_t + dis_t))
- Constraints:
  - **Power balance**: g_base_t + g_peak_t + dis_t - ch_t = demand_hat_t
  - **Capacity limits** for generators and battery power
  - **SOC dynamics** with efficiency and bounds, fixed initial SOC
  - **Ramping** for baseload: |g_base_t - g_base_{t-1}| <= ramp
  - (LP relaxation; no binaries. A small epsilon discourages charge & discharge simultaneously.)

---

## Why this is a strong portfolio piece

- Shows **end-to-end thinking**: data → ML forecasting → optimization → business objective.
- Uses **engineering-relevant** constraints (ramping, storage efficiency) that mirror real systems.
- Clean, reproducible code with visual outputs and clear README.

---

## Extending the project

- Swap GBMs for **LSTMs/Transformers** (PyTorch or Keras) for sequence modeling.
- Add **price uncertainty** and solve a **stochastic** or **robust** dispatch.
- Add **emissions** or **renewable availability** constraints and objectives.
- Enforce **no simultaneous charge/discharge** using MILP (e.g., with OR-Tools or Pyomo + CBC/HiGHS).

---

## License

MIT


---

## VS Code Quickstart
1. **Open Folder** in VS Code → this project.
2. `Ctrl+Shift+P` → **Tasks: Run Task** → **Install requirements** (creates venv + installs deps).
3. Press **F5** to run **Run main (48h forecast)**, or choose **Run main (fast smoke test)**.
4. Alternatively: `python run.py` or `make run`.
