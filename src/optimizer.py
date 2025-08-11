import numpy as np
from scipy.optimize import linprog

def economic_dispatch_lp(demand_forecast,
                         T=None,
                         c_base=40.0, c_peak=100.0, eps_cd=0.1,
                         cap_base=500.0, cap_peak=500.0,
                         ramp_base=80.0,
                         cap_charge=120.0, cap_discharge=120.0,
                         eff_charge=0.95, eff_discharge=0.95,
                         ecap_batt=600.0, soc0=300.0, soc_min=60.0, soc_max=600.0,
                         allow_shed=True, shed_cost=1e4, auto_scale_caps=True):
    d = np.array(demand_forecast, dtype=float)
    if T is None:
        T = len(d)
    H = T

    # Auto-scale if total gen < ~max demand
    if auto_scale_caps:
        maxd = float(np.max(d))
        if cap_base + cap_peak < 1.1 * maxd:
            ratio = cap_base / max(1e-6, (cap_base + cap_peak))
            total_needed = 1.2 * maxd
            cap_base = max(50.0, ratio * total_needed)
            cap_peak = max(50.0, (1 - ratio) * total_needed)

    shed_var = 1 if allow_shed else 0
    n_var_per_t = 5 + shed_var  # [g_base, g_peak, ch, dis, soc, (shed)]
    n_var = n_var_per_t * H

    # Objective
    c = np.zeros(n_var)
    for t in range(H):
        off = n_var_per_t * t
        c[off+0] = c_base
        c[off+1] = c_peak
        c[off+2] = eps_cd
        c[off+3] = eps_cd
        if shed_var: c[off+5] = shed_cost

    # Bounds
    bounds = []
    for _ in range(H):
        bounds += [(0, cap_base), (0, cap_peak), (0, cap_charge), (0, cap_discharge), (soc_min, soc_max)]
        if shed_var: bounds += [(0, None)]  # shed >= 0

    # Equalities: power balance + SOC dynamics
    A_eq, b_eq = [], []

    # Power balance: g_base + g_peak + dis - ch + shed = demand
    for t in range(H):
        row = np.zeros(n_var); off = n_var_per_t * t
        row[off+0] = 1; row[off+1] = 1; row[off+3] = 1; row[off+2] = -1
        if shed_var: row[off+5] = 1
        A_eq.append(row); b_eq.append(d[t])

    # SOC: soc_t - soc_{t-1} - eff_c*ch + (1/eff_d)*dis = 0; soc_-1 = soc0
    for t in range(H):
        row = np.zeros(n_var); off = n_var_per_t * t
        row[off+4] = 1
        row[off+2] += -eff_charge
        row[off+3] += 1.0/eff_discharge
        if t == 0:
            A_eq.append(row); b_eq.append(soc0)
        else:
            row[off - n_var_per_t + 4] += -1
            A_eq.append(row); b_eq.append(0.0)

    A_eq = np.vstack(A_eq); b_eq = np.array(b_eq, float)

    # Inequalities: baseload ramping
    A_ub, b_ub = [], []
    for t in range(1, H):
        row = np.zeros(n_var); off = n_var_per_t * t
        row[off+0] = 1; row[off - n_var_per_t + 0] = -1
        A_ub.append(row); b_ub.append(ramp_base)
        row = np.zeros(n_var); off = n_var_per_t * t
        row[off+0] = -1; row[off - n_var_per_t + 0] = 1
        A_ub.append(row); b_ub.append(ramp_base)
    A_ub = np.vstack(A_ub) if A_ub else None
    b_ub = np.array(b_ub, float) if A_ub is not None else None

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    x = res.x
    def sl(i): return x[i::n_var_per_t]
    g_base = sl(0); g_peak = sl(1); ch = sl(2); dis = sl(3); soc = sl(4)
    shed = sl(5) if shed_var else np.zeros(H)
    total_cost = c_base*np.sum(g_base) + c_peak*np.sum(g_peak) + eps_cd*(np.sum(ch)+np.sum(dis)) + shed_cost*np.sum(shed)

    return {"g_base": g_base, "g_peak": g_peak, "charge": ch, "discharge": dis,
            "soc": soc, "shed": shed, "cost": total_cost, "success": res.success, "message": res.message}
