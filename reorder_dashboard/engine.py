"""Inventory engine: the same functions as the notebook (Sections 7-8), with every assumption passed in explicitly."""
import math
import warnings
import numpy as np
import optuna
from scipy.stats import norm

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)   # experimental-feature notices only
warnings.filterwarnings("ignore", category=FutureWarning, message=".*constraints_func.*")
LT_VALUES = np.array([1, 2, 3, 4])


def lead_time_stats(lt_probs):
    p = np.asarray(lt_probs, float)
    mean = float((LT_VALUES * p).sum())
    return mean, float(np.sqrt(((LT_VALUES - mean) ** 2 * p).sum()))


def safety_stock_formula(fc, e, lt_mean, lt_sd, service):
    """Slide 35 formula: SS = Z * sqrt(sigma_D^2 * L + D_bar^2 * sigma_L^2); ROP = D_bar * L + SS."""
    z = norm.ppf(service)
    d_bar = max(float(np.mean(fc) + np.mean(e)), 1e-6)
    s_d = float(np.std(e, ddof=1))
    ss = z * math.sqrt(s_d**2 * lt_mean + d_bar**2 * lt_sd**2)
    return dict(D_bar=d_bar, sigma_D=s_d, Z=z, SS=ss, ROP=d_bar * lt_mean + ss)


def lead_time_demand_mc(fc, e, lt_probs, n=20_000, seed=0):
    g = np.random.default_rng(seed % (2**32))
    fc = np.asarray(fc, float)
    L = g.choice(LT_VALUES, n, p=lt_probs)
    start = g.integers(0, len(fc), n)
    k = np.arange(LT_VALUES.max())
    daily = np.maximum(fc[(start[:, None] + k[None, :]) % len(fc)] + g.choice(e, (n, len(k))), 0)
    return (daily * (k[None, :] < L[:, None])).sum(axis=1)


def safety_stock_mc(fc, e, lt_probs, service, seed=0):
    ltd = lead_time_demand_mc(fc, e, lt_probs, seed=seed)
    rop = float(np.quantile(ltd, service))
    return dict(SS=rop - float(ltd.mean()), ROP=rop, ltd=ltd)


def simulate(s, Q, fc, e, *, lt_probs, shelf, p_cost, h_cost, w_cost, k_cost, n_paths=300, days=84, warm=14, seed=7):
    """(s, Q) policy, daily review, lost sales, FIFO selling, units spoil after `shelf` days on hand."""
    g = np.random.default_rng(seed)
    fc = np.asarray(fc, float)
    dem = np.maximum(fc[np.arange(days) % len(fc)][None, :] + g.choice(e, (n_paths, days)), 0.0)
    lead = g.choice(LT_VALUES, (n_paths, days), p=lt_probs)
    stock = np.zeros((n_paths, shelf)); stock[:, 0] = s + Q
    pipe = np.zeros((n_paths, days + LT_VALUES.max() + 1))
    tot = dict(hold=0., lost=0., waste=0., orders=0., served=0., demand=0., received=0.)
    peak = np.zeros(n_paths)
    for d in range(days):
        arriving = pipe[:, d]; stock[:, 0] += arriving
        on_hand_morning = stock.sum(axis=1)
        need = dem[:, d].copy(); sold = np.zeros(n_paths)
        for a in range(shelf - 1, -1, -1):
            take = np.minimum(stock[:, a], need); stock[:, a] -= take; need -= take; sold += take
        expired = stock[:, -1].copy(); stock[:, 1:] = stock[:, :-1].copy(); stock[:, 0] = 0
        on_hand = stock.sum(axis=1)
        position = on_hand + pipe[:, d + 1:].sum(axis=1)
        place = position <= s
        rows = np.where(place)[0]; pipe[rows, d + lead[rows, d]] += Q
        if d >= warm:
            tot["hold"] += on_hand.sum(); tot["lost"] += need.sum(); tot["waste"] += expired.sum()
            tot["orders"] += place.sum(); tot["served"] += sold.sum(); tot["demand"] += dem[:, d].sum()
            tot["received"] += arriving.sum(); peak = np.maximum(peak, on_hand_morning)
    n = n_paths * (days - warm)
    return dict(cost=(h_cost * tot["hold"] + p_cost * tot["lost"] + w_cost * tot["waste"] + k_cost * tot["orders"]) / n,
                fill_rate=tot["served"] / tot["demand"], waste_rate=tot["waste"] / max(tot["received"], 1e-9),
                peak_stock=float(np.percentile(peak, 95)), orders_per_week=7 * tot["orders"] / n)


def feasible(r, d_bar, min_fill, max_waste, cap_days):
    return r["fill_rate"] >= min_fill and r["waste_rate"] <= max_waste and r["peak_stock"] <= cap_days * d_bar


def violation(r, d_bar, min_fill, max_waste, cap_days):
    return max(min_fill - r["fill_rate"], 0) + max(r["waste_rate"] - max_waste, 0) + max(r["peak_stock"] / (cap_days * d_bar) - 1, 0)


def optimise(fc, e, d_bar, *, sim_kw, min_fill, max_waste, cap_days, margin, n_trials=40, seed=0, sampler="gp"):
    """Bayesian optimisation (Gaussian process, Optuna) of (s, Q) under constraints. Same rules as the notebook."""
    nominal = (min_fill, max_waste, cap_days)
    t_fill, t_waste = min_fill + margin, max_waste - margin

    def objective(trial):
        s = trial.suggest_float("s", 0.0, 8 * d_bar); Q = trial.suggest_float("Q", 0.25 * d_bar, 6 * d_bar)
        r = simulate(s, Q, fc, e, **sim_kw); trial.set_user_attr("r", r); return r["cost"]

    def constraints(trial):
        r = trial.user_attrs["r"]
        return [t_fill - r["fill_rate"], r["waste_rate"] - t_waste, r["peak_stock"] - cap_days * d_bar]

    smp = (optuna.samplers.GPSampler(seed=seed % 10_000, constraints_func=constraints, deterministic_objective=True)
           if sampler == "gp" else optuna.samplers.RandomSampler(seed=seed % 10_000))
    study = optuna.create_study(direction="minimize", sampler=smp)
    study.optimize(objective, n_trials=n_trials)
    ok_margin = [t for t in study.trials if feasible(t.user_attrs["r"], d_bar, t_fill, t_waste, cap_days)]
    ok_real = [t for t in study.trials if feasible(t.user_attrs["r"], d_bar, *nominal)]
    if ok_margin or ok_real:
        best = min(ok_margin or ok_real, key=lambda t: t.value); is_ok = True
    else:
        best = min(study.trials, key=lambda t: (violation(t.user_attrs["r"], d_bar, *nominal), t.value)); is_ok = False
    trials = [{**t.user_attrs["r"], "n": t.number + 1, "s": t.params["s"], "Q": t.params["Q"],
               "feasible": feasible(t.user_attrs["r"], d_bar, *nominal)} for t in study.trials]
    return dict(s=best.params["s"], Q=best.params["Q"], cost=best.value, feasible=is_ok, result=best.user_attrs["r"], trials=trials)
