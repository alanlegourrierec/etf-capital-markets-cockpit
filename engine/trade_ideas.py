"""
Builds, describes and simulates the trade ideas.

Every idea rests on a mean-reverting SIGNAL written the same way, whichever
module produced it.

    observed value = base x exp(x)     ("log" mode, such as a spread ratio)
    observed value = base + x          ("lin" mode, such as a slope in bp)

x is the gap to the recent norm and sd its standard deviation. With this
common form, one piece of code works out entry, target, stop, expected gain
and the Monte Carlo simulation.

The rules are sober on purpose.
  target   only half of the way back to the mean
  stop     three quarters of a standard deviation further the wrong way
  horizon  two half-lives, kept between 5 and 60 sessions
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from engine import curves, etf_analytics as ea, relative_value as rv

CDS_SPREAD_DUR = 4.3          # sensitivity of a 5Y to 1 bp of spread
Z_WINDOW = 126
TARGET_FRACTION = 0.5         # share of the gap left at the target (0.5 is half of the way)
STOP_SIGMA = 0.75             # distance of the stop, in standard deviations of the signal


# ----------------------------------------------------------------------------
# 1. Signals
# ----------------------------------------------------------------------------
def _dev(q: pd.Series, window: int, log: bool, base_scale: Optional[pd.Series] = None,
         unit: str = "bp", label: str = "") -> Dict[str, object]:
    """Gap of a series to its rolling mean, in the common signal format."""
    roll = q.rolling(window, min_periods=max(20, window // 3))
    mean, sd = roll.mean(), roll.std()
    if log:
        base = np.exp(mean) if base_scale is None else base_scale * np.exp(mean)
    else:
        base = mean
    return {"x": q - mean, "base": base, "sd_series": sd, "sd": float(sd.dropna().iloc[-1]),
            "mode": "log" if log else "lin", "unit": unit, "label": label}


def signal_for(an, kind: str, ref: Tuple) -> Dict[str, object]:
    """Rebuilds the signal of an idea from its type and its reference."""
    if kind == "capstruct":
        fv = an.capital_structure()["fv"]
        return _dev(fv["resid"][ref[0]], Z_WINDOW, True, fv["fair"][ref[0]], "bp",
                    f"{ref[0]} 5Y spread and equity-implied fair value")
    if kind == "pair":
        s = an.issuers().spread5y
        return _dev(np.log(s[ref[0]] / s[ref[1]]), Z_WINDOW, True, None, "ratio",
                    f"{ref[0]} / {ref[1]} 5Y spread ratio")
    if kind == "seniority":
        oas = an.etf(ref[0]).oas_hist
        return _dev(np.log(oas[ref[1]] / oas[ref[2]]), Z_WINDOW, True, None, "ratio",
                    f"{ref[1]} over {ref[2]}, spread ratio")
    if kind in ("curve", "carry"):
        b, _ = an.basket(ref[0])
        oas = an.etf(ref[0]).oas_hist
        peers = [i for i in b.index[b["bucket"] == b.loc[ref[1], "bucket"]] if i in oas.columns]
        med = oas[peers].median(axis=1)
        return _dev(np.log(oas[ref[1]] / med), Z_WINDOW, True, med, "bp",
                    f"{ref[1]} OAS and level implied by {b.loc[ref[1], 'bucket']} peers")
    if kind == "slope":
        s = curves.slope_history(an.rates().curves["USD"])[ref[0]]
        return _dev(s, 252, False, None, "bp", f"USD {ref[0]} slope")
    if kind == "etf":
        pdb = ea.premium_discount(an.etf(ref[0]).daily)
        return _dev(pdb, 20, False, None, "bp", f"{ref[0]} premium / discount to NAV")
    raise ValueError(f"signal inconnu : {kind}")


def signal_frame(sig: Dict[str, object]) -> pd.DataFrame:
    """Observed series, norm and 2-sigma bands, in the unit of the signal."""
    x, base, sd = sig["x"], sig["base"], sig["sd_series"]
    if sig["mode"] == "log":
        return pd.DataFrame({"value": base * np.exp(x), "mean": base,
                             "upper": base * np.exp(2 * sd), "lower": base * np.exp(-2 * sd)})
    return pd.DataFrame({"value": base + x, "mean": base,
                         "upper": base + 2 * sd, "lower": base - 2 * sd})


def _value(sig: Dict[str, object], x: float) -> float:
    base0 = float(sig["base"].dropna().iloc[-1])
    return base0 * float(np.exp(x)) if sig["mode"] == "log" else base0 + x


# ----------------------------------------------------------------------------
# 2. Building an idea
# ----------------------------------------------------------------------------
def _idea(kind: str, ref: Tuple, etf: str, segment: str, typ: str, title: str,
          long_leg: str, short_leg: str, sig: Dict[str, object], k: float,
          liquidity: float, carry_per_day: float, rationale: str) -> Optional[Dict]:
    x = sig["x"].dropna()
    sd = sig["sd"]
    if x.empty or not np.isfinite(sd) or sd <= 0:
        return None
    x0 = float(x.iloc[-1])
    z = x0 / sd
    hl = rv.half_life(x.iloc[-252:].to_numpy())
    horizon = int(np.clip(2 * hl if np.isfinite(hl) else 60, 5, 60))
    entry, target = _value(sig, x0), _value(sig, TARGET_FRACTION * x0)
    stop = _value(sig, x0 + np.sign(x0) * STOP_SIGMA * sd)
    exp_pnl, risk_ = k * abs(entry - target), k * abs(stop - entry)
    conviction = int(np.clip(1 + (abs(z) >= 1.5) + (abs(z) >= 2.0) + (abs(z) >= 2.5)
                             + (hl <= 20) - (liquidity < 35), 1, 5))
    return {
        "type": typ, "etf": etf, "segment": segment, "title": title,
        "long_leg": long_leg, "short_leg": short_leg, "z": z, "entry": entry,
        "target": target, "stop": stop, "unit": sig["unit"], "half_life": hl,
        "horizon_d": horizon, "exp_pnl_bps": exp_pnl, "risk_bps": risk_,
        "rr": exp_pnl / risk_ if risk_ > 0 else np.nan,
        "carry_bps": carry_per_day * horizon, "conviction": conviction,
        "liquidity": liquidity, "status": "Live" if abs(z) >= 2.0 else "Watch",
        "rationale": rationale, "kind": kind, "ref": "|".join(map(str, ref)), "k": k,
    }


def _issuer_liquidity(basket: pd.DataFrame, ticker: str) -> float:
    g = basket[basket["ticker"] == ticker]
    return float(g["liq_score"].mean()) if len(g) else 50.0


def build_trade_ideas(an) -> pd.DataFrame:
    """Goes through every module and returns the table of ideas."""
    ideas: List[Dict] = []
    cs = an.capital_structure()["snap"]
    pairs = an.pairs()

    for etf_id, spec in cfg.ETFS.items():
        uni = an.universe_of(etf_id)
        data = an.etf(etf_id)
        basket, _fits = an.basket(etf_id)
        seg = spec.segment

        # ---- Capital structure, credit against equity
        sel = cs[(cs["universe"] == uni) & ~cs["excluded"] & (cs["ts_z"].abs() >= 1.5)]
        sel = sel.loc[sel["ts_z"].abs().sort_values(ascending=False).index].head(4)
        for tkr, r in sel.iterrows():
            cheap = r["ts_z"] > 0
            carry = (1 if cheap else -1) * r["mkt_spread"] / 252.0
            ideas.append(_idea(
                "capstruct", (tkr,), etf_id, seg, "Capital structure",
                f"{'Long' if cheap else 'Short'} {r['name']} 5Y credit, "
                f"{'short' if cheap else 'long'} equity",
                f"{tkr} 5Y credit" if cheap else f"{tkr} equity",
                f"{tkr} equity" if cheap else f"{tkr} 5Y credit",
                signal_for(an, "capstruct", (tkr,)), CDS_SPREAD_DUR,
                _issuer_liquidity(basket, tkr), carry,
                f"5Y spread at {r['mkt_spread']:.0f} bp against {r['fair_spread']:.0f} bp implied "
                f"by a distance to default of {r['dd']:.1f}. The gap sits {r['ts_z']:+.1f} sigma "
                f"from its six-month norm. Equity hedge near {r['hedge_pct']:.1f}% of bond notional."))

        # ---- Pairs within a sector
        if len(pairs):
            sel = pairs[(pairs["universe"] == uni) & (pairs["z"] >= 1.75)
                        & (pairs["half_life"] <= 40) & (pairs["df_t"] <= -2.2)].head(5)
            for _, r in sel.iterrows():
                ideas.append(_idea(
                    "pair", (r["a"], r["b"]), etf_id, seg, "Credit pair",
                    f"Long {r['a']} against {r['b']}, {r['sector']}",
                    f"{r['a']} 5Y credit", f"{r['b']} 5Y credit",
                    signal_for(an, "pair", (r["a"], r["b"])), CDS_SPREAD_DUR * r["spread_b"],
                    0.5 * (_issuer_liquidity(basket, r["a"]) + _issuer_liquidity(basket, r["b"])),
                    (r["spread_a"] - r["spread_b"]) / 252.0,
                    f"{r['a']} trades at {r['ratio']:.2f}x {r['b']} against a six-month mean of "
                    f"{r['mean_ratio']:.2f}x. Half-life {r['half_life']:.0f} days, "
                    f"daily spread correlation {r['corr']:.2f}."))

        # ---- Seniority
        sen = an.seniority(etf_id)
        if len(sen):
            sel = sen.loc[sen["z"].abs().sort_values(ascending=False).index]
            for _, r in sel[sel["z"].abs() >= 1.5].head(3).iterrows():
                wide = r["z"] > 0
                sd_jr = float(basket.loc[r["junior"], "spread_dur"])
                gap = (r["oas_junior"] - r["oas_senior"]) / 252.0
                ideas.append(_idea(
                    "seniority", (etf_id, r["junior"], r["senior"]), etf_id, seg, "Seniority",
                    f"{r['issuer']} {'compression' if wide else 'decompression'}, "
                    f"{r['junior_tier']} against {r['senior_tier']}",
                    r["junior"] if wide else r["senior"], r["senior"] if wide else r["junior"],
                    signal_for(an, "seniority", (etf_id, r["junior"], r["senior"])),
                    sd_jr * r["oas_senior"],
                    float(basket.loc[[r["junior"], r["senior"]], "liq_score"].mean()),
                    gap if wide else -gap,
                    f"The junior bond pays {r['premium_bp']:.0f} bp over the senior, a ratio of "
                    f"{r['ratio']:.2f}x against a mean of {r['mean_ratio']:.2f}x. "
                    f"Maturity-adjusted ratio {r['adj_ratio']:.2f}x."))

        # ---- Bonds rich or cheap to the credit curve of their bucket
        cand = basket[(basket["curve_z"].abs() >= 1.5) & (basket["liq_score"] >= 35)
                      & (basket["seniority"] == "Sr Unsecured")]
        made = []
        for bond, r in cand.iterrows():
            sig = signal_for(an, "curve", (etf_id, bond))
            x = sig["x"].dropna()
            if x.empty or sig["sd"] <= 0:
                continue
            z = float(x.iloc[-1]) / sig["sd"]
            if abs(z) < 1.0 or np.sign(z) != np.sign(r["curve_z"]):
                continue
            cheap = z > 0
            made.append((abs(z), _idea(
                "curve", (etf_id, bond), etf_id, seg, "Credit curve",
                f"{'Buy' if cheap else 'Reduce'} {bond}, "
                f"{'cheap' if cheap else 'rich'} to the {r['bucket']} curve",
                bond if cheap else f"{r['bucket']} peers", f"{r['bucket']} peers" if cheap else bond,
                sig, float(r["spread_dur"]), float(r["liq_score"]),
                (1 if cheap else -1) * r["total_bps"] / 63.0,
                f"OAS {r['oas']:.0f} bp against {r['fit_oas']:.0f} bp on the fitted {r['bucket']} "
                f"curve at {r['maturity_years']:.1f} years. Three-month carry and roll "
                f"{r['total_bps']:.0f} bp.")))
        ideas += [i for _, i in sorted(made, key=lambda t: -t[0])[:3]]

        # ---- Carry and roll-down, the best cushions among liquid bonds
        liquid = basket[basket["liq_score"] >= 45].sort_values("breakeven_bp", ascending=False)
        for bond, r in liquid.head(2).iterrows():
            chg = data.oas_hist[bond].diff(63).dropna()
            vol_3m = float(chg.std()) if len(chg) > 20 else float(r["oas"]) * 0.2
            risk_ = r["spread_dur"] * 2.0 * vol_3m      # loss of value if the spread widens by 2 sigma
            ideas.append({
                "type": "Carry and roll", "etf": etf_id, "segment": seg,
                "title": f"Hold {bond} for carry and roll-down",
                "long_leg": bond, "short_leg": "Outright",
                "z": r["breakeven_bp"] / vol_3m if vol_3m > 0 else np.nan,
                "entry": r["oas"], "target": r["oas"], "stop": r["oas"] + 2.0 * vol_3m, "unit": "bp",
                "half_life": np.nan, "horizon_d": 63, "exp_pnl_bps": r["total_bps"],
                "risk_bps": risk_, "rr": r["total_bps"] / risk_ if risk_ > 0 else np.nan,
                "carry_bps": r["total_bps"], "conviction": int(np.clip(
                    2 + (r["breakeven_bp"] > vol_3m) + (r["liq_score"] >= 65), 1, 5)),
                "liquidity": float(r["liq_score"]), "status": "Live",
                "rationale": f"Carry {r['carry_bps']:.0f} bp and roll {r['roll_bps']:.0f} bp over "
                             f"three months. Spreads can widen {r['breakeven_bp']:.0f} bp before the "
                             f"position loses money, against a typical three-month move of "
                             f"{vol_3m:.0f} bp.",
                "kind": "carry", "ref": f"{etf_id}|{bond}", "k": float(r["spread_dur"])})

        # ---- ETF, an AP arbitrage window or a stretched premium
        bands = ea.arb_bands(data.daily, spec, basket, cfg.ApParams())
        last = bands.iloc[-1]
        sig = signal_for(an, "etf", (etf_id,))
        z_pd = float(sig["x"].dropna().iloc[-1]) / sig["sd"] if sig["sd"] > 0 else 0.0
        if last["signal"] != 0:
            create = last["signal"] > 0
            thr = last["upper"] if create else last["lower"]
            ideas.append({
                "type": "ETF arbitrage", "etf": etf_id, "segment": seg,
                "title": f"{'Create' if create else 'Redeem'} {etf_id}, "
                         f"{'premium' if create else 'discount'} beyond the AP threshold",
                "long_leg": "Bond basket" if create else f"{etf_id} shares",
                "short_leg": f"{etf_id} shares" if create else "Bond basket",
                "z": z_pd, "entry": last["pd_bps"], "target": thr,
                "stop": last["pd_bps"], "unit": "bp", "half_life": np.nan, "horizon_d": 2,
                "exp_pnl_bps": last["edge_bps"], "risk_bps": 0.0, "rr": np.nan, "carry_bps": 0.0,
                "conviction": int(np.clip(3 + (last["edge_bps"] > 5) + (last["edge_bps"] > 15), 1, 5)),
                "liquidity": 90.0, "status": "Live",
                "rationale": f"Price sits {last['pd_bps']:+.0f} bp to NAV against a break-even of "
                             f"{thr:+.0f} bp after basket spread, fees, hedging and funding. "
                             f"Net edge {last['edge_bps']:.0f} bp on a two-unit order.",
                "kind": "etf", "ref": etf_id, "k": 1.0})
        elif abs(z_pd) >= 1.75:
            rich = z_pd > 0
            idea = _idea(
                "etf", (etf_id,), etf_id, seg, "ETF premium",
                f"{etf_id} {'premium' if rich else 'discount'} stretched against its 20-day range",
                "NAV hedge" if rich else f"{etf_id} shares",
                f"{etf_id} shares" if rich else "NAV hedge", sig, 1.0, 90.0, 0.0,
                f"Premium at {last['pd_bps']:+.0f} bp, inside the AP band of {last['lower']:+.0f} to "
                f"{last['upper']:+.0f} bp. No primary arbitrage, a secondary-market reversion only.")
            if idea:
                idea["status"] = "Watch"
                ideas.append(idea)

    # ---- Slope of the rates curve (shared by every portfolio)
    for name in ("2s10s", "5s30s"):
        sig = signal_for(an, "slope", (name,))
        x = sig["x"].dropna()
        if x.empty or sig["sd"] <= 0 or abs(float(x.iloc[-1]) / sig["sd"]) < 1.25:
            continue
        steep = x.iloc[-1] > 0
        ideas.append(_idea(
            "slope", (name,), "ALL", "Rates overlay", "Rates curve",
            f"USD {name} {'flattener' if steep else 'steepener'}, DV01-neutral",
            f"{name.split('s')[1]}Y" if steep else f"{name.split('s')[0]}Y",
            f"{name.split('s')[0]}Y" if steep else f"{name.split('s')[1]}Y",
            sig, 1.0, 95.0, 0.0,
            f"Slope at {_value(sig, float(x.iloc[-1])):.0f} bp against a one-year mean of "
            f"{float(sig['base'].dropna().iloc[-1]):.0f} bp."))

    ideas = [i for i in ideas if i]
    if not ideas:
        return pd.DataFrame()
    df = pd.DataFrame(ideas)
    df = df.sort_values(["conviction", "exp_pnl_bps"], ascending=[False, False]).reset_index(drop=True)
    codes = {"Capital structure": "CS", "Credit pair": "PR", "Seniority": "SN",
             "Credit curve": "CV", "Carry and roll": "CR", "ETF arbitrage": "AP",
             "ETF premium": "PD", "Rates curve": "RT"}
    counter: Dict[str, int] = {}
    ids = []
    for t in df["type"]:
        counter[t] = counter.get(t, 0) + 1
        ids.append(f"{codes.get(t, 'ID')}-{counter[t]:02d}")
    df.insert(0, "id", ids)
    return df


# ----------------------------------------------------------------------------
# 3. Simulating an idea
# ----------------------------------------------------------------------------
def simulate_idea(idea: pd.Series, sig: Dict[str, object], n_paths: int = 500,
                  seed: int = 11) -> Dict[str, object]:
    """
    Monte Carlo of the signal (mean-reverting process calibrated on one year).

    Returns the fan of paths in the unit of the signal, the probability of
    reaching the target before the stop, and the P&L distribution in bp.
    """
    x = sig["x"].dropna()
    x0 = float(x.iloc[-1])
    _, kappa, sigma = rv.ou_fit(x.to_numpy(), 252)
    horizon = int(idea["horizon_d"])
    paths = rv.simulate_ou(x0, 0.0, kappa, sigma, horizon, n_paths, seed)
    s = np.sign(x0) if x0 != 0 else 1.0
    x_tgt, x_stop = TARGET_FRACTION * x0, x0 + s * STOP_SIGMA * sig["sd"]
    hit_t = (paths * s <= x_tgt * s)
    hit_s = (paths * s >= x_stop * s)
    first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), horizon + 1)
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), horizon + 1)
    exit_idx = np.minimum(np.minimum(first_t, first_s), horizon)
    x_exit = paths[np.arange(n_paths), exit_idx]
    base0 = float(sig["base"].dropna().iloc[-1])
    to_val = (lambda v: base0 * np.exp(v)) if sig["mode"] == "log" else (lambda v: base0 + v)
    pnl = float(idea["k"]) * (to_val(x0) - to_val(x_exit)) * s
    carry_day = float(idea["carry_bps"]) / max(horizon, 1)
    pnl = pnl + carry_day * exit_idx
    q = np.percentile(to_val(paths), [5, 25, 50, 75, 95], axis=0)
    fan = pd.DataFrame(q.T, columns=["p05", "p25", "p50", "p75", "p95"])
    return {"fan": fan, "p_target": float((first_t < first_s).mean()),
            "p_stop": float((first_s < first_t).mean()),
            "exp_pnl": float(pnl.mean()), "pnl": pnl, "kappa": kappa, "sigma": sigma,
            "median_days": float(np.median(exit_idx))}
