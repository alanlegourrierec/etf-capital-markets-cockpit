"""
Checks the LSEG connection. Run it on the machine where Workspace is open.

    python scripts/lseg_check.py

The script opens a session, then queries block by block the RICs and fields
declared in config.py. For each item it says what answers and what is missing.
That gives the exact list of lines to fix in config.py (RICs, LSEG_FIELDS)
before the dashboard can switch to real data.
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                     # noqa: E402
from data.lseg_client import LsegClient, pick_column     # noqa: E402

F = cfg.LSEG_FIELDS
OK, KO = "[ OK ]", "[ -- ]"


def line(ok: bool, text: str) -> None:
    print(f"  {OK if ok else KO} {text}")


def main() -> int:
    print("=" * 74)
    print(" LSEG connection check")
    print("=" * 74)
    client = LsegClient(cfg.LSEG_APP_KEY)
    client.connect()
    if not client.connected:
        print(f"\n No session. {client.error}")
        print(" Check that Workspace is open and signed in, that the app key is set\n"
              " (LSEG_APP_KEY variable or .lseg_app_key file) and that one of\n"
              " lseg-data, refinitiv-data or eikon is installed.")
        return 1
    print(f"\n Session open through {client.label}"
          f" ({'app key provided' if cfg.LSEG_APP_KEY else 'app key read from the library config'})")
    end = pd.Timestamp.today().normalize()
    start = (end - timedelta(days=120)).strftime("%Y-%m-%d")
    end = end.strftime("%Y-%m-%d")

    print("\n 1. Sovereign curves")
    for ccy, ricmap in cfg.GOVT_RICS.items():
        wide = client.wide_history(list(ricmap.values()), F["yield"], start, end)
        found = [t for t, r in ricmap.items() if r in wide.columns]
        missing = [r for r in ricmap.values() if r not in wide.columns]
        line(len(found) >= 4, f"{ccy} {len(found)}/{len(ricmap)} tenors"
             + (f", missing {missing}" if missing else ""))

    print("\n 2. ETF price, NAV, shares, iNAV, benchmark")
    for etf_id, spec in cfg.ETFS.items():
        if not spec.ric:
            line(False, f"{etf_id} has no RIC in config.ETFS (simulated sleeve)")
            continue
        hist = client.history([spec.ric], None, start, end).get(spec.ric)
        if hist is None:
            line(False, f"{etf_id} no history for {spec.ric} ({client.last_error})")
            continue
        print(f"     {etf_id} {spec.ric} fields returned {list(hist.columns)[:14]}")
        for role in ("close", "bid", "ask", "volume"):
            col = pick_column(hist, F[role])
            line(col is not None, f"{etf_id} {role:<7} {col or 'none of ' + str(F[role])}")
        for role in ("nav", "shares_out"):
            got = None
            for cand in F[role]:
                w = client.wide_history([spec.ric], [cand], start, end)
                if not w.empty:
                    got = f"{cand} ({len(w)} points, last {w.iloc[-1, 0]:,.4f})"
                    break
            line(got is not None, f"{etf_id} {role:<10} {got or 'none of ' + str(F[role])}")
        line(bool(spec.inav_ric), f"{etf_id} iNAV       "
             + (spec.inav_ric or "inav_ric is empty, the intraday panel stays off"))
        line(bool(spec.bench_ric), f"{etf_id} benchmark  "
             + (spec.bench_ric or "bench_ric is empty, no tracking error on real data"))

    print("\n 3. Basket and bond analytics")
    for etf_id, spec in cfg.ETFS.items():
        if not spec.ric:
            continue
        hold, missing = client.snapshot_roles(
            [spec.ric], {"ric": F["holding_ric"], "name": F["holding_name"],
                         "weight": F["holding_weight"]}, {"EndNum": 4000})
        if hold.empty or "ric" in missing:
            line(False, f"{etf_id} holdings not available ({client.last_error or 'empty answer'})")
            continue
        hold = hold.dropna(subset=["ric"])
        weight = pd.to_numeric(hold.get("weight"), errors="coerce").sum()
        line(len(hold) >= 20, f"{etf_id} {len(hold)} lines, total weight {weight:.1f}%")
        sample = list(hold["ric"].astype(str).head(15))
        roles = {k[5:]: F[k] for k in F if k.startswith("bond_")}
        ana, missing = client.snapshot_roles(sample, roles)
        for role in roles:
            ok = role not in missing and not ana.empty
            fill = f"{ana[role].notna().mean():.0%} filled" if ok else f"none of {roles[role]}"
            line(ok, f"{etf_id} bond_{role:<10} {fill}")
        oas = client.wide_history(sample, F["oas"], start, end)
        line(not oas.empty, f"{etf_id} spread history for {oas.shape[1]}/{len(sample)} bonds"
             + ("" if not oas.empty else f", none of {F['oas']}"))

    print("\n 4. Issuers (Merton module)")
    rows = (cfg.HY_ISSUERS + cfg.IG_ISSUERS)[:6]
    eq = client.wide_history([r[2] for r in rows], F["close"], start, end)
    line(eq.shape[1] > 0, f"share prices, {eq.shape[1]}/{len(rows)} tested RICs answer")
    fund, missing = client.snapshot_roles([r[2] for r in rows],
                                          {"mktcap": F["mktcap"], "total_debt": F["total_debt"],
                                           "st_debt": F["st_debt"]})
    for role in ("mktcap", "total_debt", "st_debt"):
        line(role not in missing and not fund.empty, f"fundamental {role}")
    line(len(cfg.CDS_RICS) >= 10, f"{len(cfg.CDS_RICS)} 5Y CDS RICs filled in (10 are needed for "
         "this block to go live)")

    print("\n Each [ -- ] line points to a RIC or a field to fix in config.py.\n"
          " The Data Item Browser in Workspace (DIB) gives the exact name of TR fields.\n")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
