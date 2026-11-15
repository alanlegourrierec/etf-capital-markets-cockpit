"""
Access to LSEG data (Workspace or Eikon) from the desktop.

The client tries the installed libraries in this order.
    1. lseg.data       LSEG Data Library, the current one
    2. refinitiv.data  Refinitiv Data Library, the previous version
    3. eikon           Eikon Data API, legacy

It exposes three operations that do not depend on the library.
    history()       time series, one frame per RIC
    wide_history()  one series per RIC, trying several candidate fields
    snapshot()      point-in-time data (reference, analytics, baskets)

No exception reaches the app. A failure comes back as an empty result, and
the provider then moves that block to the simulator.

This module has not been run against a real Workspace session. The get_history
and get_data signatures were checked on lseg-data 2.1.1, and the plumbing is
tested with a fake library in tests/test_provider_mock.py.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import threading
from contextlib import suppress
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _is_tr(field: str) -> bool:
    return field.upper().startswith("TR.")


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted time index, without time zone or duplicates."""
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.isna()]
    return out[~out.index.duplicated(keep="last")].sort_index()


def pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """First column of df that matches a candidate field and holds data."""
    upper = {str(c).strip().upper(): c for c in df.columns}
    for cand in candidates:
        col = upper.get(cand.upper())
        if col is not None and df[col].notna().any():
            return col
    return None


class LsegClient:
    """One front for lseg.data, refinitiv.data or eikon."""

    def __init__(self, app_key: str = ""):
        self.app_key = app_key or ""
        self.lib = None            # Python module actually in use
        self.kind = ""             # "ld", "rd" or "ek"
        self.label = ""            # readable name of the library
        self.connected = False
        self.error = "connection not attempted"
        self.last_error = ""
        self._hdr: dict = {}       # argument that asks for field names in headers

    # --------------------------------------------------------------- session
    def connect(self, timeout: float = 25.0) -> bool:
        """Opens a session. Bounded in time so the launch never hangs."""
        worker = threading.Thread(target=self._connect_impl, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            self.connected = False
            self.error = f"timed out after {timeout:.0f} s while opening the session"
        return self.connected

    def _connect_impl(self) -> None:
        errors: List[str] = []
        for modname, kind, label in (("lseg.data", "ld", "lseg-data"),
                                     ("refinitiv.data", "rd", "refinitiv-data")):
            try:
                lib = importlib.import_module(modname)
            except Exception:
                errors.append(f"{label} not installed")
                continue
            try:
                # The library logs about ten error lines when Workspace is closed.
                # They are muted for the attempt, our own message is enough.
                logging.disable(logging.CRITICAL)
                try:
                    session = (lib.open_session(app_key=self.app_key) if self.app_key
                               else lib.open_session())
                finally:
                    logging.disable(logging.NOTSET)
                if str(getattr(session, "open_state", "")).endswith("Opened"):
                    self.lib, self.kind, self.label = lib, kind, label
                    self._hdr = self._header_kwargs(lib)
                    self.connected, self.error = True, ""
                    return
                errors.append(f"{label}: session closed, is Workspace running?")
                with suppress(Exception):
                    lib.close_session()
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        try:
            ek = importlib.import_module("eikon")
            if not self.app_key:
                errors.append("eikon: no app key (LSEG_APP_KEY)")
            else:
                ek.set_app_key(self.app_key)
                df, _ = ek.get_data("EUR=", ["BID"])      # end-to-end check
                if df is not None and len(df):
                    self.lib, self.kind, self.label = ek, "ek", "eikon"
                    self.connected, self.error = True, ""
                    return
                errors.append("eikon: no answer from the proxy")
        except ImportError:
            errors.append("eikon not installed")
        except Exception as exc:
            errors.append(f"eikon: {exc}")
        self.error = " | ".join(errors)[:300]

    @staticmethod
    def _header_kwargs(lib) -> dict:
        """Asks for field names (TR.xxx) in headers instead of their titles."""
        params = inspect.signature(lib.get_data).parameters
        if "header_type" in params and hasattr(lib, "HeaderType"):
            return {"header_type": lib.HeaderType.NAME}
        if "use_field_names_in_headers" in params:
            return {"use_field_names_in_headers": True}
        return {}

    def close(self) -> None:
        if self.connected and self.kind in ("ld", "rd"):
            with suppress(Exception):
                self.lib.close_session()
        self.connected = False

    # ------------------------------------------------------------ time series
    def history(self, rics: Iterable[str], fields: Optional[Sequence[str]] = None,
                start=None, end=None, interval: str = "daily",
                chunk: int = 40) -> Dict[str, pd.DataFrame]:
        """Time series. Returns {ric: DataFrame indexed by date}."""
        out: Dict[str, pd.DataFrame] = {}
        if not self.connected:
            return out
        rics = [r for r in dict.fromkeys(rics) if r]
        for i in range(0, len(rics), chunk):
            part = rics[i:i + chunk]
            try:
                raw = self._history_call(part, fields, start, end, interval)
                out.update(self._split(raw, part, fields))
            except Exception as exc:
                self.last_error = f"history {part[0]}… : {exc}"[:200]
        return out

    def _history_call(self, part, fields, start, end, interval) -> pd.DataFrame:
        if self.kind in ("ld", "rd"):
            kwargs = dict(universe=list(part), interval=interval, start=start, end=end,
                          **self._hdr)
            if fields:
                kwargs["fields"] = list(fields)
            return self.lib.get_history(**kwargs)
        # --- Eikon Data API, get_timeseries for prices and get_data for TR fields
        tr_fields = [f for f in (fields or []) if _is_tr(f)]
        if tr_fields:
            return self._eikon_tr_history(part, tr_fields[0], start, end)
        return self.lib.get_timeseries(list(part), start_date=start, end_date=end,
                                       interval=interval)

    def _eikon_tr_history(self, part, field: str, start, end) -> pd.DataFrame:
        """History of one TR field through get_data (long format), turned wide."""
        params = {"SDate": str(start)[:10], "EDate": str(end)[:10], "Frq": "D"}
        df, _ = self.lib.get_data(list(part), [field + ".date", field], params,
                                  field_name=True)
        cols = list(df.columns)
        date_col = next(c for c in cols if "DATE" in str(c).upper())
        val_col = next(c for c in cols[1:] if c != date_col)
        wide = df.pivot_table(index=date_col, columns=cols[0], values=val_col, aggfunc="last")
        wide.columns = pd.MultiIndex.from_product([wide.columns, [field]])
        return wide

    @staticmethod
    def _split(raw: pd.DataFrame, part: Sequence[str],
               fields: Optional[Sequence[str]]) -> Dict[str, pd.DataFrame]:
        """Brings the three shapes get_history can return to {ric: DataFrame}."""
        if raw is None or len(raw) == 0:
            return {}
        raw = _clean_index(raw)
        out: Dict[str, pd.DataFrame] = {}
        if isinstance(raw.columns, pd.MultiIndex):           # several RICs and several fields
            for ric in raw.columns.get_level_values(0).unique():
                out[str(ric)] = raw[ric]
        elif len(part) == 1:                                 # a single RIC
            out[part[0]] = raw
        else:                                                # several RICs, a single field
            name = fields[0] if fields else "VALUE"
            for ric in raw.columns:
                out[str(ric)] = raw[[ric]].rename(columns={ric: name})
        clean = {}
        for ric, df in out.items():
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if len(df):
                df.columns = [str(c).strip() for c in df.columns]
                clean[ric] = df
        return clean

    def wide_history(self, rics: Iterable[str], candidates: Sequence[str],
                     start=None, end=None, interval: str = "daily") -> pd.DataFrame:
        """
        One series per RIC (dates by RIC), taking for each the first candidate
        field that answers. Quote fields are asked for in one call, TR fields
        one at a time.
        """
        rics = [r for r in dict.fromkeys(rics) if r]
        series: Dict[str, pd.Series] = {}
        quote_fields = [c for c in candidates if not _is_tr(c)]
        if quote_fields:
            for ric, df in self.history(rics, None, start, end, interval).items():
                col = pick_column(df, quote_fields)
                if col is not None:
                    series[ric] = df[col]
        for cand in (c for c in candidates if _is_tr(c)):
            missing = [r for r in rics if r not in series]
            if not missing:
                break
            for ric, df in self.history(missing, [cand], start, end, interval).items():
                col = pick_column(df, [cand]) or (df.columns[0] if df.shape[1] == 1 else None)
                if col is not None and df[col].notna().any():
                    series[ric] = df[col]
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).sort_index()

    # ------------------------------------------------------------- snapshots
    def snapshot(self, rics: Iterable[str], fields: Sequence[str],
                 parameters: Optional[dict] = None, chunk: int = 50) -> pd.DataFrame:
        """Point-in-time data. First column is the instrument, then one column per field."""
        if not self.connected:
            return pd.DataFrame()
        rics = [r for r in dict.fromkeys(rics) if r]
        frames = []
        for i in range(0, len(rics), chunk):
            part = rics[i:i + chunk]
            try:
                if self.kind in ("ld", "rd"):
                    df = self.lib.get_data(universe=list(part), fields=list(fields),
                                           parameters=parameters or {}, **self._hdr)
                else:
                    df, _ = self.lib.get_data(list(part), list(fields), parameters or {},
                                              field_name=True)
                if df is not None and len(df):
                    frames.append(df)
            except Exception as exc:
                self.last_error = f"snapshot {part[0]}… : {exc}"[:200]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out.columns = [str(c).strip() for c in out.columns]
        return out

    def snapshot_roles(self, rics: Iterable[str],
                       roles: Dict[str, Sequence[str]],
                       parameters: Optional[dict] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fetches several "roles" (coupon, maturity and so on), each with its list
        of candidate fields. Returns a DataFrame with one row per instrument and
        one column per role found, and the list of roles that are missing.
        """
        fields = list(dict.fromkeys(f for cands in roles.values() for f in cands))
        raw = self.snapshot(rics, fields, parameters)
        if raw.empty:
            return pd.DataFrame(), list(roles)
        inst = raw.columns[0]
        out = pd.DataFrame({"instrument": raw[inst].astype(str)})
        missing = []
        for role, cands in roles.items():
            col = pick_column(raw, cands)
            if col is None:
                missing.append(role)
            else:
                out[role] = raw[col].to_numpy()
        return out, missing
