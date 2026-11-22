"""Cross-domain real-world continuous-time interacting-network dataset loaders.

All loaders compute SHA-256 checksums on downloaded public source archives, enforce strict
leak-free chronological or regime-shift splits, and cache compact deterministic ``.npz`` arrays
under ``data/raw/<domain>/``:

1. UCI Beijing Multi-Site Air Quality (Zhang et al., Proc. R. Soc. A 2017):
   12 interacting monitoring stations across Beijing (420,768 station-hours, 2013-2017),
   8 regional meteorological drivers -> 12-station log1p(PM2.5) 48-hour trajectories with
   4-fold leave-one-season-out regime shift and partial-observability support.
2. METR-LA Highway Traffic Sensor Network (Li et al., ICLR 2018):
   207 Los Angeles highway loop detectors (5-min intervals, 4 months) with physical directed
   highway graph (adj_mx.pkl). Extracts the densest connected S-sensor corridor subgraph
   for closed-loop 1-hour autonomous congestion-shockwave forecasting & road-graph discovery.
3. Binance Vision Multi-Asset Order-Flow & Volatility Contagion (May-Aug 2024):
   16 liquid interacting crypto markets (35,424 aligned 5-min bars), mapping causal signed
   order-flow / volume shock drives to realized log-volatility cascades across correlated
   training windows, chronological August 2024 flash-crash holdouts, idiosyncratic single-asset
   shocks, and partial observability (BTC + ETH hidden).
4. SNAP Temporal Reddit Hyperlink Network (Kumar et al., WWW 2018):
   571,927 timestamped directed subreddit-to-subreddit interactions (2014-2017) aggregated into
   continuous-time community activity cascades across the top 24 interacting subreddits, with
   chronological splits and held-out directed hyperlink graph evaluation.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pandas as pd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# 1. UCI Beijing Multi-Site Air Quality (12 stations, 2013-2017)
# =============================================================================

BEIJING_STATIONS: tuple[str, ...] = (
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
)

# Published GPS coordinates (lat, lon) of the 12 Beijing air-quality monitoring stations
BEIJING_STATION_COORDS: dict[str, tuple[float, float]] = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

WIND_DIR_DEG: dict[str, float] = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

BEIJING_DRIVER_NAMES: tuple[str, ...] = (
    "Wind_North",
    "Wind_East",
    "Wind_Speed",
    "Rain",
    "Temp",
    "Pressure",
    "Dewpoint",
    "Diurnal_Sin",
)


@dataclass(frozen=True, kw_only=True)
class BeijingAirQualityData:
    """UCI Beijing 12-station hourly atmospheric haze & washout episodes (48h windows)."""

    stations: tuple[str, ...]
    driver_names: tuple[str, ...]
    drivers: np.ndarray  # (E, 48, 8) standardized meteorological drivers
    targets: np.ndarray  # (E, 48, 12) log1p(PM2.5) zero-origin per episode or raw log1p
    targets_raw_log: np.ndarray  # (E, 48, 12) log1p(PM2.5)
    seasons: np.ndarray  # (E,) 0=Spring, 1=Summer, 2=Autumn, 3=Winter
    years: np.ndarray  # (E,) calendar year of episode start
    washout_flags: np.ndarray  # (E,) bool severe haze buildup + cold-front washout episodes
    total_station_hours: int
    file_sha256: dict[str, str]


def load_beijing_air_quality(raw_dir: Path, episode_hours: int = 48) -> BeijingAirQualityData:
    """Load all 12 Beijing air-quality station CSVs into non-overlapping 48h episodes."""
    zip_path = raw_dir / "beijing_aq.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}")
    hashes = {"beijing_aq.zip": _sha256_file(zip_path)}

    cache_npz = raw_dir / f"beijing_aq_12st_{episode_hours}h_cache.npz"
    if cache_npz.exists():
        z = np.load(cache_npz, allow_pickle=False)
        hashes[cache_npz.name] = _sha256_file(cache_npz)
        return BeijingAirQualityData(
            stations=BEIJING_STATIONS,
            driver_names=BEIJING_DRIVER_NAMES,
            drivers=np.asarray(z["drivers"], dtype=np.float64),
            targets=np.asarray(z["targets"], dtype=np.float64),
            targets_raw_log=np.asarray(z["targets_raw_log"], dtype=np.float64),
            seasons=np.asarray(z["seasons"], dtype=np.int64),
            years=np.asarray(z["years"], dtype=np.int64),
            washout_flags=np.asarray(z["washout_flags"], dtype=bool),
            total_station_hours=int(z["total_station_hours"]),
            file_sha256=hashes,
        )

    csv_dir = raw_dir / "PRSA_Data_20130301-20170228"
    station_dfs: dict[str, pd.DataFrame] = {}
    for st in BEIJING_STATIONS:
        matches = list(csv_dir.glob(f"PRSA_Data_{st}_*.csv"))
        if not matches:
            raise FileNotFoundError(f"Missing station CSV for {st} in {csv_dir}")
        station_dfs[st] = pd.read_csv(matches[0])

    n_rows = len(station_dfs[BEIJING_STATIONS[0]])
    pm_mat = np.zeros((n_rows, len(BEIJING_STATIONS)), dtype=np.float64)
    pm_valid = np.ones((n_rows, len(BEIJING_STATIONS)), dtype=bool)
    met_cols = ("WSPM", "RAIN", "TEMP", "PRES", "DEWP")
    met_stack = {c: np.zeros((n_rows, len(BEIJING_STATIONS)), dtype=np.float64) for c in met_cols}
    wn_stack = np.zeros((n_rows, len(BEIJING_STATIONS)), dtype=np.float64)
    we_stack = np.zeros((n_rows, len(BEIJING_STATIONS)), dtype=np.float64)

    for j, st in enumerate(BEIJING_STATIONS):
        df = station_dfs[st]
        pm_raw = df["PM2.5"].to_numpy(dtype=np.float64)
        pm_valid[:, j] = np.isfinite(pm_raw)
        pm_mat[:, j] = pd.Series(pm_raw).interpolate(limit=4).to_numpy(dtype=np.float64)
        for c in met_cols:
            met_stack[c][:, j] = (
                pd.Series(df[c].to_numpy(dtype=np.float64)).interpolate(limit=4).to_numpy()
            )
        deg = df["wd"].map(WIND_DIR_DEG).to_numpy(dtype=np.float64)
        rad = np.deg2rad(deg)
        wspd = met_stack["WSPM"][:, j]
        # Northerly wind (blowing FROM North = 0 deg -> cos=+1) clears Beijing haze
        wn_stack[:, j] = np.where(np.isfinite(rad), wspd * np.cos(rad), np.nan)
        we_stack[:, j] = np.where(np.isfinite(rad), wspd * np.sin(rad), np.nan)

    ref = station_dfs[BEIJING_STATIONS[0]]
    months = ref["month"].to_numpy(dtype=np.int64)
    years_all = ref["year"].to_numpy(dtype=np.int64)
    hours_all = ref["hour"].to_numpy(dtype=np.float64)
    # Meteorological seasons: 0=Spring(3-5), 1=Summer(6-8), 2=Autumn(9-11), 3=Winter(12,1,2)
    season_map = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3, 1: 3, 2: 3}
    seasons_all = np.array([season_map[int(m)] for m in months], dtype=np.int64)

    # Regional mean meteorological drivers (S = 8)
    drv_raw = np.column_stack(
        [
            np.nanmean(wn_stack, axis=1),
            np.nanmean(we_stack, axis=1),
            np.nanmean(met_stack["WSPM"], axis=1),
            np.log1p(np.maximum(0.0, np.nanmean(met_stack["RAIN"], axis=1))),
            np.nanmean(met_stack["TEMP"], axis=1),
            np.nanmean(met_stack["PRES"], axis=1),
            np.nanmean(met_stack["DEWP"], axis=1),
            np.sin(2.0 * np.pi * hours_all / 24.0),
        ]
    )
    pm_log = np.log1p(np.maximum(0.0, pm_mat))

    # Extract non-overlapping 48-hour episodes with 100% finite data across all 12 stations
    ep_drv, ep_tgt, ep_raw, ep_sea, ep_yr, ep_wash = [], [], [], [], [], []
    n_eps = n_rows // episode_hours
    for e_i in range(n_eps):
        s0 = e_i * episode_hours
        s1 = s0 + episode_hours
        d_win = drv_raw[s0:s1]
        y_win = pm_log[s0:s1]
        if not (np.all(np.isfinite(d_win)) and np.all(np.isfinite(y_win))):
            continue
        if np.mean(pm_valid[s0:s1]) < 0.92:
            continue
        ep_drv.append(d_win)
        ep_raw.append(y_win)
        ep_tgt.append(y_win - y_win[0:1, :])
        ep_sea.append(int(seasons_all[s0]))
        ep_yr.append(int(years_all[s0]))
        # Severe washout storm flag: episode swings by >= 1.5 log-units (4.5x PM2.5 drop/rise)
        mean_curve = np.mean(y_win, axis=1)
        ep_wash.append(bool((np.max(mean_curve) - np.min(mean_curve)) >= 1.5))

    drivers_arr = np.stack(ep_drv, axis=0)
    targets_arr = np.stack(ep_tgt, axis=0)
    raw_arr = np.stack(ep_raw, axis=0)
    seasons_arr = np.asarray(ep_sea, dtype=np.int64)
    years_arr = np.asarray(ep_yr, dtype=np.int64)
    wash_arr = np.asarray(ep_wash, dtype=bool)
    tot_hrs = int(n_rows * len(BEIJING_STATIONS))

    np.savez(
        cache_npz,
        drivers=drivers_arr,
        targets=targets_arr,
        targets_raw_log=raw_arr,
        seasons=seasons_arr,
        years=years_arr,
        washout_flags=wash_arr,
        total_station_hours=np.int64(tot_hrs),
    )
    hashes[cache_npz.name] = _sha256_file(cache_npz)
    return BeijingAirQualityData(
        stations=BEIJING_STATIONS,
        driver_names=BEIJING_DRIVER_NAMES,
        drivers=drivers_arr,
        targets=targets_arr,
        targets_raw_log=raw_arr,
        seasons=seasons_arr,
        years=years_arr,
        washout_flags=wash_arr,
        total_station_hours=tot_hrs,
        file_sha256=hashes,
    )


# =============================================================================
# 2. METR-LA Highway Traffic Sensor Network (207 loop detectors + physical road graph)
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class MetrLATrafficData:
    """METR-LA connected S-sensor highway corridor episodes + physical directed road graph."""

    sensor_ids: tuple[str, ...]
    adj_matrix: np.ndarray  # (S, S) physical directed road adjacency weights from adj_mx.pkl
    adj_binary: np.ndarray  # (S, S) off-diagonal binary physical road connectivity
    free_flow_speeds: np.ndarray  # (S,) 95th-percentile free-flow mph estimated strictly on train
    train_episodes: np.ndarray  # (E_tr, 24, S) congestion trajectories (12 hist + 12 fut)
    train_masks: np.ndarray  # (E_tr, 24, S) valid non-zero sensor mask
    val_episodes: np.ndarray  # (E_va, 24, S)
    val_masks: np.ndarray  # (E_va, 24, S)
    test_episodes: np.ndarray  # (E_te, 24, S)
    test_masks: np.ndarray  # (E_te, 24, S)
    test_shock_flags: np.ndarray  # (E_te,) localized congestion-shockwave episodes
    total_raw_observations: int
    file_sha256: dict[str, str]


def load_metr_la_traffic(
    raw_dir: Path,
    n_sensors: int = 64,
    max_train_eps: int = 180,
    max_val_eps: int = 60,
    max_test_eps: int = 160,
) -> MetrLATrafficData:
    """Load METR-LA highway corridor episodes and physical road graph strictly on official split."""
    adj_path = raw_dir / "adj_mx.pkl"
    tr_path = raw_dir / "train.parquet"
    va_path = raw_dir / "val.parquet"
    te_path = raw_dir / "test.parquet"
    for p in (adj_path, tr_path, va_path, te_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    hashes = {p.name: _sha256_file(p) for p in (adj_path, tr_path, va_path, te_path)}
    cache_npz = raw_dir / f"metr_la_s{n_sensors}_tr{max_train_eps}_te{max_test_eps}_cache.npz"

    if cache_npz.exists():
        z = np.load(cache_npz, allow_pickle=False)
        hashes[cache_npz.name] = _sha256_file(cache_npz)
        return MetrLATrafficData(
            sensor_ids=tuple(str(x) for x in z["sensor_ids"]),
            adj_matrix=np.asarray(z["adj_matrix"], dtype=np.float64),
            adj_binary=np.asarray(z["adj_binary"], dtype=bool),
            free_flow_speeds=np.asarray(z["free_flow_speeds"], dtype=np.float64),
            train_episodes=np.asarray(z["train_episodes"], dtype=np.float64),
            train_masks=np.asarray(z["train_masks"], dtype=np.float64),
            val_episodes=np.asarray(z["val_episodes"], dtype=np.float64),
            val_masks=np.asarray(z["val_masks"], dtype=np.float64),
            test_episodes=np.asarray(z["test_episodes"], dtype=np.float64),
            test_masks=np.asarray(z["test_masks"], dtype=np.float64),
            test_shock_flags=np.asarray(z["test_shock_flags"], dtype=bool),
            total_raw_observations=int(z["total_raw_observations"]),
            file_sha256=hashes,
        )

    with adj_path.open("rb") as f:
        sensor_ids_all, _, adj_full = pickle.load(f, encoding="latin1")
    adj_full = np.asarray(adj_full, dtype=np.float64)
    n_all = len(sensor_ids_all)
    off_adj = adj_full.copy()
    np.fill_diagonal(off_adj, 0.0)

    # Greedily grow a densely connected S-sensor highway corridor from the highest-degree hub
    deg = np.sum((off_adj > 0.0) | (off_adj.T > 0.0), axis=1)
    selected = [int(np.argmax(deg))]
    sel_set = set(selected)
    while len(selected) < n_sensors:
        # Pick unselected sensor with max total road-graph affinity to currently selected corridor
        scores = np.sum(off_adj[:, selected] + off_adj[selected, :].T, axis=1)
        for s_i in sel_set:
            scores[s_i] = -1.0
        nxt = int(np.argmax(scores))
        selected.append(nxt)
        sel_set.add(nxt)
    selected = sorted(selected)
    sel_arr = np.asarray(selected, dtype=np.int64)
    sub_ids = tuple(str(sensor_ids_all[i]) for i in selected)
    adj_sub = off_adj[np.ix_(sel_arr, sel_arr)]
    adj_bin = adj_sub > 0.0

    hist_cols = [f"x_t-{k}_d0" for k in range(11, 0, -1)] + ["x_t+0_d0"]
    fut_cols = [f"y_t+{k}_d0" for k in range(1, 13)]
    all_val_cols = hist_cols + fut_cols
    read_cols = ["node_id", "t0_timestamp", *all_val_cols]

    def _extract_split(
        pq_path: Path, stride_ts: int, max_eps: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        df = pd.read_parquet(pq_path, columns=read_cols)
        n_obs = len(df)
        df = df[df["node_id"].isin(sel_set)]
        ts_unique = sorted(df["t0_timestamp"].unique())
        # Non-overlapping 2-hour windows (stride >= 24 five-minute intervals)
        pick_ts = ts_unique[::stride_ts]
        df = df[df["t0_timestamp"].isin(set(pick_ts))]
        ts_to_i = {t: i for i, t in enumerate(pick_ts)}
        node_to_j = {nid: j for j, nid in enumerate(selected)}
        n_w = len(pick_ts)
        arr = np.zeros((n_w, 24, n_sensors), dtype=np.float64)
        vals = df[all_val_cols].to_numpy(dtype=np.float64)
        w_idx = df["t0_timestamp"].map(ts_to_i).to_numpy(dtype=np.int64)
        n_idx = df["node_id"].map(node_to_j).to_numpy(dtype=np.int64)
        for k_row in range(len(df)):
            arr[w_idx[k_row], :, n_idx[k_row]] = vals[k_row]
        masks = (arr > 1.0).astype(np.float64)
        # Keep windows with >= 85% valid sensor readings and genuine traffic activity
        valid_frac = np.mean(masks, axis=(1, 2))
        keep = np.where(valid_frac >= 0.85)[0]
        if len(keep) > max_eps:
            # Uniformly subsample across time to preserve full multi-week coverage
            lin = np.linspace(0, len(keep) - 1, max_eps, dtype=int)
            keep = keep[lin]
        return arr[keep], masks[keep], n_obs

    tr_spd, tr_msk, n_tr_obs = _extract_split(tr_path, stride_ts=24, max_eps=max_train_eps)
    va_spd, va_msk, n_va_obs = _extract_split(va_path, stride_ts=24, max_eps=max_val_eps)
    te_spd, te_msk, n_te_obs = _extract_split(te_path, stride_ts=24, max_eps=max_test_eps)

    # Free-flow speed per sensor estimated strictly on training split (zero test leakage!)
    free_flow = np.zeros(n_sensors, dtype=np.float64)
    for j in range(n_sensors):
        v = tr_spd[:, :, j][tr_msk[:, :, j] > 0.5]
        free_flow[j] = float(np.percentile(v, 95)) if len(v) > 0 else 65.0

    # Convert speed (mph) to normalized congestion depth: c = max(0, free_flow - speed) / 20.0
    def _to_congestion(spd: np.ndarray, msk: np.ndarray) -> np.ndarray:
        cong = np.maximum(0.0, free_flow[None, None, :] - spd) / 20.0
        return cong * msk

    tr_ep = _to_congestion(tr_spd, tr_msk)
    va_ep = _to_congestion(va_spd, va_msk)
    te_ep = _to_congestion(te_spd, te_msk)

    # Flag localized congestion-shockwave episodes on test: at least one sensor has a sharp
    # congestion jump (>= 15 mph drop = 0.75 units) while median corridor congestion is moderate
    te_jump = np.max(np.max(te_ep[:, 12:, :], axis=1) - te_ep[:, 11, :], axis=1)
    te_max_c = np.max(te_ep[:, 12:, :], axis=(1, 2))
    te_med_c = np.median(te_ep[:, 12:, :], axis=(1, 2))
    shock_flags = (te_jump >= 0.75) & (te_max_c >= 1.2) & (te_med_c <= 0.6)
    if int(np.sum(shock_flags)) < 15:
        # Fallback: top quartile of spatial heterogeneity (max - median) during test horizon
        het = te_max_c - te_med_c
        shock_flags = het >= np.percentile(het, 75)

    tot_obs = int(n_tr_obs + n_va_obs + n_te_obs)
    np.savez(
        cache_npz,
        sensor_ids=np.asarray(sub_ids),
        adj_matrix=adj_sub,
        adj_binary=adj_bin,
        free_flow_speeds=free_flow,
        train_episodes=tr_ep,
        train_masks=tr_msk,
        val_episodes=va_ep,
        val_masks=va_msk,
        test_episodes=te_ep,
        test_masks=te_msk,
        test_shock_flags=shock_flags,
        total_raw_observations=np.int64(tot_obs),
    )
    hashes[cache_npz.name] = _sha256_file(cache_npz)
    return MetrLATrafficData(
        sensor_ids=sub_ids,
        adj_matrix=adj_sub,
        adj_binary=adj_bin,
        free_flow_speeds=free_flow,
        train_episodes=tr_ep,
        train_masks=tr_msk,
        val_episodes=va_ep,
        val_masks=va_msk,
        test_episodes=te_ep,
        test_masks=te_msk,
        test_shock_flags=shock_flags,
        total_raw_observations=tot_obs,
        file_sha256=hashes,
    )


# =============================================================================
# 3. Binance Vision 16-Asset 5-Minute Order-Flow & Volatility Contagion (May-Aug 2024)
# =============================================================================

BINANCE_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "NEARUSDT",
)


@dataclass(frozen=True, kw_only=True)
class FinancialContagionData:
    """Binance 16-asset 5-min causal order-flow -> volatility cascade episodes (May-Aug 2024)."""

    symbols: tuple[str, ...]
    drivers: np.ndarray  # (E, 48, 16) causal bar-(t) signed/unsigned order-flow shock drive
    targets: np.ndarray  # (E, 48, 16) bar-(t+1) realized log-volatility deviation from episode start
    month_ids: np.ndarray  # (E,) 5=May, 6=June, 7=July, 8=Aug 2024
    idiosyncratic_flags: np.ndarray  # (E,) episodes where single-asset shock breaks macro corr
    total_5min_bars: int
    file_sha256: dict[str, str]


def load_binance_market_contagion(
    raw_dir: Path, episode_bars: int = 48
) -> FinancialContagionData:
    """Load 16-asset 5-min Binance Vision ZIPs into strict causal 4-hour cascade episodes."""
    bin_dir = raw_dir / "binance_5m"
    months = ("2024-05", "2024-06", "2024-07", "2024-08")
    hashes: dict[str, str] = {}
    for sym in BINANCE_SYMBOLS:
        for mo in months:
            zp = bin_dir / f"{sym}-5m-{mo}.zip"
            if not zp.exists():
                raise FileNotFoundError(f"Missing {zp}")
            hashes[zp.name] = _sha256_file(zp)

    cache_npz = raw_dir / f"binance_16asset_{episode_bars}bar_cache.npz"
    if cache_npz.exists():
        z = np.load(cache_npz, allow_pickle=False)
        hashes[cache_npz.name] = _sha256_file(cache_npz)
        return FinancialContagionData(
            symbols=BINANCE_SYMBOLS,
            drivers=np.asarray(z["drivers"], dtype=np.float64),
            targets=np.asarray(z["targets"], dtype=np.float64),
            month_ids=np.asarray(z["month_ids"], dtype=np.int64),
            idiosyncratic_flags=np.asarray(z["idiosyncratic_flags"], dtype=bool),
            total_5min_bars=int(z["total_5min_bars"]),
            file_sha256=hashes,
        )

    cols_b = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "close_time",
        "qvol",
        "n_trades",
        "tb_vol",
        "tb_qvol",
        "ignore",
    ]
    rv_list, flow_list = [], []
    month_vec = None
    for sym in BINANCE_SYMBOLS:
        parts = []
        mo_parts = []
        for mo_i, mo in enumerate(months, start=5):
            zp = bin_dir / f"{sym}-5m-{mo}.zip"
            with zipfile.ZipFile(zp) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    d = pd.read_csv(f, header=None, names=cols_b)
                    parts.append(d)
                    mo_parts.append(np.full(len(d), mo_i, dtype=np.int64))
        df = pd.concat(parts, ignore_index=True)
        if month_vec is None:
            month_vec = np.concatenate(mo_parts)
        hi = df["high"].to_numpy(dtype=np.float64)
        lo = df["low"].to_numpy(dtype=np.float64)
        op = df["open"].to_numpy(dtype=np.float64)
        cl = df["close"].to_numpy(dtype=np.float64)
        qv = df["qvol"].to_numpy(dtype=np.float64)
        tb = df["tb_qvol"].to_numpy(dtype=np.float64)

        # Target: intrabar log range volatility in % (log1p-compressed for heavy-tail stability)
        rv = np.log1p(np.log(np.maximum(hi / np.maximum(lo, 1e-12), 1.0)) * 100.0)
        # Causal order-flow shock drive: volume-surge magnitude weighted by sell-leverage asymmetry
        med_q = max(1.0, float(np.median(qv[: 288 * 60])))  # May-June median only (no Aug leakage)
        vol_surge = np.log1p(qv / med_q)
        sell_frac = np.clip((qv - tb) / np.maximum(qv, 1e-8), 0.0, 1.0)
        neg_ret = np.maximum(0.0, -np.log(cl / np.maximum(op, 1e-12)) * 100.0)
        # Signed-asymmetric liquidity shock drive: sell-side bursts load harder than buy bursts
        flow_drv = vol_surge * (0.5 + sell_frac) + 0.5 * np.log1p(neg_ret)
        rv_list.append(rv)
        flow_list.append(flow_drv)

    n_bars = min(len(x) for x in rv_list)
    rv_mat = np.column_stack([x[:n_bars] for x in rv_list])
    fl_mat = np.column_stack([x[:n_bars] for x in flow_list])
    month_vec = month_vec[:n_bars]

    # Strict causal alignment: driver at step t predicts volatility at step t+1 (no same-bar leakage!)
    fl_causal = fl_mat[:-1]
    rv_next = rv_mat[1:]
    mo_causal = month_vec[1:]

    # Partition into non-overlapping 4-hour (48-bar) episodes
    n_ep = len(fl_causal) // episode_bars
    ep_drv, ep_tgt, ep_mo, ep_idio = [], [], [], []
    # Compute asset-wise training baseline stats strictly on May-June (months 5-6)
    tr_mask_bars = mo_causal <= 6
    fl_mean = np.mean(fl_causal[tr_mask_bars], axis=0, keepdims=True)
    fl_std = np.maximum(1e-4, np.std(fl_causal[tr_mask_bars], axis=0, keepdims=True))
    fl_norm = (fl_causal - fl_mean) / fl_std

    for e_i in range(n_ep):
        s0 = e_i * episode_bars
        s1 = s0 + episode_bars
        d_w = fl_norm[s0:s1]
        y_w = rv_next[s0:s1]
        if not (np.all(np.isfinite(d_w)) and np.all(np.isfinite(y_w))):
            continue
        # Idiosyncratic shock score: max single-asset peak drive minus median asset drive
        peak_per_asset = np.max(d_w, axis=0)
        idio_ratio = float(np.max(peak_per_asset) - np.median(peak_per_asset))
        ep_drv.append(d_w)
        ep_tgt.append(y_w - y_w[0:1, :])
        ep_mo.append(int(mo_causal[s0]))
        ep_idio.append(idio_ratio)

    drv_arr = np.stack(ep_drv, axis=0)
    tgt_arr = np.stack(ep_tgt, axis=0)
    mo_arr = np.asarray(ep_mo, dtype=np.int64)
    idio_scores = np.asarray(ep_idio, dtype=np.float64)
    # Top 30% idiosyncratic single-asset shock episodes (threshold from May-July only)
    idio_thresh = float(np.percentile(idio_scores[mo_arr <= 7], 70))
    idio_flags = idio_scores >= idio_thresh

    np.savez(
        cache_npz,
        drivers=drv_arr,
        targets=tgt_arr,
        month_ids=mo_arr,
        idiosyncratic_flags=idio_flags,
        total_5min_bars=np.int64(n_bars * len(BINANCE_SYMBOLS)),
    )
    hashes[cache_npz.name] = _sha256_file(cache_npz)
    return FinancialContagionData(
        symbols=BINANCE_SYMBOLS,
        drivers=drv_arr,
        targets=tgt_arr,
        month_ids=mo_arr,
        idiosyncratic_flags=idio_flags,
        total_5min_bars=int(n_bars * len(BINANCE_SYMBOLS)),
        file_sha256=hashes,
    )


# =============================================================================
# 4. SNAP Real Temporal Reddit Hyperlink Community Network (571,927 links, 2014-2017)
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class RedditTemporalData:
    """SNAP Reddit temporal directed subreddit cascade episodes + held-out hyperlink graph."""

    subreddits: tuple[str, ...]
    train_episodes: np.ndarray  # (E_tr, 24, S) continuous smoothed activity trajectories (days)
    val_episodes: np.ndarray  # (E_va, 24, S)
    test_episodes: np.ndarray  # (E_te, 24, S)
    test_adj_binary: np.ndarray  # (S, S) held-out 2016-2017 directed subreddit hyperlink ground truth
    train_adj_counts: np.ndarray  # (S, S)
    total_links: int
    file_sha256: dict[str, str]


def load_reddit_temporal_network(
    raw_dir: Path, n_subreddits: int = 24, episode_days: int = 24
) -> RedditTemporalData:
    """Load SNAP soc-redditHyperlinks-title.tsv into chronological continuous activity cascades."""
    tsv_path = raw_dir / "soc-redditHyperlinks-title.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing {tsv_path}")
    hashes = {tsv_path.name: _sha256_file(tsv_path)}
    cache_npz = raw_dir / f"reddit_s{n_subreddits}_{episode_days}d_cache.npz"

    if cache_npz.exists():
        z = np.load(cache_npz, allow_pickle=False)
        hashes[cache_npz.name] = _sha256_file(cache_npz)
        return RedditTemporalData(
            subreddits=tuple(str(x) for x in z["subreddits"]),
            train_episodes=np.asarray(z["train_episodes"], dtype=np.float64),
            val_episodes=np.asarray(z["val_episodes"], dtype=np.float64),
            test_episodes=np.asarray(z["test_episodes"], dtype=np.float64),
            test_adj_binary=np.asarray(z["test_adj_binary"], dtype=bool),
            train_adj_counts=np.asarray(z["train_adj_counts"], dtype=np.int64),
            total_links=int(z["total_links"]),
            file_sha256=hashes,
        )

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"],
    )
    total_links = len(df)
    df["ts"] = pd.to_datetime(df["TIMESTAMP"])

    # Select top S subreddits strictly from training period (2014-01-01 to 2015-12-31) -> zero leakage!
    tr_df = df[(df["ts"] >= "2014-01-01") & (df["ts"] < "2016-01-01")]
    top_subs = tuple(
        str(x)
        for x in pd.concat([tr_df["SOURCE_SUBREDDIT"], tr_df["TARGET_SUBREDDIT"]])
        .value_counts()
        .head(n_subreddits)
        .index
    )
    sub_set = set(top_subs)

    # Compute daily total cross-community mention intensity (incoming + outgoing links) per subreddit
    # across all 571k links touching each selected community (dense, high SNR continuous activity!)
    df_in = df[df["TARGET_SUBREDDIT"].isin(sub_set)].copy()
    df_in["day"] = df_in["ts"].dt.floor("D")
    df_out = df[df["SOURCE_SUBREDDIT"].isin(sub_set)].copy()
    df_out["day"] = df_out["ts"].dt.floor("D")

    date_index = pd.date_range("2014-01-01", "2017-04-30", freq="D")
    in_piv = (
        df_in.pivot_table(
            index="day", columns="TARGET_SUBREDDIT", values="LINK_SENTIMENT", aggfunc="count", fill_value=0
        )
        .reindex(index=date_index, columns=top_subs, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    out_piv = (
        df_out.pivot_table(
            index="day", columns="SOURCE_SUBREDDIT", values="LINK_SENTIMENT", aggfunc="count", fill_value=0
        )
        .reindex(index=date_index, columns=top_subs, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    # Causal 3-day exponential moving average of log1p(total cross-community traffic)
    raw_log = np.log1p(in_piv + out_piv)
    ema = np.zeros_like(raw_log)
    ema[0] = raw_log[0]
    alpha = 0.45
    for t in range(1, len(raw_log)):
        ema[t] = alpha * raw_log[t] + (1.0 - alpha) * ema[t - 1]

    # Chronological split: Train 2014-2015, Val Jan-Jun 2016, Test Jul 2016 - Apr 2017
    d_arr = date_index.to_numpy()
    tr_idx = np.where(d_arr < np.datetime64("2016-01-01"))[0]
    va_idx = np.where((d_arr >= np.datetime64("2016-01-01")) & (d_arr < np.datetime64("2016-07-01")))[0]
    te_idx = np.where(d_arr >= np.datetime64("2016-07-01"))[0]

    # Standardize strictly using training mean/std
    mu = np.mean(ema[tr_idx], axis=0, keepdims=True)
    sd = np.maximum(0.1, np.std(ema[tr_idx], axis=0, keepdims=True))
    z_all = (ema - mu) / sd

    def _slice_eps(idxs: np.ndarray, stride: int) -> np.ndarray:
        eps = []
        for s in range(0, len(idxs) - episode_days + 1, stride):
            win = z_all[idxs[s : s + episode_days]]
            eps.append(win)
        return np.stack(eps, axis=0)

    tr_eps = _slice_eps(tr_idx, stride=6)
    va_eps = _slice_eps(va_idx, stride=6)
    te_eps = _slice_eps(te_idx, stride=6)

    # True held-out directed subreddit-to-subreddit interaction graph on 2016-07 -> 2017-04
    sub_internal_tr = df[
        (df["ts"] < "2016-01-01")
        & df["SOURCE_SUBREDDIT"].isin(sub_set)
        & df["TARGET_SUBREDDIT"].isin(sub_set)
        & (df["SOURCE_SUBREDDIT"] != df["TARGET_SUBREDDIT"])
    ]
    sub_internal_te = df[
        (df["ts"] >= "2016-07-01")
        & df["SOURCE_SUBREDDIT"].isin(sub_set)
        & df["TARGET_SUBREDDIT"].isin(sub_set)
        & (df["SOURCE_SUBREDDIT"] != df["TARGET_SUBREDDIT"])
    ]
    tr_adj = (
        sub_internal_tr.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=top_subs, columns=top_subs, fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    te_adj = (
        sub_internal_te.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=top_subs, columns=top_subs, fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    # Active persistent directed cross-community links in held-out period (>= 5 links)
    te_adj_bin = te_adj >= 5
    np.fill_diagonal(te_adj_bin, False)

    np.savez(
        cache_npz,
        subreddits=np.asarray(top_subs),
        train_episodes=tr_eps,
        val_episodes=va_eps,
        test_episodes=te_eps,
        test_adj_binary=te_adj_bin,
        train_adj_counts=tr_adj,
        total_links=np.int64(total_links),
    )
    hashes[cache_npz.name] = _sha256_file(cache_npz)
    return RedditTemporalData(
        subreddits=top_subs,
        train_episodes=tr_eps,
        val_episodes=va_eps,
        test_episodes=te_eps,
        test_adj_binary=te_adj_bin,
        train_adj_counts=tr_adj,
        total_links=total_links,
        file_sha256=hashes,
    )
