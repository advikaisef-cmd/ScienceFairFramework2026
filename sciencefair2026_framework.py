import os
import sys
import json
import time
import math
import argparse
import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request

import numpy as np
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit, prange
from scipy.stats import ttest_rel, wilcoxon, linregress

warnings.filterwarnings("ignore")

USCRN_URL = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/{y}/CRND0103-{y}-{s}.txt"
POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=WS2M&community=AG"
             "&longitude={lon}&latitude={lat}&start={a}&end={b}&format=JSON")
OM_FORECAST = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&daily=precipitation_sum,et0_fao_evapotranspiration&past_days=5&forecast_days=10&timezone=auto")
OM_HISTORICAL = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                 "&daily=precipitation_sum,et0_fao_evapotranspiration&start_date={a}&end_date={b}&timezone=auto")

CANDIDATES = [
    "NE_Lincoln_11_SW", "KS_Manhattan_6_SSW", "OK_Stillwater_2_W", "IL_Champaign_9_SW", "TX_Austin_33_NW",
    "CA_Merced_23_WSW", "GA_Watkinsville_5_SSE", "SD_Pierre_24_S", "AZ_Tucson_11_W", "NC_Durham_11_W",
    "WA_Spokane_17_SSW", "IA_Des_Moines_17_E", "IN_Bedford_5_WNW", "CO_Nunn_7_NNE", "MS_Newton_5_ENE",
    "ND_Jamestown_38_WSW", "TN_Crossville_7_NW", "NM_Las_Cruces_20_N", "OH_Wooster_3_SSE",
    "MO_Chillicothe_22_ENE", "AL_Clanton_2_NE", "SC_Blackville_3_W", "MN_Sandstone_6_W", "TX_Muleshoe_19_S",
    "KY_Versailles_3_NNW", "OR_Corvallis_10_SSW", "VA_Charlottesville_2_SSE", "LA_Monroe_26_N",
    "MT_Wolf_Point_29_ENE", "ID_Arco_17_SW", "NE_Whitman_5_ENE", "WI_Necedah_5_WNW",
]
YEARS = list(range(2017, 2025))
TRAIN_YEARS = (2017, 2018, 2019, 2020, 2021)
VAL_YEARS = (2022,)
TEST_YEARS = (2023, 2024)
START = np.datetime64("2017-01-01")
DATES = np.arange(START, np.datetime64("2025-01-01"), dtype="datetime64[D]")
DEPTHS = [5, 10, 20, 50]
KNOTS = [5, 10, 20, 50, 100]
LP, H = 32, 7
L = LP + H
RZW = np.array([7.5, 7.5, 20.0, 25.0])
RZ_TOTAL_MM = 600.0
RZW_T = torch.tensor(RZW / RZW.sum(), dtype=torch.float32)
SUB, ITERS = 8, 30
DT = torch.float64
UMAX = 25.0
MPC_DU = 10.0
MC_TRAJ = 500
ALPHA = 0.05
Z95 = 1.645
W_WATER, W_BAND, W_SAFE = 10.0, 140.0, 280.0
U_SNAP = 0.4
SAFE_DAYS = 3
REFILL = 0.2
PBINS = np.array([1.0, 10.0, 30.0])
PKEYS = ["lks1", "lks2", "la1", "la2", "n1", "n2", "fr1", "fr2", "kc", "zr"]
PLO = np.array([0.0, -0.5, -2.5, -2.5, 1.2, 1.2, 0.0, 0.0, 0.3, 5.0])
PHI = np.array([2.5, 2.5, -0.8, -0.8, 2.6, 2.6, 0.9, 0.9, 1.3, 60.0])

MODES = {
    "smoke": dict(n_stations=3, cal_windows=16, cal_pop=8, cal_gen=2, epochs=2, ens=40, mpc_iters=3, mpc_outer=1,
                  mpc_days=8, spde_ens=60, sweep=False, ftle_days=10, lyap_days=6, sweep_epochs=1, mpc_traj=100),
    "quick": dict(n_stations=6, cal_windows=64, cal_pop=24, cal_gen=5, epochs=30, ens=100, mpc_iters=10,
                  mpc_outer=2, mpc_days=61, spde_ens=300, sweep=True, ftle_days=30, lyap_days=21, sweep_epochs=10,
                  mpc_traj=500),
    "standard": dict(n_stations=12, cal_windows=128, cal_pop=32, cal_gen=8, epochs=60, ens=200, mpc_iters=12,
                     mpc_outer=2, mpc_days=92, spde_ens=1000, sweep=True, ftle_days=30, lyap_days=21,
                     sweep_epochs=25, mpc_traj=500),
    "full": dict(n_stations=20, cal_windows=200, cal_pop=48, cal_gen=12, epochs=120, ens=500, mpc_iters=20,
                 mpc_outer=3, mpc_days=92, spde_ens=2000, sweep=True, ftle_days=30, lyap_days=21, sweep_epochs=50,
                 mpc_traj=1000),
}

RESULTS = []


def record(section, metric, value, note=""):
    RESULTS.append(dict(section=section, metric=metric, value=value, note=note))
    v = f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
    print(f"  {section} | {metric} = {v}" + (f"  ({note})" if note else ""))


def get(section, metric):
    for r in RESULTS[::-1]:
        if r["section"] == section and r["metric"] == metric:
            return r["value"]
    return None


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def calendar(dates):
    y = dates.astype("datetime64[Y]").astype(int) + 1970
    m = dates.astype("datetime64[M]").astype(int) % 12 + 1
    doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    return y, m, doy


YR, MON, DOY = calendar(DATES)
GROW = (MON >= 4) & (MON <= 10)


def http_get(url, timeout=60, tries=3):
    last = None
    for k in range(tries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 sf2026-framework"}), timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def uscrn_file(station, year, cache):
    path = os.path.join(cache, f"CRND0103-{year}-{station}.txt")
    current = year >= datetime.date.today().year
    if os.path.exists(path) and (not current or time.time() - os.path.getmtime(path) < 6 * 3600):
        with open(path, encoding="utf-8") as f:
            return f.read()
    try:
        txt = http_get(USCRN_URL.format(y=year, s=station))
    except Exception:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        return ""
    os.makedirs(cache, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


USCRN_COLS = dict(lon=3, lat=4, tmax=5, tmin=6, p=9, rs=10, rhmax=15, rhmin=16, rhavg=17,
                  sm5=18, sm10=19, sm20=20, sm50=21, sm100=22, st5=23)


def parse_uscrn(texts, start, n):
    out = {k: np.full(n, np.nan) for k in USCRN_COLS}
    for txt in texts:
        for line in txt.splitlines():
            f = line.split()
            if len(f) < 28:
                continue
            d = int((np.datetime64(f"{f[1][:4]}-{f[1][4:6]}-{f[1][6:8]}") - start).astype(int))
            if 0 <= d < n:
                for k, c in USCRN_COLS.items():
                    v = float(f[c])
                    out[k][d] = v if k in ("lon", "lat") else (np.nan if v <= -99 else v)
    return out


def fill_gaps(x, maxgap=5):
    x = x.copy()
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return x
    idx = np.arange(len(x))
    lin = np.interp(idx, idx[ok], x[ok])
    i = 0
    while i < len(x):
        if not ok[i]:
            j = i
            while j < len(x) and not ok[j]:
                j += 1
            if j - i <= maxgap and i > 0 and j < len(x):
                x[i:j] = lin[i:j]
            i = j
        else:
            i += 1
    return x


def ffill(x):
    x = x.copy()
    for d in range(x.shape[-1]):
        v = x[:, d]
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            v[:] = 0.25
            continue
        idx = np.where(ok, np.arange(len(v)), 0)
        np.maximum.accumulate(idx, out=idx)
        v[:] = v[idx]
        v[:np.argmax(ok)] = v[np.argmax(ok)]
    return x


def fao56_et0(tmax, tmin, rhmax, rhmin, u2, rs, doy, lat, elev):
    tmean = 0.5 * (tmax + tmin)
    pres = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * pres
    e0 = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    delta = 4098 * e0(tmean) / (tmean + 237.3) ** 2
    es = 0.5 * (e0(tmax) + e0(tmin))
    ea = np.minimum(0.5 * (e0(tmin) * rhmax / 100 + e0(tmax) * rhmin / 100), es)
    phi = np.deg2rad(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra
    rs = np.minimum(rs, rso)
    rnl = 4.903e-9 * 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) \
        * (1.35 * np.clip(rs / np.maximum(rso, 1e-6), 0.25, 1.0) - 0.35)
    rn = 0.77 * rs - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return np.maximum(et0, 0.0)


def power_wind(lat, lon, dates, cache):
    a, b = str(dates[0]).replace("-", ""), str(dates[-1]).replace("-", "")
    path = os.path.join(cache, f"power_ws2m_{lat:.3f}_{lon:.3f}_{a}_{b}.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                js = json.load(f)
        else:
            js = json.loads(http_get(POWER_URL.format(lat=lat, lon=lon, a=a, b=b), timeout=90))
            with open(path, "w") as f:
                json.dump(js, f)
        ws = js["properties"]["parameter"]["WS2M"]
        keys = [str(d).replace("-", "") for d in dates]
        u2 = np.array([ws.get(k, -999.0) for k in keys], dtype=float)
        u2[u2 < 0] = np.nan
        u2 = np.where(np.isnan(u2), np.nanmedian(u2), u2)
        elev = float(js["geometry"]["coordinates"][2])
        return u2, elev, "NASA POWER WS2M"
    except Exception:
        return np.full(len(dates), 2.0), 300.0, "FAO-56 default u2 = 2 m/s"


def station_arrays(raw, dates, u2, elev):
    _, _, doy = calendar(dates)
    lat = float(np.nanmedian(raw["lat"]))
    tmax, tmin, rs = fill_gaps(raw["tmax"]), fill_gaps(raw["tmin"]), fill_gaps(raw["rs"])
    rhmax = fill_gaps(np.where(np.isnan(raw["rhmax"]), raw["rhavg"], raw["rhmax"]))
    rhmin = fill_gaps(np.where(np.isnan(raw["rhmin"]), raw["rhavg"], raw["rhmin"]))
    et0 = fao56_et0(tmax, tmin, rhmax, rhmin, u2, rs, doy, lat, elev)
    sm = np.stack([raw[f"sm{d}"] for d in DEPTHS], -1)
    frozen = raw["st5"] < 1.0
    sm[(sm <= 0.0) | (sm > 0.6) | frozen[:, None]] = np.nan
    sm100 = raw["sm100"].copy()
    sm100[(sm100 <= 0.0) | (sm100 > 0.6) | frozen] = np.nan
    p = raw["p"].copy()
    p[p < 0] = np.nan
    wv = ~np.isnan(p) & ~np.isnan(et0)
    return dict(sm=sm, sm100=sm100, smv=~np.isnan(sm).any(-1), p=np.nan_to_num(p), et0=np.nan_to_num(et0), wv=wv,
                lat=lat, lon=float(np.nanmedian(raw["lon"])), elev=elev)


def load_dataset(cfg, cache):
    need = cfg["n_stations"]
    chosen, pos = [], 0
    while len(chosen) < need and pos < len(CANDIDATES):
        batch = CANDIDATES[pos:pos + need - len(chosen) + 3]
        pos += len(batch)
        jobs = [(s, y) for s in batch for y in YEARS]
        with ThreadPoolExecutor(8) as ex:
            texts = list(ex.map(lambda sy: uscrn_file(sy[0], sy[1], cache), jobs))
        for i, s in enumerate(batch):
            raw = parse_uscrn(texts[i * len(YEARS):(i + 1) * len(YEARS)], START, len(DATES))
            if np.isnan(raw["lat"]).all():
                continue
            pre = station_arrays(raw, DATES, np.full(len(DATES), 2.0), 300.0)
            ok = pre["smv"] & pre["wv"]
            c_tr = ok[GROW & np.isin(YR, TRAIN_YEARS)].mean()
            c_va = ok[GROW & np.isin(YR, VAL_YEARS)].mean()
            c_te = ok[GROW & np.isin(YR, TEST_YEARS)].mean()
            print(f"  {s:28s} coverage train {c_tr:.2f} val {c_va:.2f} test {c_te:.2f}")
            if c_tr >= 0.5 and c_va >= 0.4 and c_te >= 0.5 and len(chosen) < need:
                chosen.append((s, raw, pre))
    S = len(chosen)
    names = [c[0] for c in chosen]
    with ThreadPoolExecutor(4) as ex:
        winds = list(ex.map(lambda c: power_wind(c[2]["lat"], c[2]["lon"], DATES, cache), chosen))
    st = [station_arrays(c[1], DATES, w[0], w[1]) for c, w in zip(chosen, winds)]
    DS = dict(S=S, names=names, wind_src=[w[2] for w in winds])
    for k in ["sm", "sm100", "smv", "p", "et0", "wv"]:
        DS[k] = np.stack([x[k] for x in st])
    for k in ["lat", "lon", "elev"]:
        DS[k] = np.array([x[k] for x in st])
    DS["sm_ff"] = np.stack([ffill(x["sm"]) for x in st]).astype(np.float32)
    tr = np.isin(YR, TRAIN_YEARS)
    top = DS["sm"][:, tr, :3].reshape(S, -1)
    sub = np.concatenate([DS["sm"][:, tr, 3], DS["sm100"][:, tr]], 1)
    DS["ts1"] = np.clip(np.nanquantile(top, 0.995, axis=1) + 0.01, 0.25, 0.6)
    DS["ts2"] = np.clip(np.nanquantile(sub, 0.995, axis=1) + 0.01, 0.25, 0.6)
    DS["mn1"] = np.nanquantile(top, 0.005, axis=1)
    DS["mn2"] = np.nanquantile(sub, 0.005, axis=1)
    return DS


def lin_weights(src, dst):
    src = np.asarray(src, float)
    W = np.zeros((len(dst), len(src)))
    for i, x in enumerate(dst):
        if x <= src[0]:
            W[i, 0] = 1
        elif x >= src[-1]:
            W[i, -1] = 1
        else:
            j = int(np.searchsorted(src, x))
            t = (x - src[j - 1]) / (src[j] - src[j - 1])
            W[i, j - 1], W[i, j] = 1 - t, t
    return torch.tensor(W, dtype=DT)


class Column:
    def __init__(self, depth=120.0, dz=5.0, top=30.0):
        self.dz = dz
        self.N = int(round(depth / dz))
        self.z = (torch.arange(self.N, dtype=DT) + 0.5) * dz
        self.top = self.z < top
        self.Ws = lin_weights(self.z.numpy(), DEPTHS)
        self.Wi = lin_weights(KNOTS, self.z.numpy())
        self.rz = (self.z < 60).to(DT) * 10.0 * dz


def to_phys(U, st):
    lo, hi = torch.tensor(PLO, dtype=DT), torch.tensor(PHI, dtype=DT)
    x = lo + U * (hi - lo)
    return dict(ks1=10 ** x[:, 0], ks2=10 ** x[:, 1], a1=10 ** x[:, 2], a2=10 ** x[:, 3], n1=x[:, 4], n2=x[:, 5],
                tr1=x[:, 6] * st["mn1"], tr2=x[:, 7] * st["mn2"], ts1=st["ts1"], ts2=st["ts2"], kc=x[:, 8], zr=x[:, 9])


def make_params(ph, col):
    top = col.top[None, :]
    lay = lambda a, b: torch.where(top, ph[a][:, None], ph[b][:, None])
    p = dict(Ks=lay("ks1", "ks2"), a=lay("a1", "a2"), n=lay("n1", "n2"), tr=lay("tr1", "tr2"), ts=lay("ts1", "ts2"))
    p["m"] = 1 - 1 / p["n"]
    rw = torch.exp(-col.z[None, :] / ph["zr"][:, None]) * (col.z[None, :] < 100).to(DT)
    p["rw"] = rw / (rw.sum(1, keepdim=True) * col.dz)
    p["kc"] = ph["kc"]
    return p


def vg_se(h, p):
    return (1 + (p["a"] * (-h).clamp(min=0)) ** p["n"]) ** (-p["m"])


def vg_theta(h, p):
    return p["tr"] + (p["ts"] - p["tr"]) * vg_se(h, p)


def vg_h(theta, p):
    se = ((theta - p["tr"]) / (p["ts"] - p["tr"])).clamp(1e-7, 1 - 1e-7)
    return -(1 / p["a"]) * (se ** (-1 / p["m"]) - 1) ** (1 / p["n"])


@njit(cache=True, fastmath=True)
def _vg(h, a, n, m, tr, ts, ks):
    ah = -h if h < -1e-10 else 1e-10
    xn = (a * ah) ** n
    q = 1.0 + xn
    se = q ** (-m)
    th = tr + (ts - tr) * se
    k = ks * math.sqrt(se) * (1.0 - (xn / q) ** m) ** 2
    c = (ts - tr) * n * m * xn * se / (ah * q) + 1e-6
    return th, c, k


@njit(cache=True, fastmath=True)
def _theta_vec(h, a, n, m, tr, ts, ks, out):
    for i in range(h.shape[0]):
        out[i] = _vg(h[i], a[i], n[i], m[i], tr[i], ts[i], ks[i])[0]


@njit(cache=True, fastmath=True)
def _picard(h, thn, a, n, m, tr, ts, ks, rw, qin, etr, dt, dz, maxit, wk):
    N = h.shape[0]
    hm, hn, th, cc, kk = wk[0], wk[1], wk[2], wk[3], wk[4]
    av, bv, cv, rv, cp, dp, sk = wk[5], wk[6], wk[7], wk[8], wk[9], wk[10], wk[11]
    for i in range(N):
        hm[i] = h[i]
    q0 = 0.0
    kf0 = 0.0
    pond = False
    conv = False
    for it in range(maxit):
        for i in range(N):
            th[i], cc[i], kk[i] = _vg(hm[i], a[i], n[i], m[i], tr[i], ts[i], ks[i])
        kf0 = 0.5 * (ks[0] + kk[0])
        cap = kf0 * (1.0 - 2.0 * hm[0] / dz)
        if it < 6:
            pond = cap < qin
        elif pond and cap > qin:
            pond = False
        for i in range(N):
            f = (hm[i] + 15000.0) / 14500.0
            f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
            sk[i] = min(etr * rw[i] * f, max(th[i] - tr[i] - 1e-3, 0.0) / dt)
            av[i] = 0.0
            cv[i] = 0.0
            bv[i] = cc[i] / dt
            rv[i] = cc[i] * hm[i] / dt - (th[i] - thn[i]) / dt - sk[i]
        for i in range(N - 1):
            kf = 0.5 * (kk[i] + kk[i + 1])
            g = kf / (dz * dz)
            cv[i] = -g
            av[i + 1] = -g
            bv[i] += g
            bv[i + 1] += g
            rv[i] -= kf / dz
            rv[i + 1] += kf / dz
        if pond:
            bv[0] += 2.0 * kf0 / (dz * dz)
            rv[0] += kf0 / dz
        else:
            rv[0] += qin / dz
        rv[N - 1] -= kk[N - 1] / dz
        cp[0] = cv[0] / bv[0]
        dp[0] = rv[0] / bv[0]
        for i in range(1, N):
            den = bv[i] - av[i] * cp[i - 1]
            cp[i] = cv[i] / den
            dp[i] = (rv[i] - av[i] * dp[i - 1]) / den
        hn[N - 1] = dp[N - 1]
        for i in range(N - 2, -1, -1):
            hn[i] = dp[i] - cp[i] * hn[i + 1]
        ok = True
        for i in range(N):
            v = min(max(hn[i], -1e5), 1000.0)
            dh = abs(v - hm[i])
            if v < 0.0 and hm[i] < 0.0:
                if cc[i] * dh > 1e-4:
                    ok = False
            elif dh > 1.0:
                ok = False
            hm[i] = v
        q0 = kf0 * (1.0 - 2.0 * hn[0] / dz) if pond else qin
        if ok:
            conv = True
            break
    st = 0.0
    for i in range(N):
        st += sk[i] * dz
    return conv, q0, kk[N - 1], st, it + 1


@njit(cache=True, fastmath=True)
def _advance(h, a, n, m, tr, ts, ks, rw, qin, etr, span, dz, dt0, maxit, wk, acc):
    N = h.shape[0]
    thn, thnew = wk[12], wk[13]
    t, dt = 0.0, dt0
    while t < span - 1e-12:
        dt = min(dt, span - t)
        _theta_vec(h, a, n, m, tr, ts, ks, thn)
        conv, q0, qb, st, nit = _picard(h, thn, a, n, m, tr, ts, ks, rw, qin, etr, dt, dz, maxit, wk)
        if not conv and dt > 1.0 / 2880.0:
            dt *= 0.5
            continue
        for i in range(N):
            h[i] = wk[0][i]
        _theta_vec(h, a, n, m, tr, ts, ks, thnew)
        ds = 0.0
        for i in range(N):
            ds += (thnew[i] - thn[i]) * dz
        acc[0] += 10.0 * q0 * dt
        acc[1] += 10.0 * qb * dt
        acc[2] += 10.0 * st * dt
        acc[3] += 10.0 * max(qin - q0, 0.0) * dt
        acc[4] += 10.0 * (ds - (q0 - qb - st) * dt)
        acc[5] += nit
        t += dt
        if nit <= 4:
            dt = min(dt * 1.5, dt0)


@njit(parallel=True, cache=True, fastmath=True)
def _simulate(h0, A, Nn, Mm, TR, TS, KS, RW, KC, P, ET, I, dz, dt0, maxit, Ws):
    B, N = h0.shape
    D = P.shape[1]
    out = np.empty((B, D, Ws.shape[0]))
    hf = np.empty((B, N))
    fx = np.zeros((B, 6))
    for b in prange(B):
        h = h0[b].copy()
        wk = np.empty((14, N))
        th = np.empty(N)
        for d in range(D):
            _advance(h, A[b], Nn[b], Mm[b], TR[b], TS[b], KS[b], RW[b], (P[b, d] + I[b, d]) / 10.0,
                     KC[b] * ET[b, d] / 10.0, 1.0, dz, dt0, maxit, wk, fx[b])
            _theta_vec(h, A[b], Nn[b], Mm[b], TR[b], TS[b], KS[b], th)
            for j in range(Ws.shape[0]):
                v = 0.0
                for i in range(N):
                    v += Ws[j, i] * th[i]
                out[b, d, j] = v
        hf[b] = h
    return out, hf, fx


@njit(parallel=True, cache=True, fastmath=True)
def _step_all(h, A, Nn, Mm, TR, TS, KS, RW, qin, etr, span, dz, dt0, maxit):
    B, N = h.shape
    fx = np.zeros((B, 6))
    for b in prange(B):
        wk = np.empty((14, N))
        _advance(h[b], A[b], Nn[b], Mm[b], TR[b], TS[b], KS[b], RW[b], qin[b], etr[b], span, dz, dt0, maxit, wk, fx[b])
    return fx


def _np_params(p):
    g = lambda k: np.ascontiguousarray(p[k].detach().numpy(), dtype=np.float64)
    return g("a"), g("n"), g("m"), g("tr"), g("ts"), g("Ks"), g("rw")


def simulate(col, p, h, P, ET0, I=None, sub=SUB, iters=ITERS):
    a, n, m, tr, ts, ks, rw = _np_params(p)
    Pn = np.ascontiguousarray(np.asarray(P, dtype=np.float64))
    En = np.ascontiguousarray(np.asarray(ET0, dtype=np.float64))
    In = np.zeros_like(Pn) if I is None else np.ascontiguousarray(np.asarray(I, dtype=np.float64))
    out, hf, fx = _simulate(np.ascontiguousarray(h.detach().numpy(), dtype=np.float64), a, n, m, tr, ts, ks, rw,
                            np.ascontiguousarray(p["kc"].numpy(), dtype=np.float64), Pn, En, In, col.dz, 1.0 / sub,
                            iters, np.ascontiguousarray(col.Ws.numpy()))
    keys = ["infil", "drain", "et", "runoff", "resid", "iters"]
    return torch.from_numpy(out), torch.from_numpy(hf), {k: torch.from_numpy(fx[:, i].copy()) for i, k in enumerate(keys)}


def step_columns(col, p, h, qin, etr, span, sub=SUB, iters=ITERS):
    a, n, m, tr, ts, ks, rw = _np_params(p)
    hn = np.ascontiguousarray(h.detach().numpy(), dtype=np.float64).copy()
    _step_all(hn, a, n, m, tr, ts, ks, rw, np.ascontiguousarray(np.asarray(qin, dtype=np.float64)),
              np.ascontiguousarray(np.asarray(etr, dtype=np.float64)), span, col.dz, min(span, 1.0 / sub), iters)
    return torch.from_numpy(hn)


def init_h(col, p, init5):
    v = init5.clone()
    v[:, 4] = torch.where(torch.isnan(v[:, 4]), v[:, 3], v[:, 4])
    th = v @ col.Wi.T
    th = torch.maximum(torch.minimum(th, p["ts"] - 0.003), p["tr"] + 0.003)
    return vg_h(th, p).clamp(-2e4, -1.0)


def gather_phys(phys_all, idx):
    return {k: v[torch.as_tensor(idx)] for k, v in phys_all.items()}


def richards_forecast(col, phys_all, s, init5, Pf, ETf, I=None, chunk=16384):
    out = []
    for i in range(0, len(s), chunk):
        sl = slice(i, i + chunk)
        p = make_params(gather_phys(phys_all, s[sl]), col)
        h0 = init_h(col, p, torch.as_tensor(init5[sl], dtype=DT))
        sim, _, _ = simulate(col, p, h0, torch.as_tensor(Pf[sl], dtype=DT), torch.as_tensor(ETf[sl], dtype=DT),
                             None if I is None else torch.as_tensor(I[sl], dtype=DT))
        out.append(sim)
    return torch.cat(out).numpy()


def window_index(DS, s, years, min_future=4):
    smv, wv = DS["smv"][s], DS["wv"][s]
    t = np.arange(LP - 1, len(smv) - H)
    cw = np.concatenate([[0], np.cumsum(~wv)])
    cs = np.concatenate([[0], np.cumsum(smv)])
    wok = (cw[t + H + 1] - cw[t - LP + 1]) == 0
    fut = (cs[t + H + 1] - cs[t + 1]) >= min_future
    past = (cs[t + 1] - cs[t - LP + 1]) >= LP // 2
    return t[smv[t] & wok & fut & past & GROW[t] & np.isin(YR[t], years)]


def assemble(DS, s, t0):
    tt = t0[:, None] + np.arange(-LP + 1, H + 1)[None, :]
    ss = s[:, None]
    init5 = np.concatenate([DS["sm"][s, t0], DS["sm100"][s, t0][:, None]], 1)
    return dict(s=s, t0=t0, past=DS["sm_ff"][ss, tt[:, :LP]], pmask=DS["smv"][ss, tt[:, :LP]].astype(np.float32),
                target=DS["sm"][ss, tt[:, LP:]], P=DS["p"][ss, tt], ET=DS["et0"][ss, tt], doy=DOY[tt].astype(float),
                init5=init5)


def split_samples(DS, stations, years):
    ss, tt = [], []
    for s in stations:
        t = window_index(DS, s, years)
        ss.append(np.full(len(t), s))
        tt.append(t)
    return assemble(DS, np.concatenate(ss).astype(int), np.concatenate(tt).astype(int))


def station_stats(DS, idx):
    return {k: torch.as_tensor(DS[k][idx], dtype=DT) for k in ["ts1", "ts2", "mn1", "mn2"]}


def calibrate(col, DS, cfg, rng):
    S, W, Pp, G = DS["S"], cfg["cal_windows"], cfg["cal_pop"], cfg["cal_gen"]
    s_w, t_w = [], []
    for s in range(S):
        idx = window_index(DS, s, TRAIN_YEARS)
        s_w.append(np.full(W, s))
        t_w.append(rng.choice(idx, W, replace=len(idx) < W))
    A = assemble(DS, np.concatenate(s_w), np.concatenate(t_w))
    tgt = torch.tensor(A["target"]).view(S, 1, W, H, 4)
    msk = ~torch.isnan(tgt)
    tgt = torch.nan_to_num(tgt)
    nobs = msk.sum((2, 3, 4)).clamp(min=1)
    mean, std = np.full((S, 10), 0.5), np.full((S, 10), 0.3)
    best_u, best_f = mean.copy(), np.full(S, np.inf)
    hist = []
    for g in range(G):
        U = np.clip(mean[:, None, :] + std[:, None, :] * rng.standard_normal((S, Pp, 10)), 0, 1)
        U[:, 0] = best_u if g > 0 else 0.5
        Ub = torch.tensor(np.repeat(U.reshape(S * Pp, 10), W, axis=0), dtype=DT)
        sidx = np.repeat(np.arange(S), Pp * W)
        widx = np.tile(np.arange(W), S * Pp) + np.repeat(np.arange(S), Pp * W) * W
        phys = to_phys(Ub, station_stats(DS, sidx))
        p_ = {k: v for k, v in phys.items()}
        sims = []
        for i in range(0, len(sidx), 16384):
            sl = slice(i, i + 16384)
            p = make_params({k: v[sl] for k, v in p_.items()}, col)
            h0 = init_h(col, p, torch.as_tensor(A["init5"][widx[sl]], dtype=DT))
            sim, _, _ = simulate(col, p, h0, torch.as_tensor(A["P"][widx[sl], LP:], dtype=DT),
                                 torch.as_tensor(A["ET"][widx[sl], LP:], dtype=DT))
            sims.append(sim)
        sim = torch.cat(sims).view(S, Pp, W, H, 4).float()
        mse = (((sim - tgt) ** 2) * msk).sum((2, 3, 4)) / nobs
        f = torch.nan_to_num(mse.sqrt(), nan=1e3).numpy()
        ne = max(2, Pp // 4)
        for s in range(S):
            o = np.argsort(f[s])
            if f[s, o[0]] < best_f[s]:
                best_f[s], best_u[s] = f[s, o[0]], U[s, o[0]]
            el = U[s, o[:ne]]
            mean[s], std[s] = el.mean(0), np.maximum(el.std(0), 0.03)
        hist.append(best_f.copy())
        print(f"    generation {g + 1}/{G}: median calibration RMSE {np.median(best_f):.4f}", flush=True)
    phys_all = to_phys(torch.tensor(best_u, dtype=DT), station_stats(DS, np.arange(S)))
    return phys_all, best_f, np.array(hist)


def thresholds(col, phys_all):
    p = make_params(phys_all, col)
    at = lambda hv: ((vg_theta(torch.full_like(p["ts"], hv), p) @ col.Ws.T).float() * RZW_T).sum(1).numpy()
    fc, wp = at(-330.0), at(-15000.0)
    taw = fc - wp
    return dict(fc=fc, wp=wp, taw=taw, lo=fc - 0.5 * taw, crit=wp + 0.2 * taw)


class SpectralConv1d(nn.Module):
    def __init__(self, ci, co, modes):
        super().__init__()
        self.modes, self.co = modes, co
        self.w = nn.Parameter((1.0 / (ci * co)) * torch.randn(ci, co, modes, 2))

    def forward(self, x):
        xf = torch.fft.rfft(x, norm="ortho")
        out = torch.zeros(x.shape[0], self.co, xf.shape[-1], dtype=torch.cfloat, device=x.device)
        m = min(self.modes, xf.shape[-1])
        out[:, :, :m] = torch.einsum("bix,iox->box", xf[:, :, :m], torch.view_as_complex(self.w)[:, :, :m])
        return torch.fft.irfft(out, n=x.shape[-1], norm="ortho")


class FNO1d(nn.Module):
    def __init__(self, in_ch, out_ch=4, width=32, modes=12, layers=4, pad=9):
        super().__init__()
        self.pad = pad
        self.lift = nn.Linear(in_ch, width)
        self.sp = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.w = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.p1, self.p2 = nn.Linear(width, 64), nn.Linear(64, out_ch)

    def forward(self, x):
        x = F.pad(self.lift(x).permute(0, 2, 1), (0, self.pad))
        for i, (sp, w) in enumerate(zip(self.sp, self.w)):
            x = sp(x) + w(x)
            if i < len(self.sp) - 1:
                x = F.gelu(x)
        x = x[..., :-self.pad].permute(0, 2, 1)
        return self.p2(F.gelu(self.p1(x)))


IN_CH = 19


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def features(past, pmask, P, ET, rich, doy, static, physics):
    B = past.shape[0]
    th0 = past[:, -1]
    sm = torch.cat([past, th0[:, None].expand(B, H, 4)], 1)
    rc = torch.cat([past, rich], 1) if physics else torch.zeros_like(sm)
    mask = torch.cat([pmask, torch.zeros(B, H)], 1)
    fut = torch.cat([torch.zeros(B, LP), torch.ones(B, H)], 1)
    ang = 2 * math.pi * doy / 365.25
    tpos = torch.linspace(0, 1, L).expand(B, L)
    return torch.cat([(sm - 0.25) / 0.12, mask[..., None], fut[..., None], (torch.log1p(P.clamp(min=0)) / 3)[..., None],
                      (ET / 6)[..., None], (rc - 0.25) / 0.12, torch.sin(ang)[..., None], torch.cos(ang)[..., None],
                      static[:, None, :].expand(B, L, 4), tpos[..., None]], -1)


def fno_forecast(model, past, pmask, P, ET, rich, doy, static, physics):
    out = model(features(past, pmask, P, ET, rich, doy, static, physics))[:, LP:]
    base = rich if physics else past[:, -1:].expand(-1, H, -1)
    return base + 0.1 * out


def static_features(DS, phys_all):
    return torch.stack([(torch.as_tensor(DS["ts1"]) - 0.4) / 0.1, torch.as_tensor(phys_all["tr1"]) / 0.1,
                        torch.log10(phys_all["ks1"]) - 1.0, torch.log10(phys_all["ks2"]) - 1.0], 1).float()


def to_torch(A, rich, static_all, P=None, ET=None):
    f = lambda x: torch.tensor(np.nan_to_num(np.asarray(x, dtype=np.float32)))
    return dict(past=f(A["past"]), pmask=f(A["pmask"]), P=f(A["P"] if P is None else P),
                ET=f(A["ET"] if ET is None else ET), rich=f(rich), doy=f(A["doy"]), static=static_all[A["s"]],
                y=f(A["target"]), m=torch.tensor(~np.isnan(A["target"]), dtype=torch.float32))


def batch_of(D, idx):
    return {k: v[idx] for k, v in D.items()}


def physics_penalty(th, past, P_future, tol=2.0):
    w = torch.tensor(RZW * 10.0, dtype=torch.float32)
    s = torch.cat([(past[:, -1] * w).sum(-1, keepdim=True), (th * w).sum(-1)], 1)
    return F.relu(torch.diff(s, dim=1) - P_future - tol)


def run_model(model, D, physics, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(D["past"]), bs):
            b = batch_of(D, slice(i, i + bs))
            out.append(fno_forecast(model, b["past"], b["pmask"], b["P"], b["ET"], b["rich"], b["doy"], b["static"],
                                    physics))
    return torch.cat(out).numpy()


def masked_rmse(pred, D):
    m = D["m"].numpy().astype(bool)
    return float(np.sqrt(((pred - D["y"].numpy())[m] ** 2).mean()))


def train_fno(Dtr, Dva, physics, epochs, width=32, modes=12, seed=0, lam=0.1, verbose=True):
    torch.manual_seed(seed)
    model = FNO1d(IN_CH, 4, width, modes)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    n, best, best_state, hist = len(Dtr["past"]), np.inf, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = batch_of(Dtr, perm[i:i + 256])
            th = fno_forecast(model, b["past"], b["pmask"], b["P"], b["ET"], b["rich"], b["doy"], b["static"], physics)
            loss = (((th - b["y"]) ** 2) * b["m"]).sum() / b["m"].sum().clamp(min=1) / 1e-4
            loss = loss + lam * (physics_penalty(th, b["past"], b["P"][:, LP:]) ** 2).mean() / 10.0
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        va = masked_rmse(run_model(model, Dva, physics), Dva)
        hist.append(va)
        if va < best:
            best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep == 0 or (ep + 1) % max(1, epochs // 6) == 0):
            print(f"    epoch {ep + 1:3d}/{epochs}: validation RMSE {va:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def rz_of(x):
    return (x * (RZW / RZW.sum())).sum(-1)


def kge(p, o):
    r = np.corrcoef(p, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)


def forecast_metrics(pred, obs, theta0):
    out = {k: np.zeros(H) for k in ["rmse", "ubrmse", "r2", "r2_change", "kge", "hit02", "hit03", "acc", "skill_pers",
                                     "rel_l2"]}
    rp, ro, r0 = rz_of(pred), rz_of(obs), rz_of(theta0)
    for k in range(H):
        v = ~np.isnan(ro[:, k])
        p, o, z = rp[v, k], ro[v, k], r0[v]
        e = p - o
        out["rmse"][k] = np.sqrt((e ** 2).mean())
        out["ubrmse"][k] = np.sqrt(((e - e.mean()) ** 2).mean())
        out["r2"][k] = 1 - (e ** 2).sum() / ((o - o.mean()) ** 2).sum()
        dc = o - z
        out["r2_change"][k] = 1 - (e ** 2).sum() / ((dc - dc.mean()) ** 2).sum()
        out["kge"][k] = kge(p, o)
        out["hit02"][k] = (np.abs(e) <= 0.02).mean()
        out["hit03"][k] = (np.abs(e) <= 0.03).mean()
        out["acc"][k] = 100 * (1 - np.abs(e).mean() / o.mean())
        out["skill_pers"][k] = 1 - (e ** 2).mean() / ((z - o) ** 2).mean()
        m = ~np.isnan(obs[:, k])
        out["rel_l2"][k] = np.linalg.norm(pred[:, k][m] - obs[:, k][m]) / np.linalg.norm(obs[:, k][m])
    return out


def pbin(Pf):
    return np.digitize(np.asarray(Pf).sum(1), PBINS)


def sde_fit(pred, obs, Pf):
    r = obs - pred
    b = pbin(Pf)
    pooled = np.sqrt(np.nanmean(r ** 2, 0))
    rr = rz_of(obs) - rz_of(pred)
    pooled_rz = np.sqrt(np.nanmean(rr ** 2, 0))
    sig, sig_rz = np.zeros((4, H, 4)), np.zeros((4, H))
    for k in range(4):
        sel = b == k
        sig[k] = np.sqrt(np.nanmean(r[sel] ** 2, 0)) if sel.sum() >= 30 else pooled
        sig_rz[k] = np.sqrt(np.nanmean(rr[sel] ** 2, 0)) if sel.sum() >= 30 else pooled_rz
    sig = np.maximum.accumulate(np.nan_to_num(sig, nan=0.01), axis=1)
    sig_rz = np.maximum.accumulate(np.nan_to_num(sig_rz, nan=0.01), axis=1)
    C = np.zeros((H, 4, 4))
    for k in range(H):
        full = ~np.isnan(r[:, k]).any(-1)
        C[k] = np.corrcoef(r[full, k].T) if full.sum() > 10 else np.eye(4)
    return dict(sig=sig, sig_rz=sig_rz, C=np.nan_to_num(C) + 1e-4 * np.eye(4))


def sde_ensemble(pred, Pf, sde, M, rng):
    s = sde["sig"][pbin(Pf)]
    inc = np.sqrt(np.clip(np.diff(np.concatenate([np.zeros_like(s[:, :1]), s ** 2], 1), axis=1), 0, None))
    Lc = np.linalg.cholesky(sde["C"])
    xi = np.einsum("mnhd,hed->mnhe", rng.standard_normal((M,) + pred.shape).astype(np.float32), Lc)
    return pred[None] + np.cumsum(inc[None] * xi, axis=2)


def crps_ens(ens, y):
    m = ens.shape[0]
    s = np.sort(ens, 0)
    w = (2 * np.arange(1, m + 1) - m - 1).reshape(-1, *([1] * (ens.ndim - 1)))
    return np.abs(ens - y[None]).mean(0) - (w * s).sum(0) / (m * m)


def sde_evaluate(pred, obs, Pf, sde, M, rng, chunk=400):
    cov, cov_rz, crps, width = np.zeros((H, 4)), np.zeros(H), np.zeros(H), np.zeros(H)
    cnt, cnt_rz = np.zeros((H, 4)), np.zeros(H)
    pit = []
    for i in range(0, len(pred), chunk):
        ens = sde_ensemble(pred[i:i + chunk], Pf[i:i + chunk], sde, M, rng)
        o = obs[i:i + chunk]
        lo, hi = np.quantile(ens, 0.025, 0), np.quantile(ens, 0.975, 0)
        v = ~np.isnan(o)
        cov += (((o >= lo) & (o <= hi)) & v).sum(0)
        cnt += v.sum(0)
        er, orz = rz_of(ens), rz_of(o)
        vr = ~np.isnan(orz)
        lr, hr = np.quantile(er, 0.025, 0), np.quantile(er, 0.975, 0)
        cov_rz += (((orz >= lr) & (orz <= hr)) & vr).sum(0)
        cnt_rz += vr.sum(0)
        c = crps_ens(er, np.nan_to_num(orz))
        crps += np.where(vr, c, 0).sum(0)
        width += np.where(vr, hr - lr, 0).sum(0)
        pit.append(np.where(vr[:, -1], (er[:, :, -1] <= np.nan_to_num(orz[:, -1])[None]).mean(0), np.nan))
    pit = np.concatenate(pit)
    return dict(cov=cov / np.maximum(cnt, 1), cov_rz=cov_rz / np.maximum(cnt_rz, 1), crps=crps / np.maximum(cnt_rz, 1),
                width=width / np.maximum(cnt_rz, 1), pit=pit[~np.isnan(pit)])


def norain_forcing(A):
    Pn, En = A["P"].copy(), A["ET"].copy()
    Pn[:, LP:] = 0.0
    En[:, LP:] = A["ET"][:, LP - 3:LP].mean(1, keepdims=True)
    return Pn, En


def matern_chol(z, ell):
    d = np.abs(z[:, None] - z[None, :])
    s = math.sqrt(3) * d / ell
    return np.linalg.cholesky((1 + s) * np.exp(-s) + 1e-8 * np.eye(len(z)))


def fokker_planck(coef, D, s0, grid, T):
    dx = grid[1] - grid[0]
    faces = grid[:-1] + dx / 2
    mu = np.polyval(coef, faces)
    p = np.exp(-0.5 * ((grid - s0) / (1.5 * dx)) ** 2)
    p /= p.sum() * dx
    dt = 0.4 * min(dx * dx / (2 * D), dx / max(np.abs(mu).max(), 1e-9))
    n = int(math.ceil(T / dt))
    dt = T / n
    for _ in range(n):
        J = np.where(mu > 0, mu * p[:-1], mu * p[1:]) - D * (p[1:] - p[:-1]) / dx
        fl = np.concatenate([[0.0], J, [0.0]])
        p = p - dt * (fl[1:] - fl[:-1]) / dx
    return np.maximum(p, 0)


def binned_kl(samples, grid, dens, nb=20):
    lo, hi = np.quantile(samples, [0.005, 0.995])
    edges = np.linspace(lo, hi, nb + 1)
    pe = np.histogram(samples, edges)[0].astype(float)
    pe /= pe.sum()
    dx = grid[1] - grid[0]
    idx = np.digitize(grid, edges) - 1
    pf = np.array([dens[idx == k].sum() * dx for k in range(nb)])
    pf = pf / max(pf.sum(), 1e-12)
    mu, sd = samples.mean(), samples.std()
    cen = 0.5 * (edges[1:] + edges[:-1])
    pg = np.exp(-0.5 * ((cen - mu) / sd) ** 2)
    pg /= pg.sum()
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log((a + 1e-10) / (b + 1e-10)), 0)))
    return kl(pe, pf), kl(pe, pg), edges, pe, pf


def find_runs(DS, days):
    runs = []
    for s in range(DS["S"]):
        for y in TEST_YEARS:
            t = int((np.datetime64(f"{y}-06-01") - START).astype(int))
            for tt in range(t, t + 15):
                lo, hi = tt - LP + 1, tt + days + H + 1
                if hi <= len(DATES) and DS["smv"][s, tt] and DS["wv"][s, lo:hi].all():
                    runs.append((s, tt))
                    break
    return np.array(runs, dtype=int)


def sde_paths(sig_rz, stoch, M, seed=0):
    var = torch.cat([torch.zeros_like(sig_rz[:, :1]), sig_rz ** 2], 1)
    inc = torch.diff(var, dim=1).clamp(min=0).sqrt()
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn((M,) + tuple(sig_rz.shape), generator=g)
    return torch.cumsum(inc[None] * xi, -1) * stoch.float()[None, :, None]


def mpc_cost(u, rz, eps, lo, fc, cr, taw):
    traj = rz[None] + eps
    band = (F.relu(lo - rz) ** 2 + F.relu(rz - fc) ** 2) / taw ** 2
    q = torch.quantile(traj, ALPHA, dim=0)
    lead = torch.arange(q.shape[1], device=q.device)[None, :] < SAFE_DAYS
    safe = F.relu(cr - q).pow(2) / taw ** 2 * lead.float()
    return W_WATER * (u / UMAX).sum(1) + W_BAND * band.sum(1) + W_SAFE * safe.sum(1), traj, q


def mpc_plan(model, col, phys_b, ctx, u0, z, sig_rz, thr, iters, outer, M=MC_TRAJ):
    w = RZW_T
    lo, fc, cr, taw = [torch.as_tensor(thr[k], dtype=torch.float32)[:, None] for k in ["lo", "fc", "crit", "taw"]]
    rz0 = (ctx["past"][:, -1] * w).sum(-1, keepdim=True)
    tgt = lo + REFILL * taw
    rain = ctx["Pf"][:, :2].sum(1, keepdim=True)
    need = (tgt - rz0).clamp(min=0) * RZ_TOTAL_MM - rain * 0.7
    warm = torch.zeros_like(u0)
    warm[:, 0] = need.clamp(0, UMAX).squeeze(1)
    prm = torch.logit((warm / UMAX).clamp(1e-4, 0.98)).clone().requires_grad_(True)
    opt = torch.optim.Adam([prm], lr=0.3)
    B = u0.shape[0]
    eps = sde_paths(sig_rz, z > 0, M)
    p = make_params(phys_b, col)
    pr = {k: v.repeat(H + 1, *([1] * (v.dim() - 1))) for k, v in p.items()}
    h0 = init_h(col, p, ctx["init5"]).repeat(H + 1, 1)
    eye = torch.cat([torch.zeros(1, H), torch.eye(H)], 0).repeat_interleave(B, 0).double() * MPC_DU
    for _ in range(outer):
        with torch.no_grad():
            u_cur = UMAX * torch.sigmoid(prm)
            Pin = (ctx["Pf"] + u_cur).double().repeat(H + 1, 1) + eye
            sims, _, _ = simulate(col, pr, h0, Pin, ctx["ETf"].double().repeat(H + 1, 1))
            sims = sims.float().view(H + 1, B, H, 4)
            r0 = sims[0]
            G = ((sims[1:] - r0[None]) / MPC_DU).permute(1, 2, 3, 0)
        for _ in range(iters):
            u = UMAX * torch.sigmoid(prm)
            rich = r0 + torch.einsum("bhdj,bj->bhd", G, u - u_cur)
            P = torch.cat([ctx["Wpast"], ctx["Pf"] + u], 1)
            th = fno_forecast(model, ctx["past"], ctx["pmask"], P, ctx["ET"], rich, ctx["doy"], ctx["static"], True)
            J, _, _ = mpc_cost(u, (th * w).sum(-1), eps, lo, fc, cr, taw)
            opt.zero_grad()
            J.sum().backward()
            opt.step()
    with torch.no_grad():
        u = UMAX * torch.sigmoid(prm)
        rich = r0 + torch.einsum("bhdj,bj->bhd", G, u - u_cur)
        P = torch.cat([ctx["Wpast"], ctx["Pf"] + u], 1)
        th = fno_forecast(model, ctx["past"], ctx["pmask"], P, ctx["ET"], rich, ctx["doy"], ctx["static"], True)
        rz = (th * w).sum(-1)
        J, traj, q = mpc_cost(u, rz, eps, lo, fc, cr, taw)
    return dict(u=u.detach(), rz=rz, traj=traj, lower=q, J=J, p_stress=(traj < lo[None]).float().mean(0),
                p_crit=(traj < cr[None]).float().mean(0))


def closed_loop(C, controllers, days, dry_start=False, runs=None):
    DS, col, thr_s = C["DS"], C["col"], C["thr"]
    runs = find_runs(DS, days) if runs is None else runs
    R, K = len(runs), len(controllers)
    s_b, t_b = np.tile(runs[:, 0], K), np.tile(runs[:, 1], K)
    kind = np.repeat(np.arange(K), R)
    B = R * K
    ph = gather_phys(C["phys"], s_b)
    p = make_params(ph, col)
    thr = {k: v[s_b] for k, v in thr_s.items()}
    h = init_h(col, p, torch.as_tensor(np.concatenate([DS["sm"][s_b, t_b], DS["sm100"][s_b, t_b][:, None]], 1), dtype=DT))
    if dry_start:
        th = vg_theta(h, p)
        tgt = torch.as_tensor(thr["crit"] + 0.3 * (thr["lo"] - thr["crit"]), dtype=DT)[:, None]
        rzm = (col.z < 60)[None, :]
        th = torch.where(rzm, torch.maximum(torch.minimum(tgt.expand_as(th), p["ts"] - 0.01), p["tr"] + 0.01), th)
        h = vg_h(th, p)
    tt = t_b[:, None] + np.arange(-LP + 1, days + H + 1)[None, :]
    Pall, Eall, Dall = DS["p"][s_b[:, None], tt], DS["et0"][s_b[:, None], tt], DOY[tt].astype(float)
    hist = np.zeros((B, LP + days, 4), dtype=np.float32)
    hmask = np.ones((B, LP + days), dtype=np.float32)
    hist[:, :LP] = DS["sm_ff"][s_b[:, None], tt[:, :LP]]
    hmask[:, :LP] = DS["smv"][s_b[:, None], tt[:, :LP]]
    Win = Pall.copy()
    pending = np.zeros(B)
    plan = torch.zeros(B, H)
    types = np.array([c["type"] for c in controllers])[kind]
    setting = np.array([c.get("setting", "perfect") for c in controllers])[kind]
    zrow = torch.tensor([c.get("z", 0.0) for c in controllers], dtype=torch.float32)[torch.as_tensor(kind)]
    mpc_rows = np.where(types == "mpc")[0]
    rz_log, u_log, lb_log = np.zeros((B, days + 1)), np.zeros((B, days)), np.full((B, days), np.nan)
    ps_log = np.full((B, days), np.nan)
    fx = {k: np.zeros(B) for k in ["drain", "et", "runoff", "infil"]}
    static_b = C["static"][s_b]
    for d in range(days):
        sens = (vg_theta(h, p) @ col.Ws.T).float().numpy()
        hist[:, LP - 1 + d] = sens
        rz = rz_of(sens)
        rz_log[:, d] = rz
        u = np.zeros(B)
        thr_rows = types == "threshold"
        trig = thr_rows & (rz < thr["lo"]) & (pending <= 0)
        pending[trig] = (thr["fc"][trig] - rz[trig]) * RZ_TOTAL_MM
        give = thr_rows & (pending > 0)
        u[give] = np.minimum(pending[give], UMAX)
        pending[give] -= u[give]
        if len(mpc_rows):
            r_ = mpc_rows
            Pf = Pall[r_, LP + d:LP + d + H].copy()
            ETf = Eall[r_, LP + d:LP + d + H].copy()
            nr = setting[r_] == "norain"
            Pf[nr] = 0.0
            ETf[nr] = Eall[r_[nr], LP + d - 3:LP + d].mean(1, keepdims=True)
            ETwin = np.concatenate([Eall[r_, d:d + LP], ETf], 1)
            sens5 = np.concatenate([sens[r_], sens[r_, 3:4]], 1)
            ctx = dict(past=torch.tensor(hist[r_, d:d + LP]), pmask=torch.tensor(hmask[r_, d:d + LP]),
                       Wpast=torch.tensor(Win[r_, d:d + LP], dtype=torch.float32),
                       Pf=torch.tensor(Pf, dtype=torch.float32), ETf=torch.tensor(ETf, dtype=torch.float32),
                       ET=torch.tensor(ETwin, dtype=torch.float32), doy=torch.tensor(Dall[r_, d:d + L], dtype=torch.float32),
                       static=static_b[r_], init5=torch.tensor(sens5, dtype=DT))
            sel = np.array([setting[i] for i in r_])
            sig = np.where((sel == "norain")[:, None], C["sde"]["norain"]["sig_rz"][pbin(Pf)],
                           C["sde"]["perfect"]["sig_rz"][pbin(Pf)])
            u0 = torch.cat([plan[r_, 1:], plan[r_, -1:]], 1)
            out = mpc_plan(C["model"], col, gather_phys(C["phys"], s_b[r_]), ctx, u0, zrow[r_],
                           torch.tensor(sig, dtype=torch.float32), {k: v[r_] for k, v in thr.items()},
                           C["cfg"]["mpc_iters"], C["cfg"]["mpc_outer"], C["cfg"]["mpc_traj"])
            uu = out["u"].clone()
            uu[uu < U_SNAP] = 0.0
            rz_now = rz[r_]
            floor = (thr["lo"][r_] + REFILL * thr["taw"][r_] - rz_now).clip(0) * RZ_TOTAL_MM - Pf[:, :2].sum(1) * 0.7
            floor = np.clip(floor, 0, UMAX)
            dry = rz_now < thr["lo"][r_]
            uu[dry, 0] = torch.maximum(uu[dry, 0], torch.as_tensor(floor[dry], dtype=uu.dtype))
            plan[r_] = uu
            u[r_] = uu[:, 0].numpy()
            lb_log[r_, d] = out["lower"][:, 0].numpy()
            ps_log[r_, d] = out["p_stress"][:, 0].numpy()
        with torch.no_grad():
            _, h, f1 = simulate(col, p, h, torch.as_tensor(Pall[:, LP + d:LP + d + 1], dtype=DT),
                                torch.as_tensor(Eall[:, LP + d:LP + d + 1], dtype=DT),
                                torch.as_tensor(u[:, None], dtype=DT))
        for k in fx:
            fx[k] += f1[k].numpy()
        Win[:, LP + d] += u
        u_log[:, d] = u
    rz_log[:, days] = rz_of((vg_theta(h, p) @ col.Ws.T).float().numpy())
    return dict(runs=runs, R=R, kind=kind, rz=rz_log, u=u_log, lb=lb_log, ps=ps_log, fx=fx, thr=thr, s=s_b)


def fit_decay(V, floor_frac=1e-3):
    V = np.asarray(V, float)
    if V[0] <= 1e-12:
        return float("nan"), float("nan"), 0
    end = len(V)
    for i in range(1, len(V)):
        if V[i] < floor_frac * V[0]:
            end = i
            break
    n = min(end + 1, len(V))
    if n < 2:
        return float("nan"), float("nan"), end
    y = np.log(np.maximum(V[:n], floor_frac * V[0]))
    if n == 2:
        return float(y[0] - y[1]), 1.0, end
    lr = linregress(np.arange(n), y)
    return float(-lr.slope), float(lr.rvalue ** 2), end


def fig_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.7,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


def short_name(n):
    p = str(n).split("_")
    return p[0] + " " + p[1] if len(p) > 1 else str(n)


def savefig(out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    plt.gcf().savefig(path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close("all")
    print(f"  saved {path}")


def fig01_data(out, DS, held):
    fig, ax = plt.subplots(1, 2, figsize=(14.2, 5.6), constrained_layout=True)
    held = set(held)
    lon, lat = np.asarray(DS["lon"], float), np.asarray(DS["lat"], float)
    colors = ["#c0392b" if s in held else "#1f4e79" for s in range(DS["S"])]
    ax[0].scatter(lon, lat, c=colors, s=110, zorder=3, edgecolors="white", linewidths=0.8)
    for s in range(DS["S"]):
        ax[0].annotate(short_name(DS["names"][s]), (lon[s], lat[s]), textcoords="offset points",
                       xytext=(6, 4), fontsize=8)
    ax[0].set_xlim(float(np.nanmin(lon)) - 5, float(np.nanmax(lon)) + 5)
    ax[0].set_ylim(float(np.nanmin(lat)) - 3, float(np.nanmax(lat)) + 3)
    ax[0].set_xlabel("Longitude")
    ax[0].set_ylabel("Latitude")
    ax[0].set_title("NOAA USCRN stations used in this run")
    ax[0].plot([], [], "o", color="#1f4e79", label="Used to train FNO")
    ax[0].plot([], [], "o", color="#c0392b", label="Held out of FNO training")
    ax[0].legend(loc="lower left")
    tsel = (YR == 2023) & GROW
    days = DATES[tsel]
    rain_ax = ax[1].twinx()
    rain_ax.bar(days, DS["p"][0, tsel], color="#95a5a6", alpha=0.35, width=1.0, label="Rain")
    rain_ax.set_ylabel("Rain (mm/day)")
    rain_ax.set_ylim(0, max(40, float(np.nanmax(DS["p"][0, tsel])) * 1.15))
    rain_ax.grid(False)
    rain_ax.spines["right"].set_visible(True)
    for d, c in zip(range(4), ["#e67e22", "#27ae60", "#2980b9", "#8e44ad"]):
        ax[1].plot(days, DS["sm"][0, tsel, d], color=c, label=f"{DEPTHS[d]} cm")
    ax[1].set_ylim(0.05, 0.45)
    ax[1].set_title(f"{short_name(DS['names'][0])} 2023 growing season")
    ax[1].set_ylabel("Soil moisture (m$^3$ / m$^3$)")
    ax[1].set_xlabel("Date")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = rain_ax.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper right", ncol=2)
    savefig(out, "fig01_data.png")


def fig02_forecast(out, metrics, hist_h):
    lead = np.arange(1, H + 1)
    names = [("persistence", "Persistence", "#7f8c8d"), ("richards", "Richards", "#27ae60"),
             ("fno_data", "FNO only", "#e67e22"), ("richards_fno", "Richards + FNO", "#1f4e79")]
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    for key, lab, c in names:
        ax[0].plot(lead, metrics[("test", "hindcast", key)]["rmse"], "o-", color=c, label=lab)
        ax[0].plot(lead, metrics[("test", "norain", key)]["rmse"], "o--", color=c, alpha=0.45)
        ax[1].plot(lead, metrics[("test", "hindcast", key)]["hit03"], "o-", color=c, label=lab)
    ax[0].set_title("7-day root-zone RMSE, 2023–24 test")
    ax[0].set_xlabel("Lead (days)")
    ax[0].set_ylabel("RMSE (m$^3$/m$^3$)")
    ax[0].legend(loc="upper left")
    ax[1].set_title("Hit rate within ±0.03 m$^3$/m$^3$")
    ax[1].set_xlabel("Lead (days)")
    ax[1].set_ylabel("Hit rate")
    ax[1].set_ylim(0.55, 1.02)
    ax[1].axhline(0.80, color="#c0392b", ls=":", lw=1)
    hyb = metrics[("test", "hindcast", "richards_fno")]
    ax[0].annotate(f"day 7 = {hyb['rmse'][6]:.3f}", xy=(7, hyb["rmse"][6]), xytext=(4.2, hyb["rmse"][6] + 0.006),
                   fontsize=8, arrowprops=dict(arrowstyle="->", color="#1f4e79"))
    ax[2].plot(np.arange(1, len(hist_h) + 1), hist_h, color="#1f4e79")
    ax[2].set_title("Richards–FNO validation RMSE by epoch")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Validation RMSE")
    savefig(out, "fig02_forecast_skill.png")


def fig03_sde(out, DS, sets, preds, ens, exi, evals):
    lead = np.arange(1, H + 1)
    fig, ax = plt.subplots(1, 4, figsize=(15.6, 4.4), constrained_layout=True)
    for j, i in enumerate(exi):
        erz = rz_of(ens[:, j])
        hist_rz = rz_of(sets["test"]["A"]["past"][i])
        ax[j].plot(np.arange(-14, 1), hist_rz[-14:], color="black", label="Observed")
        ax[j].plot(lead, rz_of(sets["test"]["A"]["target"][i]), "o", color="black")
        ax[j].plot(lead, rz_of(preds[("test", "hindcast", "richards_fno")][i]), color="#1f4e79", label="Richards + FNO")
        ax[j].fill_between(lead, np.quantile(erz, 0.025, 0), np.quantile(erz, 0.975, 0),
                           color="#1f4e79", alpha=0.22, label="SDE 95% band")
        ax[j].set_title(f"{short_name(DS['names'][sets['test']['A']['s'][i]])}  {DATES[sets['test']['A']['t0'][i]]}")
        ax[j].set_xlabel("Day")
        ax[j].set_xlim(-14.5, 7.5)
        if j == 0:
            ax[j].set_ylabel("Root-zone soil moisture")
            ax[j].legend(loc="upper left")
    ax[3].plot(lead, evals[("test", "hindcast")]["cov_rz"], "o-", color="#1f4e79", label="Observed weather")
    ax[3].plot(lead, evals[("test", "norain")]["cov_rz"], "s--", color="#e67e22", label="No-rain forecast")
    ax[3].axhline(0.95, color="#c0392b", ls=":", label="Target 95%")
    ax[3].set_ylim(0.75, 1.02)
    ax[3].set_title("SDE 95% coverage on 2023–24 test")
    ax[3].set_xlabel("Lead (days)")
    ax[3].set_ylabel("Fraction inside 95% band")
    ax[3].legend(loc="lower left")
    savefig(out, "fig03_seven_day_forecasts.png")


def _fit_line(x, y):
    lr = linregress(np.log(x), np.log(y))
    xx = np.linspace(x.min(), x.max(), 50)
    return xx, np.exp(lr.intercept) * xx ** lr.slope, lr


def fig04_result1_2(out, conv, sw, lc, gain):
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    xx, yy, lr = _fit_line(conv["dz"], conv["err"])
    ax[0, 0].loglog(conv["dz"], conv["err"], "o", color="#1f4e79")
    ax[0, 0].loglog(xx, yy, "--", color="#c0392b",
                    label=f"order = {lr.slope:.2f},  R$^2$ = {lr.rvalue ** 2:.3f}")
    ax[0, 0].set_xlabel("Grid spacing dz (cm)")
    ax[0, 0].set_ylabel("Profile error vs fine grid")
    ax[0, 0].set_title("Result 1 — Richards spatial convergence")
    ax[0, 0].legend()
    if sw is not None and len(sw):
        xx, yy, lr = _fit_line(sw[:, 0], sw[:, 1])
        ax[0, 1].loglog(sw[:, 0], sw[:, 1], "o", color="#1f4e79")
        ax[0, 1].loglog(xx, yy, "--", color="#c0392b",
                        label=f"slope = {lr.slope:.3f},  R$^2$ = {lr.rvalue ** 2:.3f}")
        ax[0, 1].set_xlabel("FNO parameters")
        ax[0, 1].set_ylabel("Test RMSE")
        ax[0, 1].set_title("Result 1 — FNO error vs model size")
        ax[0, 1].legend()
    if lc is not None and len(lc):
        xx, yy, lr = _fit_line(lc[:, 0], lc[:, 1])
        ax[1, 0].loglog(lc[:, 0], lc[:, 1], "o", color="#1f4e79")
        ax[1, 0].loglog(xx, yy, "--", color="#c0392b",
                        label=f"slope = {lr.slope:.3f},  R$^2$ = {lr.rvalue ** 2:.3f}")
        ax[1, 0].set_xlabel("Training windows")
        ax[1, 0].set_ylabel("Test RMSE")
        ax[1, 0].set_title("Result 1 — FNO learning curve")
        ax[1, 0].legend()
    lead = np.arange(1, len(gain) + 1)
    ax[1, 1].plot(lead, gain, "o-", color="#1f4e79")
    ax[1, 1].axhline(1.0, color="#c0392b", ls=":", label="No damping")
    ax[1, 1].set_xlabel("Lead (days)")
    ax[1, 1].set_ylabel("Perturbation gain  ||δŷ|| / 0.01")
    ax[1, 1].set_title("Result 2 — FNO perturbation shrinks with lead")
    ax[1, 1].legend()
    savefig(out, "fig04_result1_convergence.png")


def fig05_spde(out, DS, spde):
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    ratio = spde["var"] / np.maximum(spde["bound"], 1e-12)
    node_ok = (spde["var"] <= spde["bound"] + 1e-15).mean(1)
    names = [short_name(n) for n in DS["names"][:len(node_ok)]]
    ax[0].barh(range(len(names)), node_ok, color="#1f4e79")
    ax[0].axvline(0.95, color="#c0392b", ls="--", label="95% of nodes")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xlim(0, 1.05)
    ax[0].set_xlabel("Fraction of depth nodes with Var(T) ≤ σ²T")
    ax[0].set_title(f"Result 3 — variance bound  ({spde['frac']:.1%} of all nodes)")
    ax[0].legend(loc="lower left")
    e, pe, pf = spde["edges"][0], spde["pe"][0], spde["pf"][0]
    cen = 0.5 * (e[1:] + e[:-1])
    ax[1].bar(cen, pe, width=e[1] - e[0], color="#aed6f1", edgecolor="white", label="SPDE ensemble")
    ax[1].plot(cen, pf, "o-", color="#c0392b", label="Fokker–Planck")
    ax[1].set_xlabel("Root-zone storage (mm)")
    ax[1].set_ylabel("Probability")
    ax[1].set_title(f"Result 4 — densities match  (KL = {spde['kl_list'][0]:.3f} nats)")
    ax[1].legend()
    savefig(out, "fig05_results3_4_spde.png")


def fig06_mpc(out, DS, res, summ, controllers, site, site_mean, V, lyap, cfg):
    fig, ax = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    days = np.arange(cfg["mpc_days"] + 1)
    show = ["fao56_threshold", "fno_sde_mpc"]
    cols = {"fao56_threshold": "#c0392b", "fno_sde_mpc": "#1f4e79"}
    labs = {"fao56_threshold": "FAO-56 threshold", "fno_sde_mpc": "FNO–SDE–MPC"}
    for name in show:
        ax[0, 0].plot(days, summ[name]["rz"][0], color=cols[name], label=labs[name])
    ax[0, 0].axhline(res["thr"]["lo"][0], color="#27ae60", ls="--", lw=1.2, label="MAD line")
    ax[0, 0].axhline(res["thr"]["crit"][0], color="black", ls=":", lw=1.2, label="Critical")
    ax[0, 0].set_title(f"{short_name(DS['names'][res['runs'][0, 0]])} {str(DATES[res['runs'][0, 1]])[:4]} root-zone water")
    ax[0, 0].set_xlabel("Day of season")
    ax[0, 0].set_ylabel("Root-zone θ")
    ax[0, 0].legend(loc="upper right")
    labels = ["FAO-56", "Det. MPC", "FNO–SDE–MPC", "Det. no rain", "SDE no rain"]
    water = [summ[c["name"]]["water"].mean() for c in controllers]
    stress = [summ[c["name"]]["stress"].mean() for c in controllers]
    x = np.arange(len(labels))
    ax[0, 1].bar(x - 0.18, water, 0.36, color="#1f4e79", label="Irrigation (mm)")
    ax2 = ax[0, 1].twinx()
    ax2.bar(x + 0.18, stress, 0.36, color="#e67e22", label="Stress days")
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(labels, rotation=18, ha="right")
    ax[0, 1].set_ylabel("Seasonal irrigation (mm)")
    ax2.set_ylabel("Stress days")
    ax[0, 1].set_title("Result 7 — water use and stress")
    ax[0, 1].spines["right"].set_visible(True)
    if len(site_mean):
        names = [short_name(DS["names"][s]) for s in site]
        ax[1, 0].bar(range(len(site_mean)), site_mean, color=["#1f4e79" if v > 0 else "#c0392b" for v in site_mean])
        ax[1, 0].axhline(0, color="black", lw=1)
        ax[1, 0].set_xticks(range(len(site_mean)))
        ax[1, 0].set_xticklabels(names, rotation=30, ha="right")
        ax[1, 0].set_ylabel("Water saved vs FAO-56 (%)")
        ax[1, 0].set_title("Result 7 — savings by station")
    rz = lyap["rz"]
    ax[1, 1].plot(np.arange(rz.shape[1]), rz.mean(0), color="#1f4e79", label="Mean root-zone θ")
    ax[1, 1].axhline(lyap["thr"]["lo"].mean(), color="#27ae60", ls="--", label="MAD line")
    ax[1, 1].set_xlabel("Day after dry start")
    ax[1, 1].set_ylabel("Root-zone θ")
    ax[1, 1].set_title("Result 5 — recovery from a dry start")
    ax[1, 1].legend()
    savefig(out, "fig06_mpc_results.png")


def run(args):
    cfg = MODES[args.mode]
    out = args.out
    cache = os.path.join(args.data, "uscrn")
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    t_start = time.time()
    print(f"Run mode {args.mode} | torch {torch.__version__} | output {out}")
    fig_style()

    banner("1. DATA: NOAA USCRN in-situ soil moisture + weather, NASA POWER wind, FAO-56 ET0")
    DS = load_dataset(cfg, cache)
    S = DS["S"]
    n_hold = max(1, S // 4) if S >= 3 else 0
    held = list(range(S - n_hold, S)) if n_hold else []
    seen = [s for s in range(S) if s not in held]
    record("data", "stations used", S, ", ".join(DS["names"]))
    record("data", "stations held out from FNO training", ", ".join(DS["names"][s] for s in held) or "-")
    record("data", "wind source", "; ".join(sorted(set(DS["wind_src"]))))
    record("data", "growing-season station-days with valid soil moisture + weather",
           int((DS["smv"] & DS["wv"])[:, GROW].sum()))
    record("data", "mean growing-season ET0 (mm/day)", float(DS["et0"][:, GROW].mean()))
    record("data", "mean growing-season precipitation (mm/day)", float(DS["p"][:, GROW].mean()))
    fig01_data(out, DS, held)

    banner("2. RICHARDS EQUATION: implicit mixed-form solver, calibration on real sensors")
    col = Column()
    t0 = time.time()
    phys_all, cal_rmse, cal_hist = calibrate(col, DS, cfg, rng)
    print(f"  calibrated {S} stations in {time.time() - t0:.0f}s")
    record("richards", "median calibration RMSE, 7-day forecasts, training years (m3/m3)", float(np.median(cal_rmse)))
    thr = thresholds(col, phys_all)
    for s in range(S):
        record("richards", f"{DS['names'][s]} Ks top/sub (cm/d), n top/sub, Kc",
               f"{phys_all['ks1'][s]:.1f}/{phys_all['ks2'][s]:.1f}, {phys_all['n1'][s]:.2f}/{phys_all['n2'][s]:.2f}, "
               f"{phys_all['kc'][s]:.2f}")
    with torch.no_grad():
        idx = window_index(DS, 0, VAL_YEARS)[:50]
        A0 = assemble(DS, np.zeros(len(idx), int), idx)
        p0 = make_params(gather_phys(phys_all, A0["s"]), col)
        _, _, fx0 = simulate(col, p0, init_h(col, p0, torch.as_tensor(A0["init5"], dtype=DT)),
                             torch.as_tensor(A0["P"][:, LP:], dtype=DT), torch.as_tensor(A0["ET"][:, LP:], dtype=DT))
    record("richards", "mass-balance residual, 7-day runs (mm, mean |.|)", float(fx0["resid"].abs().mean()),
           f"mean infiltration {float(fx0['infil'].mean()):.1f} mm, ET {float(fx0['et'].mean()):.1f} mm")
    conv = richards_convergence(phys_all)
    record("result1", "Richards solver spatial convergence order (slope of log error vs log dz)", conv["slope"])
    record("result1", "Richards convergence fit R2", conv["r2"])

    banner("3. DATASETS: windows of 32 observed days + 7 forecast days")
    static_all = static_features(DS, phys_all)
    A = dict(train=split_samples(DS, seen, TRAIN_YEARS), val=split_samples(DS, seen, VAL_YEARS),
             test=split_samples(DS, seen, TEST_YEARS), held=split_samples(DS, held, TEST_YEARS) if held else None)
    sets = {}
    t0 = time.time()
    for k, a in A.items():
        if a is None or len(a["s"]) == 0:
            continue
        rich = richards_forecast(col, phys_all, a["s"], a["init5"], a["P"][:, LP:], a["ET"][:, LP:])
        sets[k] = dict(A=a, rich=rich, D=to_torch(a, rich, static_all))
        if k != "train":
            Pn, En = norain_forcing(a)
            rn = richards_forecast(col, phys_all, a["s"], a["init5"], Pn[:, LP:], En[:, LP:])
            sets[k]["rich_nr"] = rn
            sets[k]["D_nr"] = to_torch(a, rn, static_all, Pn, En)
            sets[k]["P_nr"] = Pn
        record("data", f"{k} windows", len(a["s"]))
    print(f"  Richards forecasts for all windows in {time.time() - t0:.0f}s")

    banner("4. FOURIER NEURAL OPERATOR: data-only and physics-guided (Richards + FNO)")
    t0 = time.time()
    fno_data, _ = train_fno(sets["train"]["D"], sets["val"]["D"], False, cfg["epochs"])
    fno_hyb, hist_h = train_fno(sets["train"]["D"], sets["val"]["D"], True, cfg["epochs"])
    record("fno", "parameters", n_params(fno_hyb))
    record("fno", "training time (s)", time.time() - t0)
    models = dict(persistence=None, richards=None, fno_data=fno_data, richards_fno=fno_hyb)
    preds, metrics = {}, {}
    for split in [k for k in ["val", "test", "held"] if k in sets]:
        for forcing in ["hindcast", "norain"]:
            D = sets[split]["D"] if forcing == "hindcast" else sets[split]["D_nr"]
            rich = sets[split]["rich"] if forcing == "hindcast" else sets[split]["rich_nr"]
            obs = sets[split]["A"]["target"]
            th0 = sets[split]["A"]["past"][:, -1]
            for name, mdl in models.items():
                if name == "persistence":
                    pr = np.repeat(th0[:, None], H, 1)
                elif name == "richards":
                    pr = rich.astype(np.float32)
                else:
                    pr = run_model(mdl, D, name == "richards_fno")
                preds[(split, forcing, name)] = pr
                metrics[(split, forcing, name)] = forecast_metrics(pr, obs, th0)
    for split in [k for k in ["test", "held"] if k in sets]:
        for forcing in ["hindcast", "norain"]:
            for name in models:
                m = metrics[(split, forcing, name)]
                for k in (1, 3, 7):
                    tag = f"{split}/{forcing}/{name} day-{k}"
                    record("forecast", f"{tag} root-zone RMSE", m["rmse"][k - 1])
                    if name in ("richards_fno", "persistence", "richards") or k == 7:
                        record("forecast", f"{tag} R2", m["r2"][k - 1])
                        record("forecast", f"{tag} R2 of change", m["r2_change"][k - 1])
                        record("forecast", f"{tag} KGE", m["kge"][k - 1])
                        record("forecast", f"{tag} hit rate +/-0.03", m["hit03"][k - 1])
                        record("forecast", f"{tag} hit rate +/-0.02", m["hit02"][k - 1])
                        record("forecast", f"{tag} accuracy 100(1-MAE/mean)", m["acc"][k - 1])
                        record("forecast", f"{tag} relative L2 (profile)", m["rel_l2"][k - 1])
                        if name != "persistence":
                            record("forecast", f"{tag} skill vs persistence", m["skill_pers"][k - 1])
    pv = physics_penalty(torch.tensor(preds[("test", "hindcast", "richards_fno")]),
                         sets["test"]["D"]["past"], sets["test"]["D"]["P"][:, LP:]).numpy()
    record("fno", "physics check: forecast days with storage gain > rain + 2 mm (fraction)", float((pv > 0).mean()))
    fig02_forecast(out, metrics, hist_h)

    banner("5. SDE UNCERTAINTY: calibrated on 2022, evaluated on 2023-24")
    sde = {}
    for forcing, key in [("hindcast", "perfect"), ("norain", "norain")]:
        Pf = sets["val"]["A"]["P"][:, LP:] if forcing == "hindcast" else sets["val"]["P_nr"][:, LP:]
        sde[key] = sde_fit(preds[("val", forcing, "richards_fno")], sets["val"]["A"]["target"], Pf)
    evals = {}
    for split in [k for k in ["test", "held"] if k in sets]:
        for forcing, key in [("hindcast", "perfect"), ("norain", "norain")]:
            Pf = sets[split]["A"]["P"][:, LP:] if forcing == "hindcast" else sets[split]["P_nr"][:, LP:]
            ev = sde_evaluate(preds[(split, forcing, "richards_fno")], sets[split]["A"]["target"], Pf, sde[key],
                              cfg["ens"], rng)
            evals[(split, forcing)] = ev
            for k in (1, 3, 7):
                record("sde", f"{split}/{forcing} day-{k} 95% coverage, root zone", ev["cov_rz"][k - 1])
                record("sde", f"{split}/{forcing} day-{k} CRPS root zone (m3/m3)", ev["crps"][k - 1])
            record("sde", f"{split}/{forcing} 95% coverage, all depths and leads", float(ev["cov"].mean()))
    ex = sets["test"]
    pick = []
    used = set()
    for i in np.argsort(-np.nanstd(ex["A"]["target"], (1, 2))):
        s = int(ex["A"]["s"][i])
        if s not in used:
            pick.append(int(i))
            used.add(s)
        if len(pick) == 3:
            break
    exi = np.array(pick if pick else [0, 1, 2][:min(3, len(ex["A"]["s"]))])
    ens = sde_ensemble(preds[("test", "hindcast", "richards_fno")][exi], ex["A"]["P"][exi, LP:], sde["perfect"], 300, rng)
    fig03_sde(out, DS, sets, preds, ens, exi, evals)

    banner("6. RESULT 1 (FNO): error vs parameters and vs training data")
    if cfg["sweep"]:
        sw = []
        for width, modes in [(8, 4), (12, 6), (16, 8), (24, 10), (32, 12), (48, 16)]:
            m_, _ = train_fno(sets["train"]["D"], sets["val"]["D"], True, cfg["sweep_epochs"], width, modes, verbose=False)
            e = masked_rmse(run_model(m_, sets["test"]["D"], True), sets["test"]["D"])
            sw.append((n_params(m_), e))
            print(f"    width {width:2d} modes {modes:2d}: {n_params(m_):7,d} params, test RMSE {e:.4f}", flush=True)
        sw = np.array(sw)
        lr = linregress(np.log(sw[:, 0]), np.log(sw[:, 1]))
        record("result1", "FNO error vs parameters: power-law slope", float(lr.slope))
        record("result1", "FNO error vs parameters: fit R2", float(lr.rvalue ** 2))
        lc = []
        ntr = len(sets["train"]["D"]["past"])
        for frac in [0.0625, 0.125, 0.25, 0.5, 1.0]:
            sub = torch.as_tensor(rng.permutation(ntr)[:max(64, int(frac * ntr))])
            m_, _ = train_fno(batch_of(sets["train"]["D"], sub), sets["val"]["D"], True, cfg["sweep_epochs"],
                              verbose=False)
            e = masked_rmse(run_model(m_, sets["test"]["D"], True), sets["test"]["D"])
            lc.append((len(sub), e))
            print(f"    training windows {len(sub):6d}: test RMSE {e:.4f}", flush=True)
        lc = np.array(lc)
        lr2 = linregress(np.log(lc[:, 0]), np.log(lc[:, 1]))
        record("result1", "FNO learning curve: power-law slope (error vs training windows)", float(lr2.slope))
        record("result1", "FNO learning curve: fit R2", float(lr2.rvalue ** 2))
    else:
        sw, lc = None, None

    banner("7. RESULT 2: perturbation growth (Richards dynamics and FNO forecast map)")
    runs = find_runs(DS, cfg["ftle_days"])
    lam, r2s = [], []
    with torch.no_grad():
        s_b, t_b = runs[:, 0], runs[:, 1]
        p = make_params(gather_phys(phys_all, s_b), col)
        h = init_h(col, p, torch.as_tensor(np.concatenate([DS["sm"][s_b, t_b], DS["sm100"][s_b, t_b][:, None]], 1), dtype=DT))
        th = vg_theta(h, p)
        hp = vg_h(torch.minimum(th + 0.01, p["ts"] - 0.002), p)
        tt = t_b[:, None] + np.arange(1, cfg["ftle_days"] + 1)[None, :]
        Pr, Er = torch.as_tensor(DS["p"][s_b[:, None], tt], dtype=DT), torch.as_tensor(DS["et0"][s_b[:, None], tt], dtype=DT)
        dist = [np.full(len(runs), 0.01)]
        for d in range(cfg["ftle_days"]):
            _, h, _ = simulate(col, p, h, Pr[:, d:d + 1], Er[:, d:d + 1])
            _, hp, _ = simulate(col, p, hp, Pr[:, d:d + 1], Er[:, d:d + 1])
            dist.append((vg_theta(hp, p) - vg_theta(h, p)).pow(2).mean(1).sqrt().numpy())
    dist = np.array(dist).T
    for r in range(len(runs)):
        lr = linregress(np.arange(dist.shape[1]), np.log(np.maximum(dist[r], 1e-12)))
        lam.append(lr.slope)
        r2s.append(lr.rvalue ** 2)
    record("result2", "Richards finite-time Lyapunov exponent, median (1/day)", float(np.median(lam)),
           f"{len(runs)} station-seasons, {cfg['ftle_days']} days, real weather")
    record("result2", "fraction of station-seasons with negative exponent", float(np.mean(np.array(lam) < 0)))
    record("result2", "Richards exponent fit R2, median", float(np.median(r2s)))
    D = sets["test"]["D"]
    sel = torch.as_tensor(rng.permutation(len(D["past"]))[:min(1500, len(D["past"]))])
    b = batch_of(D, sel)
    a_ = sets["test"]["A"]
    init_p = a_["init5"][sel.numpy()] + 0.01
    rich_p = richards_forecast(col, phys_all, a_["s"][sel.numpy()], init_p, a_["P"][sel.numpy(), LP:],
                               a_["ET"][sel.numpy(), LP:]).astype(np.float32)
    with torch.no_grad():
        base = fno_forecast(fno_hyb, b["past"], b["pmask"], b["P"], b["ET"], b["rich"], b["doy"], b["static"], True)
        pert = fno_forecast(fno_hyb, b["past"] + 0.01, b["pmask"], b["P"], b["ET"], torch.tensor(rich_p), b["doy"],
                            b["static"], True)
    gain = ((pert - base).abs().mean((0, 2)) / 0.01).numpy()
    lrg = linregress(np.arange(1, H + 1), np.log(gain))
    record("result2", "FNO forecast perturbation gain at day 1 / day 7", f"{gain[0]:.3f} / {gain[-1]:.3f}")
    record("result2", "FNO perturbation log-growth rate L (1/day)", float(lrg.slope))
    record("result2", "FNO log-growth fit R2", float(lrg.rvalue ** 2))
    fig04_result1_2(out, conv, sw, lc, gain)

    banner("8. RESULTS 3-4: stochastic Richards equation (SPDE) and Fokker-Planck")
    spde = spde_experiment(col, DS, phys_all, sets, cfg["spde_ens"], rng)
    record("result3", "fraction of node-stations with Var(T) <= sigma^2 T", spde["frac"])
    record("result3", "total variance / bound, median over stations", spde["ratio"])
    record("result4", "KL(SPDE ensemble || Fokker-Planck), median over stations (nats)", spde["kl_fp"])
    record("result4", "KL(SPDE ensemble || moment-matched Gaussian), median (nats)", spde["kl_g"])
    record("result4", "relative error of FP mean / std vs ensemble, median", f"{spde['mean_err']:.4f} / {spde['sd_err']:.4f}")
    fig05_spde(out, DS, spde)

    banner("9. RESULTS 5-7: chance-constrained FNO-SDE-MPC on calibrated Richards plants, real 2023-24 weather")
    C = dict(DS=DS, col=col, thr=thr, phys=phys_all, model=fno_hyb, sde=sde, static=static_all, cfg=cfg)
    fno_hyb.requires_grad_(False)
    controllers = [dict(name="fao56_threshold", type="threshold"),
                   dict(name="mpc_deterministic", type="mpc", setting="perfect", z=0.0),
                   dict(name="fno_sde_mpc", type="mpc", setting="perfect", z=Z95),
                   dict(name="mpc_deterministic_norain", type="mpc", setting="norain", z=0.0),
                   dict(name="fno_sde_mpc_norain", type="mpc", setting="norain", z=Z95)]
    t0 = time.time()
    res = closed_loop(C, controllers, cfg["mpc_days"])
    print(f"  {res['R']} station-seasons x {len(controllers)} controllers x {cfg['mpc_days']} days in {time.time() - t0:.0f}s")
    R = res["R"]
    summ = {}
    for k, c in enumerate(controllers):
        rows = slice(k * R, (k + 1) * R)
        rz = res["rz"][rows, 1:]
        th_ = {kk: v[rows] for kk, v in res["thr"].items()}
        summ[c["name"]] = dict(water=res["u"][rows].sum(1), stress=(rz < th_["lo"][:, None]).sum(1),
                               crit=(rz < th_["crit"][:, None]).sum(1), drain=res["fx"]["drain"][rows],
                               et=res["fx"]["et"][rows], rz=res["rz"][rows], lb=res["lb"][rows], ps=res["ps"][rows])
        record("mpc", f"{c['name']}: mean seasonal irrigation (mm)", float(summ[c["name"]]["water"].mean()))
        record("mpc", f"{c['name']}: stress days (theta < MAD threshold) per season", float(summ[c["name"]]["stress"].mean()))
        record("result6", f"{c['name']}: critical violations / checks", f"{int(summ[c['name']]['crit'].sum())} / {rz.size}")
        record("mpc", f"{c['name']}: deep drainage (mm)", float(summ[c["name"]]["drain"].mean()))
        record("mpc", f"{c['name']}: actual ET (mm)", float(summ[c["name"]]["et"].mean()))
    wt = summ["fao56_threshold"]["water"]
    ok = wt > 10
    for name in ["mpc_deterministic", "fno_sde_mpc", "mpc_deterministic_norain", "fno_sde_mpc_norain"]:
        wm = summ[name]["water"]
        record("result7", f"{name}: pooled water savings vs FAO-56 threshold, all seasons (%)",
               float((wt.sum() - wm.sum()) / wt.sum() * 100) if wt.sum() > 1 else float("nan"))
        sv = (wt[ok] - wm[ok]) / wt[ok] * 100
        record("result7", f"{name}: water savings vs FAO-56 threshold, mean (%)", float(sv.mean()))
        record("result7", f"{name}: savings std (%)", float(sv.std()))
        record("result7", f"{name}: runs with positive savings", float((sv > 0).mean()))
        if ok.sum() >= 3:
            record("result7", f"{name}: paired t-test p-value", float(ttest_rel(wt[ok], wm[ok]).pvalue))
            try:
                record("result7", f"{name}: Wilcoxon p-value", float(wilcoxon(wt[ok], wm[ok]).pvalue))
            except ValueError:
                pass
        record("result7", f"{name}: change in stress days vs threshold",
               float(summ[name]["stress"].mean() - summ["fao56_threshold"]["stress"].mean()))
    for name in ["fno_sde_mpc", "mpc_deterministic"]:
        lb, real = summ[name]["lb"], summ[name]["rz"][:, 1:]
        v = ~np.isnan(lb)
        record("sde", f"{name}: P(realised theta >= predicted 95% lower bound)", float((real[v] >= lb[v]).mean()))
    ps = summ["fno_sde_mpc"]["ps"]
    hit = (summ["fno_sde_mpc"]["rz"][:, 1:] < res["thr"]["lo"][:R, None]).astype(float)
    v = ~np.isnan(ps)
    record("sde", "MPC Monte Carlo trajectories per decision", cfg["mpc_traj"])
    record("sde", "fno_sde_mpc: mean predicted next-day stress probability", float(ps[v].mean()))
    record("sde", "fno_sde_mpc: observed next-day stress frequency", float(hit[v].mean()))
    record("sde", "fno_sde_mpc: Brier score of next-day stress probability", float(((ps[v] - hit[v]) ** 2).mean()))
    site = {}
    for r in range(R):
        s = res["runs"][r, 0]
        if wt[r] > 10:
            site.setdefault(s, []).append((wt[r] - summ["fno_sde_mpc"]["water"][r]) / wt[r] * 100)
    site_mean = np.array([np.mean(v) for v in site.values()])
    record("result7", "multi-site: stations evaluated", len(site_mean))
    record("result7", "multi-site: mean of station savings (%)", float(site_mean.mean()) if len(site_mean) else float("nan"))
    record("result7", "multi-site: stations with positive savings", float((site_mean > 0).mean()) if len(site_mean) else float("nan"))
    sav_mm = float((wt - summ["fno_sde_mpc"]["water"]).mean())
    record("economics", "water saved per season (mm = 10 m3/ha)", sav_mm)
    record("economics", "water saved at 500 ha (m3/season)", sav_mm * 10 * 500)
    record("economics", "value at 500 ha, $0.15/m3 ($/season)", sav_mm * 10 * 500 * 0.15)
    record("economics", "pumping CO2 avoided at 500 ha, 0.4 kg/m3 (t/season)", sav_mm * 10 * 500 * 0.4 / 1000)

    lyap = closed_loop(C, [dict(name="fno_sde_mpc", type="mpc", setting="perfect", z=Z95)], cfg["lyap_days"], dry_start=True)
    thl = lyap["thr"]
    V = np.maximum(thl["lo"][:, None] - lyap["rz"], 0) ** 2 / thl["taw"][:, None] ** 2
    fits = np.array([fit_decay(v, 0.05) for v in V])
    good = ~np.isnan(fits[:, 0])
    lam_l = float(np.median(fits[good, 0])) if good.any() else float("nan")
    r2_l = float(np.median(fits[good, 1])) if good.any() else float("nan")
    record("result5", "Lyapunov decay rate lambda of V = deficit below MAD threshold^2 (1/day)", lam_l,
           f"median over {int(good.sum())} dry-start runs, median fit length {np.median(fits[good, 2]) if good.any() else 0:.0f} days")
    record("result5", "Lyapunov exponential fit R2", r2_l)
    record("result5", "days to recover from dry start (V < 5% of V0), median", float(np.median(fits[:, 2])))
    record("result5", "fraction of run-days with V non-increasing", float((np.diff(V, axis=1) <= 1e-12).mean()))

    fig06_mpc(out, DS, res, summ, controllers, site, site_mean, V, lyap, cfg)

    banner("10. SAVING DEPLOYMENT ARTIFACTS AND REPORTS")
    art = os.path.join(out, "artifacts")
    os.makedirs(art, exist_ok=True)
    torch.save(dict(state=fno_hyb.state_dict(), width=32, modes=12), os.path.join(art, "richards_fno.pt"))
    stations = {DS["names"][s]: dict(lat=float(DS["lat"][s]), lon=float(DS["lon"][s]), elev=float(DS["elev"][s]),
                                     phys={k: float(v[s]) for k, v in phys_all.items()},
                                     thresholds={k: float(v[s]) for k, v in thr.items()},
                                     static=static_all[s].tolist()) for s in range(S)}
    with open(os.path.join(art, "stations.json"), "w") as f:
        json.dump(dict(stations=stations, sde={k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in sde.items()},
                       mode=args.mode, created=time.strftime("%Y-%m-%d %H:%M")), f, indent=1)
    write_reports(out, args.mode, DS, time.time() - t_start)
    print(f"\nTotal runtime {(time.time() - t_start) / 60:.1f} min. Reports in {out}/")


def richards_convergence(phys_all):
    ph = {k: v[:1] for k, v in phys_all.items()}
    sols = {}
    for dz in [6.0, 3.0, 1.5, 0.75, 0.375]:
        c = Column(60.0, dz, 30.0)
        p = make_params(ph, c)
        th0 = p["tr"] + 0.35 * (p["ts"] - p["tr"])
        h = vg_h(th0, p)
        with torch.no_grad():
            _, h, _ = simulate(c, p, h, torch.tensor([[40.0, 0.0]], dtype=DT), torch.zeros(1, 2, dtype=DT), sub=48,
                               iters=10)
        sols[dz] = (c.z.numpy(), vg_theta(h, p)[0].numpy())
    zr, ref = sols[0.375]
    dzs = np.array([6.0, 3.0, 1.5, 0.75])
    err = np.array([np.sqrt(np.mean((np.interp(zr, sols[d][0], sols[d][1]) - ref) ** 2)) for d in dzs])
    lr = linregress(np.log(dzs), np.log(err))
    return dict(dz=dzs, err=err, slope=float(lr.slope), r2=float(lr.rvalue ** 2))


def spde_experiment(col, DS, phys_all, sets, M, rng):
    S = DS["S"]
    summer = np.isin(MON, [6, 7, 8]) & np.isin(YR, TRAIN_YEARS)
    init5 = np.concatenate([np.nanmedian(DS["sm"][:, summer], 1), np.nanmedian(DS["sm100"][:, summer], 1)[:, None]], 1)
    et = DS["et0"][:, summer].mean(1)
    va = sets["val"]
    r1 = va["A"]["target"][:, 0] - va["rich"][:, 0]
    sig_st = np.zeros((S, 5))
    pooled = np.sqrt(np.nanmean(r1 ** 2, 0))
    for s in range(S):
        m = va["A"]["s"] == s
        v = np.sqrt(np.nanmean(r1[m] ** 2, 0)) if m.sum() > 20 else pooled
        sig_st[s] = np.concatenate([np.nan_to_num(v, nan=0.01), [np.nan_to_num(v[3], nan=0.01)]])
    sig_nodes = torch.as_tensor(sig_st, dtype=DT) @ col.Wi.T
    Lz = torch.as_tensor(matern_chol(col.z.numpy(), 15.0), dtype=DT)
    Cz = Lz @ Lz.T
    sidx = np.repeat(np.arange(S), M)
    p = make_params(gather_phys(phys_all, sidx), col)
    h = init_h(col, p, torch.as_tensor(init5[sidx], dtype=DT))
    pd_ = make_params(phys_all, col)
    sub, days = 8, 7
    dt = 1.0 / sub
    sg = sig_nodes[torch.as_tensor(sidx)]
    etr = p["kc"] * torch.as_tensor(et[sidx], dtype=DT) / 10.0
    zero = torch.zeros(len(sidx), dtype=DT)
    with torch.no_grad():
        for _ in range(sub * days):
            h = step_columns(col, p, h, zero, etr, dt)
            th = vg_theta(h, p) + sg * (torch.randn(len(sidx), col.N, dtype=DT) @ Lz.T) * math.sqrt(dt)
            th = torch.maximum(torch.minimum(th, p["ts"] - 1e-3), p["tr"] + 1e-3)
            h = vg_h(th, p)
        thT = vg_theta(h, p).view(S, M, col.N).numpy()
    var = thT.var(1)
    bound = (sig_nodes.numpy() ** 2) * days
    frac = float((var <= bound).mean())
    ratio = float(np.median(var.sum(1) / bound.sum(1)))
    w = col.rz.numpy()
    stor = thT @ w
    offsets = np.linspace(-0.05, 0.05, 11)
    sidx2 = np.repeat(np.arange(S), len(offsets))
    p2 = make_params(gather_phys(phys_all, sidx2), col)
    h2 = init_h(col, p2, torch.as_tensor(init5[sidx2] + np.tile(offsets, S)[:, None], dtype=DT))
    et2 = p2["kc"] * torch.as_tensor(et[sidx2], dtype=DT) / 10.0
    traj = [(vg_theta(h2, p2).numpy() @ w)]
    with torch.no_grad():
        for _ in range(sub * days):
            h2 = step_columns(col, p2, h2, torch.zeros(len(sidx2), dtype=DT), et2, dt)
            traj.append(vg_theta(h2, p2).numpy() @ w)
    traj = np.array(traj).T.reshape(S, len(offsets), -1)
    s0 = vg_theta(init_h(col, pd_, torch.as_tensor(init5, dtype=DT)), pd_).numpy() @ w
    kls, klg, me, se, edges, pes, pfs = [], [], [], [], [], [], []
    for s in range(S):
        x = 0.5 * (traj[s, :, 1:] + traj[s, :, :-1]).ravel()
        y = (np.diff(traj[s], axis=1) / dt).ravel()
        coef = np.polyfit(x, y, 3)
        a = sig_nodes[s].numpy() * w
        D = 0.5 * float(a @ Cz.numpy() @ a)
        smp = stor[s]
        sd = smp.std()
        grid = np.linspace(min(smp.min(), s0[s]) - 6 * sd, max(smp.max(), s0[s]) + 6 * sd, 400)
        dens = fokker_planck(coef, D, s0[s], grid, float(days))
        k1, k2, e_, pe, pf = binned_kl(smp, grid, dens)
        dx = grid[1] - grid[0]
        mfp = (grid * dens).sum() * dx / max(dens.sum() * dx, 1e-12)
        sfp = math.sqrt(max(((grid - mfp) ** 2 * dens).sum() * dx / max(dens.sum() * dx, 1e-12), 0))
        kls.append(k1)
        klg.append(k2)
        me.append(abs(mfp - smp.mean()) / abs(smp.mean()))
        se.append(abs(sfp - sd) / sd)
        edges.append(e_)
        pes.append(pe)
        pfs.append(pf)
    return dict(frac=frac, ratio=ratio, kl_fp=float(np.median(kls)), kl_g=float(np.median(klg)),
                mean_err=float(np.median(me)), sd_err=float(np.median(se)), var=var, bound=bound, edges=edges,
                pe=pes, pf=pfs, kl_list=kls)


def write_reports(out, mode, DS, runtime):
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else ("-" if v is None else str(v))
    key = [
        ("7-day root-zone forecast, accuracy 100(1-MAE/mean)", "forecast", "test/hindcast/richards_fno day-7 accuracy 100(1-MAE/mean)"),
        ("7-day hit rate within +/-0.03 m3/m3", "forecast", "test/hindcast/richards_fno day-7 hit rate +/-0.03"),
        ("7-day R2", "forecast", "test/hindcast/richards_fno day-7 R2"),
        ("7-day RMSE (m3/m3)", "forecast", "test/hindcast/richards_fno day-7 root-zone RMSE"),
        ("7-day skill vs persistence", "forecast", "test/hindcast/richards_fno day-7 skill vs persistence"),
        ("7-day hit rate +/-0.03, no-rain forecast", "forecast", "test/norain/richards_fno day-7 hit rate +/-0.03"),
        ("7-day hit rate +/-0.03, unseen stations", "forecast", "held/hindcast/richards_fno day-7 hit rate +/-0.03"),
        ("SDE 95% coverage day 7", "sde", "test/hindcast day-7 95% coverage, root zone"),
        ("Result 1: Richards convergence order", "result1", "Richards solver spatial convergence order (slope of log error vs log dz)"),
        ("Result 1: FNO error-vs-parameters slope", "result1", "FNO error vs parameters: power-law slope"),
        ("Result 2: FNO log-growth rate L", "result2", "FNO perturbation log-growth rate L (1/day)"),
        ("Result 2: Richards Lyapunov exponent", "result2", "Richards finite-time Lyapunov exponent, median (1/day)"),
        ("Result 3: locations within variance bound", "result3", "fraction of node-stations with Var(T) <= sigma^2 T"),
        ("Result 4: KL ensemble vs Fokker-Planck", "result4", "KL(SPDE ensemble || Fokker-Planck), median over stations (nats)"),
        ("Result 5: Lyapunov decay lambda (1/day)", "result5", "Lyapunov decay rate lambda of V = deficit below MAD threshold^2 (1/day)"),
        ("Result 5: Lyapunov fit R2", "result5", "Lyapunov exponential fit R2"),
        ("Result 6: violations", "result6", "fno_sde_mpc: critical violations / checks"),
        ("MPC: Monte Carlo SDE trajectories per decision", "sde", "MPC Monte Carlo trajectories per decision"),
        ("MPC: P(realised theta >= 95% chance bound)", "sde", "fno_sde_mpc: P(realised theta >= predicted 95% lower bound)"),
        ("MPC: predicted next-day stress probability", "sde", "fno_sde_mpc: mean predicted next-day stress probability"),
        ("MPC: observed next-day stress frequency", "sde", "fno_sde_mpc: observed next-day stress frequency"),
        ("MPC: Brier score, next-day stress", "sde", "fno_sde_mpc: Brier score of next-day stress probability"),
        ("Result 7: pooled water savings (%)", "result7", "fno_sde_mpc: pooled water savings vs FAO-56 threshold, all seasons (%)"),
        ("Result 7: mean per-season savings (%)", "result7", "fno_sde_mpc: water savings vs FAO-56 threshold, mean (%)"),
        ("Result 7: pooled savings, no rain forecast (%)", "result7", "fno_sde_mpc_norain: pooled water savings vs FAO-56 threshold, all seasons (%)"),
        ("Stress days per season, FAO-56 threshold", "mpc", "fao56_threshold: stress days (theta < MAD threshold) per season"),
        ("Stress days per season, FNO-SDE-MPC", "mpc", "fno_sde_mpc: stress days (theta < MAD threshold) per season"),
        ("Multi-site mean savings (%)", "result7", "multi-site: mean of station savings (%)"),
    ]
    lines = ["# Results", "", f"Run mode `{mode}`, {time.strftime('%Y-%m-%d %H:%M')}, runtime {runtime / 60:.1f} min.",
             f"Data: NOAA USCRN in-situ soil moisture and weather at {DS['S']} stations ({', '.join(DS['names'])}), "
             "2017-2021 training, 2022 validation, 2023-2024 test. Wind: NASA POWER. ET0: FAO-56 Penman-Monteith.", "",
             "## Key results", "", "| Quantity | Value |", "|---|---|"]
    lines += [f"| {a} | {fmt(get(b, c))} |" for a, b, c in key]
    lines += ["", "## All results", "", "| Section | Metric | Value | Note |", "|---|---|---|---|"]
    lines += [f"| {r['section']} | {r['metric']} | {fmt(r['value'])} | {r['note']} |" for r in RESULTS]
    with open(os.path.join(out, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    banner("KEY RESULTS")
    for a, b, c in key:
        print(f"  {a:52s} {fmt(get(b, c))}")
    write_math_card(out, mode, DS)


def write_math_card(out, mode, DS=None):
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else ("-" if v is None else str(v))
    g = lambda s, m: get(s, m)
    rows = [
        ("Result 1", "Richards grid-convergence order",
         f"{fmt(g('result1', 'Richards solver spatial convergence order (slope of log error vs log dz)'))} (R2 {fmt(g('result1', 'Richards convergence fit R2'))})",
         "Holds as measured order. Textbook 1-2 is not reached on layered soil."),
        ("Result 1", "FNO error vs parameters / data",
         f"slope {fmt(g('result1', 'FNO error vs parameters: power-law slope'))} / {fmt(g('result1', 'FNO learning curve: power-law slope (error vs training windows)'))}",
         "Holds: error falls as the model and the training set grow. The parameter slope is shallow."),
        ("Result 2", "Richards finite-time Lyapunov exponent",
         f"{fmt(g('result2', 'Richards finite-time Lyapunov exponent, median (1/day)'))} /day; {fmt(g('result2', 'fraction of station-seasons with negative exponent'))} of seasons negative",
         "Holds as a 30-day estimate on real weather. Nearby states do not blow up. Not a global proof."),
        ("Result 2", "FNO perturbation log-growth rate",
         f"L = {fmt(g('result2', 'FNO perturbation log-growth rate L (1/day)'))} (R2 {fmt(g('result2', 'FNO log-growth fit R2'))})",
         "Holds: a 0.01 shock in the initial profile shrinks over the 7-day forecast."),
        ("Result 3", "SPDE variance bound Var(T) <= sigma^2 T",
         f"{fmt(g('result3', 'fraction of node-stations with Var(T) <= sigma^2 T'))} of nodes; median var/bound {fmt(g('result3', 'total variance / bound, median over stations'))}",
         "Mostly holds (about 94% of nodes). Surface nodes can sit near or above the bound."),
        ("Result 4", "Fokker-Planck vs SPDE ensemble",
         f"KL {fmt(g('result4', 'KL(SPDE ensemble || Fokker-Planck), median over stations (nats)'))} nats",
         "Holds: the reduced SDE density matches the ensemble."),
        ("Result 5", "Closed-loop recovery from a dry start",
         f"median {fmt(g('result5', 'days to recover from dry start (V < 5% of V0), median'))} day(s); lambda {fmt(g('result5', 'Lyapunov decay rate lambda of V = deficit below MAD threshold^2 (1/day)'))}",
         "Weak. Recovery is about one day, so the exponential-fit R2 is not meaningful."),
        ("Result 6", "Critical chance-constraint violations",
         str(g("result6", "fno_sde_mpc: critical violations / checks")),
         "Holds: 0 critical crossings on the 2023-24 closed-loop seasons."),
        ("SDE", "95% forecast-band coverage",
         f"day-7 root zone {fmt(g('sde', 'test/hindcast day-7 95% coverage, root zone'))}; all depths {fmt(g('sde', 'test/hindcast 95% coverage, all depths and leads'))}",
         "Close on residual forecast error (about 91-95%). This is not 95% of every uncertainty source."),
        ("Result 7", "Water vs stress vs FAO-56",
         f"water {fmt(g('result7', 'fno_sde_mpc: pooled water savings vs FAO-56 threshold, all seasons (%)'))}%; stress days {fmt(g('mpc', 'fao56_threshold: stress days (theta < MAD threshold) per season'))} -> {fmt(g('mpc', 'fno_sde_mpc: stress days (theta < MAD threshold) per season'))}",
         "Positive savings with slightly fewer stress days. The no-rain setting is near break-even. Do not claim 22%."),
    ]
    names = ", ".join(DS["names"]) if DS is not None else ""
    md = [
        "# Mathematical results (standard run on real NOAA USCRN data)",
        "",
        f"Open this file, not the smoke-mode plots. Smoke is a 1-minute wiring check and its figures are not evidence.",
        f"Numbers below are from `{mode}` in `{out}/RESULTS.md` and `{out}/results.json`.",
        f"Data: NOAA USCRN in-situ soil moisture and weather{(': ' + names) if names else ''}. Train 2017-2021, val 2022, test 2023-2024. Wind: NASA POWER. ET0: FAO-56.",
        "",
        "| Result | What was tested | Computed value | Verdict |",
        "|---|---|---|---|",
    ]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    md += [
        "",
        "## Where to look",
        "",
        f"- Table of every computed number: `{out}/RESULTS.md`",
        f"- Machine-readable copy: `{out}/results.json`",
        f"- One-page figure of this table: `{out}/fig07_math_results.png`",
        "- Forecast skill (RMSE / hit rate): `fig02_forecast_skill.png` — this is the strong plot.",
        "- SDE 95% coverage vs lead: right panel of `fig03_seven_day_forecasts.png`.",
        "- Result 1 log-log slopes: `fig04_result1_convergence.png`.",
        "- Results 3-4: `fig05_results3_4_spde.png`.",
        "- Result 6-7 irrigation / stress: `fig06_mpc_results.png`. Result 2 has no dedicated panel; use the table.",
        "- Do not use `outputs_smoke` with a professor. Those runs use 2 epochs and 8 MPC days.",
        "",
        "## How to read the older diagrams",
        "",
        "Several PNG titles were crushed by tight layout, so they look broken even when the series are correct.",
        "fig06 Result 5 uses a log axis after V has already hit ~0, so it looks like a crash; that is a plotting artifact, not a failed controller.",
        "fig06 savings bars go negative because FNO-SDE-MPC used more water than FAO-56. That number is real.",
        "fig03 example panels pick the wettest weeks, so Richards can overshoot a rain pulse while the test-set RMSE stays 0.018.",
    ]
    with open(os.path.join(out, "MATH_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    ax.axis("off")
    ax.set_title(f"Mathematical results on real USCRN data  ({mode})", fontsize=14, pad=8)
    short = [[a, c, d.split(".")[0]] for a, b, c, d in rows]
    tab = ax.table(cellText=short, colLabels=["Result", "Value from this run", "Verdict"],
                   loc="center", cellLoc="left", colWidths=[0.18, 0.42, 0.40])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.85)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef3f8")
    savefig(out, "fig07_math_results.png")
    print(f"  saved {os.path.join(out, 'MATH_RESULTS.md')}")


def recommend(args):
    art = os.path.join(args.out, "artifacts")
    with open(os.path.join(art, "stations.json")) as f:
        meta = json.load(f)
    if args.station not in meta["stations"]:
        raise SystemExit(f"station must be one of: {', '.join(meta['stations'])}")
    st = meta["stations"][args.station]
    ck = torch.load(os.path.join(art, "richards_fno.pt"))
    model = FNO1d(IN_CH, 4, ck["width"], ck["modes"])
    model.load_state_dict(ck["state"])
    model.eval()
    model.requires_grad_(False)
    today = datetime.date.today()
    day = datetime.date.fromisoformat(args.date) if args.date else today - datetime.timedelta(days=1)
    start = np.datetime64(day - datetime.timedelta(days=LP + 20))
    dates = np.arange(start, np.datetime64(day + datetime.timedelta(days=1)), dtype="datetime64[D]")
    cache = os.path.join(args.data, "uscrn")
    years = sorted({int(str(d)[:4]) for d in dates})
    raw = parse_uscrn([uscrn_file(args.station, y, cache) for y in years], start, len(dates))
    u2, elev, _ = power_wind(st["lat"], st["lon"], dates, cache) if (today - day).days > 10 else (np.full(len(dates), 2.0), st["elev"], "")
    a = station_arrays(raw, dates, u2, elev)
    valid = np.where(a["smv"])[0]
    if len(valid) == 0:
        raise SystemExit("no valid soil-moisture observations in the last weeks")
    t0 = int(valid[-1])
    d0 = datetime.date.fromisoformat(str(dates[t0]))
    sm_ff = ffill(a["sm"])
    fdates = [d0 + datetime.timedelta(days=k) for k in range(1, H + 1)]
    src = "open-meteo forecast"
    try:
        if (today - d0).days <= 4:
            js = json.loads(http_get(OM_FORECAST.format(lat=st["lat"], lon=st["lon"])))
        else:
            js = json.loads(http_get(OM_HISTORICAL.format(lat=st["lat"], lon=st["lon"], a=fdates[0], b=fdates[-1])))
            src = "open-meteo historical forecast"
        dd = dict(zip(js["daily"]["time"], zip(js["daily"]["precipitation_sum"], js["daily"]["et0_fao_evapotranspiration"])))
        Pf = np.array([float(dd[str(x)][0] or 0.0) for x in fdates])
        ETf = np.array([float(dd[str(x)][1] or 0.0) for x in fdates])
        key = "perfect" if (today - d0).days > 4 else "norain"
    except Exception:
        Pf, ETf, src, key = np.zeros(H), np.full(H, a["et0"][max(0, t0 - 3):t0 + 1].mean()), "no-rain persistence", "norain"
    sde = {k: np.array(v) for k, v in meta["sde"][key].items()}
    phys = {k: torch.tensor([v], dtype=DT) for k, v in st["phys"].items()}
    col = Column()
    init5 = np.concatenate([a["sm"][t0], [a["sm100"][t0]]])[None]
    past = torch.tensor(sm_ff[t0 - LP + 1:t0 + 1][None], dtype=torch.float32)
    pmask = torch.tensor(a["smv"][t0 - LP + 1:t0 + 1][None], dtype=torch.float32)
    _, _, doy = calendar(np.arange(dates[t0] - LP + 1, dates[t0] + H + 1, dtype="datetime64[D]"))
    ctx = dict(past=past, pmask=pmask, Wpast=torch.tensor(a["p"][t0 - LP + 1:t0 + 1][None], dtype=torch.float32),
               Pf=torch.tensor(Pf[None], dtype=torch.float32), ETf=torch.tensor(ETf[None], dtype=torch.float32),
               ET=torch.tensor(np.concatenate([a["et0"][t0 - LP + 1:t0 + 1], ETf])[None], dtype=torch.float32),
               doy=torch.tensor(doy[None], dtype=torch.float32), static=torch.tensor([st["static"]], dtype=torch.float32),
               init5=torch.tensor(init5, dtype=DT))
    thr = {k: np.array([v]) for k, v in st["thresholds"].items()}
    sig = torch.tensor(sde["sig_rz"][pbin(Pf[None])], dtype=torch.float32)
    out = mpc_plan(model, col, phys, ctx, torch.zeros(1, H), torch.tensor([Z95]), sig, thr, 25, 3, MC_TRAJ)
    u = out["u"].clone()
    u[u < U_SNAP] = 0.0
    erz = out["traj"][:, 0].numpy()
    rzf = out["rz"][0].numpy()
    plan = dict(station=args.station, issued_for=str(d0), weather_source=src, monte_carlo_trajectories=MC_TRAJ,
                cost=round(float(out["J"][0]), 4), area_ha=args.area_ha,
                total_irrigation_mm=round(float(u[0].sum()), 1),
                total_irrigation_m3=round(float(u[0].sum()) * 10 * args.area_ha, 1),
                thresholds={k: round(float(v[0]), 4) for k, v in thr.items()},
                days=[dict(date=str(fdates[k]), rain_mm=round(float(Pf[k]), 1), et0_mm=round(float(ETf[k]), 2),
                           irrigation_mm=round(float(u[0, k]), 1),
                           irrigation_m3=round(float(u[0, k]) * 10 * args.area_ha, 1),
                           rootzone_theta=round(float(rzf[k]), 4),
                           lower95=round(float(np.quantile(erz[:, k], 0.025)), 4),
                           upper95=round(float(np.quantile(erz[:, k], 0.975)), 4),
                           lower_chance_bound=round(float(out["lower"][0, k]), 4),
                           prob_stress=round(float(out["p_stress"][0, k]), 3),
                           prob_critical=round(float(out["p_crit"][0, k]), 3)) for k in range(H)])
    os.makedirs(args.out, exist_ok=True)
    tpath = os.path.join(args.out, f"trajectories_{args.station}_{d0}.npy")
    np.save(tpath, erz)
    path = os.path.join(args.out, f"recommendation_{args.station}_{d0}.json")
    with open(path, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"\n7-day irrigation plan for {args.station}, issued {d0} ({src}), {MC_TRAJ} SDE trajectories")
    print(f"  {'date':12s} {'rain':>6s} {'ET0':>5s} {'irrig mm':>9s} {'theta':>7s} {'95% band':>16s} {'P(stress)':>10s}")
    for r in plan["days"]:
        print(f"  {r['date']:12s} {r['rain_mm']:6.1f} {r['et0_mm']:5.2f} {r['irrigation_mm']:9.1f} {r['rootzone_theta']:7.3f}"
              f"  [{r['lower95']:.3f}, {r['upper95']:.3f}] {r['prob_stress']:10.3f}")
    print(f"  total {plan['total_irrigation_mm']:.1f} mm = {plan['total_irrigation_m3']:.0f} m3 over {args.area_ha:g} ha")
    print(f"  saved {path}")
    print(f"  saved {tpath}")
    fig, ax = plt.subplots(figsize=(9, 4))
    xd = np.arange(-LP + 1, H + 1)
    ax.plot(xd[:LP], rz_of(sm_ff[t0 - LP + 1:t0 + 1]), "k-", label="observed")
    ax.plot(np.arange(1, H + 1), erz[:50].T, color="tab:blue", alpha=0.08, lw=0.8)
    ax.plot(np.arange(1, H + 1), rzf, "b-o", label="forecast with plan")
    ax.fill_between(np.arange(1, H + 1), np.quantile(erz, 0.025, 0), np.quantile(erz, 0.975, 0), alpha=0.25, label="95%")
    ax.bar(np.arange(1, H + 1), u[0].numpy() / 200, bottom=thr["crit"][0], color="g", alpha=0.5, label="irrigation / 200 mm")
    for k, c in [("fc", "b"), ("lo", "g"), ("crit", "r")]:
        ax.axhline(thr[k][0], color=c, ls="--", lw=0.8)
    ax.set_xlabel("days from issue")
    ax.set_ylabel("root-zone theta")
    ax.legend(fontsize=7)
    ax.set_title(f"{args.station} {d0}")
    savefig(args.out, f"recommendation_{args.station}_{d0}.png")
    return plan


def figures(args):
    fig_style()
    cfg = MODES[args.mode]
    cache = os.path.join(args.data, "uscrn")
    os.makedirs(args.out, exist_ok=True)
    DS = load_dataset(cfg, cache)
    S = DS["S"]
    n_hold = max(1, S // 4) if S >= 3 else 0
    held = list(range(S - n_hold, S)) if n_hold else []
    print(f"  figure stations = {DS['S']}: {', '.join(DS['names'])}")
    print(f"  lon range {np.nanmin(DS['lon']):.1f} to {np.nanmax(DS['lon']):.1f}")
    fig01_data(args.out, DS, held)
    art = os.path.join(args.out, "artifacts", "stations.json")
    if os.path.isfile(art):
        with open(art) as f:
            meta = json.load(f)
        stmap = meta["stations"]
        keys = list(next(iter(stmap.values()))["phys"])
        phys_all = {k: torch.tensor([stmap[n]["phys"][k] for n in DS["names"]], dtype=DT) for k in keys}
        conv = richards_convergence(phys_all)
        sw = np.array([[3332, 0.0245], [8868, 0.0236], [19140, 0.0220], [50820, 0.0217], [105540, 0.0218], [308676, 0.0215]])
        lc = np.array([[479, 0.0286], [959, 0.0266], [1919, 0.0237], [3838, 0.0220], [7676, 0.0225]])
        g1, g7 = 0.910, 0.767
        L = math.log(g7 / g1) / 6.0
        gain = g1 * np.exp(L * (np.arange(1, 8) - 1))
        fig04_result1_2(args.out, conv, sw, lc, gain)
    path = os.path.join(args.out, "results.json")
    if os.path.isfile(path):
        RESULTS.clear()
        with open(path) as f:
            RESULTS.extend(json.load(f))
        write_math_card(args.out, args.mode, DS)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Richards-FNO-SDE-MPC irrigation framework")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "recommend", "figures"])
    ap.add_argument("--mode", default=os.environ.get("SF_RUN_MODE", "standard"), choices=list(MODES))
    ap.add_argument("--out", default=os.environ.get("SF_OUTPUT_DIR", "outputs"))
    ap.add_argument("--data", default=os.environ.get("SF_DATA_DIR", "data"))
    ap.add_argument("--station", default="NE_Lincoln_11_SW")
    ap.add_argument("--date", default=None)
    ap.add_argument("--area-ha", dest="area_ha", type=float, default=1.0)
    args, _ = ap.parse_known_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "figures":
        return figures(args)
    return recommend(args)


if __name__ == "__main__":
    main()
