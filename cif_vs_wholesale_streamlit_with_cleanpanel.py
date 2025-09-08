# -*- coding: utf-8 -*-
# Streamlit — CIF vs Wholesale: Cleaning/Panel + Validation + Modeling (R Shiny parity)

import os
import io
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order as vecm_select_order
from statsmodels.tsa.stattools import adfuller, kpss, coint, grangercausalitytests
from pathlib import Path


BASE_DIR = Path(r"C:\Users\KimByeolha\OneDrive - 트릿지\예측지계산_20250901")

# 순서대로 탐색할 폴더들
DEFAULT_DIRS = [
    BASE_DIR,
    Path.cwd(),                         # 현재 실행 폴더
    (Path(__file__).parent if "__file__" in globals() else Path.cwd()),
    ((Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "samples"),
  # 샘플 폴더
    Path.cwd() / "data",                # ./data
    Path.home() / "Downloads",          # 다운로드
    Path(r"/mnt/data"),                 # (리눅스/클라우드)
]

def load_default_by_stem(stem_or_filename: str) -> Optional[pd.DataFrame]:
    """
    stem_or_filename이 확장자를 안 가지면 .csv/.xlsx/.xls를 순서대로 시도.
    DEFAULT_DIRS를 앞에서부터 탐색해서 처음 찾는 파일을 read_any로 읽음.
    """
    # 이미 확장자가 붙어 있으면 그대로 시도
    if Path(stem_or_filename).suffix:
        candidates = [stem_or_filename]
    else:
        candidates = [stem_or_filename + ext for ext in (".csv", ".xlsx", ".xls")]

    for d in DEFAULT_DIRS:
        for name in candidates:
            p = d / name
            if p.exists():
                try:
                    return read_any(str(p))
                except Exception:
                    pass
    return None


# Optional Prophet
try:
    from prophet import Prophet  # type: ignore
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# ---------------------------
# Global Plot Style
# ---------------------------
DPI = 160
FIGSIZE = (11, 5)

def pretty_ax(ax, title=None, ylabel=None, rotate=30):
    ax.grid(True, linewidth=0.6, alpha=0.3)
    if title: ax.set_title(title, fontsize=13, pad=6)
    if ylabel: ax.set_ylabel(ylabel)

    # 날짜 축: 자동 간격 + 간결 포맷(연·월 중복 제거)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 라벨 회전으로 겹침 방지
    ax.tick_params(axis='x', labelrotation=rotate, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # 여백 자동 조정
    ax.get_figure().tight_layout()
    return ax


# ---------------------------
# Utilities
# ---------------------------
def month_floor(s):
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

_EXCEL_ORIGIN = pd.Timestamp("1899-12-30")

def parse_dt_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return month_floor(s)
    if pd.api.types.is_numeric_dtype(s):
        dt = _EXCEL_ORIGIN + pd.to_timedelta(s.astype(float), unit="D")
        return month_floor(dt)
    s = s.astype(str).str.replace(r"[./]", "-", regex=True)
    return month_floor(pd.to_datetime(s, errors="coerce"))

def numify_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(r"[^0-9.,-]", "", regex=True)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def read_any(path_or_buffer, name: Optional[str] = None) -> Optional[pd.DataFrame]:
    if path_or_buffer is None:
        return None
    if hasattr(path_or_buffer, "read"):  # UploadedFile
        data = path_or_buffer.read()
        bio = io.BytesIO(data)
        ext = (os.path.splitext(name or "")[1] or "").lower()
        if ext in (".xlsx", ".xls"):
            try:
                return pd.read_excel(bio)
            except Exception:
                bio.seek(0)
                return pd.read_excel(bio, engine="openpyxl")
        for enc in ("utf-8", "cp949", "latin1"):
            try:
                bio.seek(0)
                return pd.read_csv(bio, encoding=enc)
            except Exception:
                continue
        bio.seek(0)
        return pd.read_csv(bio, encoding_errors="ignore")
    else:
        path = str(path_or_buffer)
        ext = (os.path.splitext(name or path)[1] or "").lower()
        if ext in (".xlsx", ".xls"):
            try:
                return pd.read_excel(path)
            except Exception:
                return pd.read_excel(path, engine="openpyxl")
        for enc in ("utf-8", "cp949", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path, encoding_errors="ignore")

# ---------------------------
# Column auto-detect helpers
# ---------------------------
def _find_col(df, candidates):
    """Case/sep-insensitive column finder."""
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())
    nmap = {norm(c): c for c in cols}
    for cand in candidates:
        key = norm(cand)
        if key in nmap:
            return nmap[key]
    return None

# ---------------------------
# Cleaning (robust mapping)
# ---------------------------
def clean_cif_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or len(df_raw) == 0:
        raise ValueError("CIF: 빈 데이터")
    pc = _find_col(df_raw, ["PRODUCT_CODE","product_code","product","hs","hs_code"])
    dt = _find_col(df_raw, ["DATE","date","ym","yearmonth","ymd","month"])
    ex = _find_col(df_raw, ["EXPORT_COUNTRY_CODE","export_country_code","export_iso2","export","iso_export"])
    im = _find_col(df_raw, ["IMPORT_COUNTRY_CODE","import_country_code","import_iso2","import","iso_import"])
    up = _find_col(df_raw, ["UNIT_PRICE_USD","unit_price_usd","unitpriceusd","price_usd","usd_per_kg","cif_usd","price"])
    miss = [k for k,v in {"PRODUCT_CODE":pc,"DATE":dt,"EXPORT_COUNTRY_CODE":ex,"IMPORT_COUNTRY_CODE":im,"UNIT_PRICE_USD":up}.items() if v is None]
    if miss:
        raise ValueError(f"CIF 필수 컬럼을 찾을 수 없음: {', '.join(miss)} | 현재 컬럼: {list(df_raw.columns)}")

    df = df_raw.rename(columns={pc:"PRODUCT_CODE", dt:"DATE", ex:"EXPORT_COUNTRY_CODE", im:"IMPORT_COUNTRY_CODE", up:"UNIT_PRICE_USD"}).copy()

    df["PRODUCT_CODE"] = df["PRODUCT_CODE"].astype(str)
    if pd.api.types.is_numeric_dtype(df["DATE"]):
        df["DATE"] = _EXCEL_ORIGIN + pd.to_timedelta(df["DATE"].astype(float), unit="D")
    df["DATE"] = parse_dt_series(df["DATE"])
    df["EXPORT_COUNTRY_CODE"] = df["EXPORT_COUNTRY_CODE"].astype(str).str.upper()
    df["IMPORT_COUNTRY_CODE"] = df["IMPORT_COUNTRY_CODE"].astype(str).str.upper()
    df["UNIT_PRICE_USD"] = numify_series(df["UNIT_PRICE_USD"])

    df = df[(df["DATE"].notna()) & (df["UNIT_PRICE_USD"].notna()) & (df["UNIT_PRICE_USD"] > 0)]
    df = df[(df["PRODUCT_CODE"].str.len() > 0) & (df["EXPORT_COUNTRY_CODE"].str.len() > 0) & (df["IMPORT_COUNTRY_CODE"].str.len() > 0)]

    grp = (df.groupby(["PRODUCT_CODE","EXPORT_COUNTRY_CODE","IMPORT_COUNTRY_CODE","DATE"], as_index=False)["UNIT_PRICE_USD"].median())
    grp = grp.rename(columns={
        "PRODUCT_CODE":"product_code",
        "EXPORT_COUNTRY_CODE":"export_iso2",
        "IMPORT_COUNTRY_CODE":"import_iso2",
        "DATE":"date",
        "UNIT_PRICE_USD":"unit_price",
    }).sort_values(["product_code","export_iso2","import_iso2","date"]).reset_index(drop=True)

    grp["ln_unit_price"]  = np.log(grp["unit_price"])
    grp["dln_unit_price"] = grp.groupby(["product_code","export_iso2","import_iso2"])["ln_unit_price"].diff()
    return grp

def clean_wholesale_v2(df_raw: pd.DataFrame, aggregate_monthly: bool = True) -> pd.DataFrame:
    if df_raw is None or len(df_raw) == 0:
        raise ValueError("Wholesale: 빈 데이터")
    pc = _find_col(df_raw, ["PRODUCT_CODE","product_code","product","hs","hs_code"])
    dt = _find_col(df_raw, ["DATE","date","ym","yearmonth","ymd","month"])
    cc = _find_col(df_raw, ["COUNTRY_CODE","country_code","country_iso2","iso2","iso_code","country"])
    pr = _find_col(df_raw, ["UNIT_PRICE_AVG_USD","unit_price_avg_usd","price_usd_per_kg","avg_price_usd",
                            "unit_price_usd","price_usd","usd_per_kg","wh_usd","price"])
    miss = [k for k,v in {"PRODUCT_CODE":pc,"DATE":dt,"COUNTRY_CODE":cc,"UNIT_PRICE_AVG_USD":pr}.items() if v is None]
    if miss:
        raise ValueError(f"Wholesale 필수 컬럼을 찾을 수 없음: {', '.join(miss)} | 현재 컬럼: {list(df_raw.columns)}")

    df = df_raw.rename(columns={pc:"PRODUCT_CODE", dt:"DATE", cc:"COUNTRY_CODE", pr:"UNIT_PRICE_AVG_USD"}).copy()

    df["PRODUCT_CODE"] = df["PRODUCT_CODE"].astype(str)
    if pd.api.types.is_numeric_dtype(df["DATE"]):
        df["DATE"] = _EXCEL_ORIGIN + pd.to_timedelta(df["DATE"].astype(float), unit="D")
    df["DATE"] = parse_dt_series(df["DATE"])
    df["COUNTRY_CODE"] = df["COUNTRY_CODE"].astype(str).str.upper()
    df["UNIT_PRICE_AVG_USD"] = numify_series(df["UNIT_PRICE_AVG_USD"])

    df = df[(df["DATE"].notna()) & (df["UNIT_PRICE_AVG_USD"].notna()) & (df["UNIT_PRICE_AVG_USD"] > 0)]
    df = df[(df["PRODUCT_CODE"].str.len() > 0) & (df["COUNTRY_CODE"].str.len() > 0)]

    if aggregate_monthly:
        df = df.groupby(["PRODUCT_CODE","COUNTRY_CODE","DATE"], as_index=False)["UNIT_PRICE_AVG_USD"].mean()

    out = df.rename(columns={
        "PRODUCT_CODE":"product_code",
        "COUNTRY_CODE":"country_iso2",
        "DATE":"date",
        "UNIT_PRICE_AVG_USD":"price_usd_per_kg",
    }).sort_values(["product_code","country_iso2","date"]).reset_index(drop=True)

    out["ln_price"]  = np.log(out["price_usd_per_kg"])
    out["dln_price"] = out.groupby(["product_code","country_iso2"])["ln_price"].diff()
    return out

# ---------------------------
# Pair builder (user-chosen)
# ---------------------------
def build_pair_panel(
    cif_m: pd.DataFrame,
    wh_m: pd.DataFrame,
    key_side: str,           # 'export' or 'import'
    cif_product: str,
    cif_iso: str,            # export_iso2 or import_iso2 (key_side에 따라)
    wh_product: str,
    wh_iso: str,             # country_iso2
    join: str = "inner"      # 'inner' | 'left_keep_CIF' | 'right_keep_Wholesale'
) -> pd.DataFrame:

    join_map = {"inner": "inner", "left_keep_CIF": "left", "right_keep_Wholesale": "right"}
    how = join_map.get(join, "inner")

    # CIF filter
    if key_side == "export":
        cif_sel = cif_m[(cif_m["product_code"] == cif_product) & (cif_m["export_iso2"] == cif_iso)]
    else:
        cif_sel = cif_m[(cif_m["product_code"] == cif_product) & (cif_m["import_iso2"] == cif_iso)]
    cif_sel = cif_sel[["date","unit_price"]].rename(columns={"unit_price":"cif"})

    # WH filter
    wh_sel = wh_m[(wh_m["product_code"] == wh_product) & (wh_m["country_iso2"] == wh_iso)]
    wh_sel = wh_sel[["date","price_usd_per_kg"]].rename(columns={"price_usd_per_kg":"wh"})

    # Merge by date only (의도적으로 제품/국가 cross 허용)
    m = pd.merge(cif_sel, wh_sel, on="date", how=how)
    # log-safe rows only
    m = m[(m["cif"].notna()) & (m["wh"].notna()) & (m["cif"] > 0) & (m["wh"] > 0)]

    # De-dup by date (차분 0 남발 방지)
    m = m.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    # Meta
    m["pair_cif"] = f"{cif_product}:{key_side.upper()}-{cif_iso}"
    m["pair_wh"]  = f"{wh_product}:WH-{wh_iso}"

    # logs & diffs
    m["ln_cif"]  = np.log(m["cif"])
    m["ln_wh"]   = np.log(m["wh"])
    m["dln_cif"] = m["ln_cif"].diff()
    m["dln_wh"]  = m["ln_wh"].diff()
    return m

# ---------------------------
# Tests & Reports
# ---------------------------
def adf_report(x: pd.Series, name: str, regression: str = "c") -> str:
    x = x.dropna().astype(float)
    if len(x) < 10:
        return f"{name}: 데이터 부족 (n={len(x)})"
    stat, pval, _, _, crit, _ = adfuller(x, regression=regression, autolag="AIC")
    cv5 = crit.get("5%")
    dec = "정상(기각)" if stat < cv5 else "비정상(기각 실패)"
    return f"ADF[{regression}] stat={stat:.3f}, cv5={cv5:.3f}, p={pval:.3f} → {dec}"

def kpss_report(x: pd.Series, name: str, regression: str = "c") -> str:
    x = x.dropna().astype(float)
    if len(x) < 10:
        return f"{name}: 데이터 부족 (n={len(x)})"
    try:
        stat, pval, _, crit = kpss(x, regression=regression, nlags="auto")
        cv5 = float(crit.get("5%"))
        dec = "비정상(기각)" if stat > cv5 else "정상(기각 실패)"
        return f"KPSS[{regression}] stat={stat:.3f}, cv5={cv5:.3f}, p≈{pval:.3f} → {dec}"
    except Exception as e:
        return f"KPSS 오류: {e}"

def eg_coint_report(x: pd.Series, y: pd.Series, namex: str, namey: str) -> str:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 30:
        return f"Engle–Granger: 표본 부족 n={len(df)} (≥30 권장)"
    tstat, pval, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    dec = "공적분 있음(귀무: 없음 기각)" if pval < 0.05 else "공적분 없음(기각 실패)"
    return f"EG {namex}~{namey}: t={tstat:.3f}, p={pval:.3f} → {dec}"

def select_order(X, maxlags=12, deterministic="co"):
    try:
        return vecm_select_order(X, maxlags=maxlags, deterministic=deterministic)
    except Exception:
        class Dummy: k_ar_diff = 1
        return Dummy()

def johansen_rank(X, det_order=0):
    res = coint_johansen(X, det_order, k_ar_diff=1)
    return 1 if (res.lr1[0] > res.cvt[0, 1]) else 0  # 5%

def johansen_report(df2: pd.DataFrame) -> str:
    X = df2[["ln_cif", "ln_wh"]].dropna()
    if len(X) < 24:
        return f"Johansen: 표본 부족 n={len(X)}"
    try:
        order = select_order(X, maxlags=min(12, max(2, len(X)//10)), deterministic="co")
        k = int(getattr(order, "k_ar_diff", 1))
        res = coint_johansen(X, det_order=0, k_ar_diff=max(1, k))
        r0 = res.lr1[0] > res.cvt[0, 1]
        return f"Johansen: r={'≥1' if r0 else '0'} (trace @5%)"
    except Exception as e:
        return f"Johansen 오류: {e}"

def granger_table_dlog(df_xy: pd.DataFrame, maxlag: int = 6) -> pd.DataFrame:
    d = df_xy[["dln_cif", "dln_wh"]].dropna()
    if len(d) < maxlag + 5:
        return pd.DataFrame({"msg": [f"표본 부족 n={len(d)}"]})
    out = []
    res1 = grangercausalitytests(d[["dln_cif", "dln_wh"]], maxlag=maxlag, verbose=False)  # wh→cif
    res2 = grangercausalitytests(d[["dln_wh", "dln_cif"]], maxlag=maxlag, verbose=False)  # cif→wh
    for L in range(1, maxlag + 1):
        p1 = res1[L][0]["ssr_ftest"][1]
        p2 = res2[L][0]["ssr_ftest"][1]
        out.append({"lag": L, "H0: wh ↛ cif (p)": p1, "H0: cif ↛ wh (p)": p2})
    return pd.DataFrame(out)

# ---------------------------
# Modeling helpers
# ---------------------------
def rmse(a, b): a = np.asarray(a); b = np.asarray(b); return float(np.sqrt(np.nanmean((a - b) ** 2)))
def mape(a, b): a = np.asarray(a); b = np.asarray(b); return float(np.nanmean(np.abs((a - b) / a)) * 100)
def hit_ratio(y_true_log, y_hat_log):
    dy_true = np.diff(np.asarray(y_true_log)); dy_hat = np.diff(np.asarray(y_hat_log))
    ok = np.isfinite(dy_true) & np.isfinite(dy_hat)
    if ok.sum() == 0: return np.nan
    return float((np.sign(dy_true[ok]) == np.sign(dy_hat[ok])).mean() * 100)

def ols_with_aic(df, L):
    d = df.copy()
    d["wh_lag"] = d["ln_wh"].shift(L)
    d = d.dropna()
    X = sm.add_constant(d[["wh_lag"]])
    y = d["ln_cif"]
    model = sm.OLS(y, X).fit()
    return model, len(d), model.aic

def pick_vecm(df, nmax=12):
    X = df[["ln_cif", "ln_wh"]].dropna()
    if len(X) < 24:
        return None
    try:
        order = select_order(X, maxlags=nmax, deterministic="co")
        k = int(getattr(order, "k_ar_diff", 1))
        rank = johansen_rank(X, det_order=0)
        if rank >= 1:
            vecm = VECM(X, k_ar_diff=max(1, k), coint_rank=1, deterministic="co")
            res = vecm.fit()
            return {"res": res, "k": max(1, k), "rank": 1}
        return None
    except Exception:
        return None

def make_ect(df, vecm_res=None):
    X = df[["ln_cif", "ln_wh"]].dropna()
    beta = None
    if vecm_res is not None:
        try:
            beta = np.asarray(vecm_res.beta[:, [0]])
        except Exception:
            beta = None
    if beta is None:
        eg = sm.OLS(X["ln_cif"], sm.add_constant(X["ln_wh"])).fit()
        b1 = eg.params["ln_wh"]; beta = np.array([[1.0], [-b1]])
    M = np.column_stack([df["ln_cif"].values, df["ln_wh"].values])
    ect_raw = M @ beta
    mu = np.nanmean(ect_raw)
    ect = ect_raw - mu
    return ect.ravel(), beta, float(mu)

def design_matrix_for_ecm(df, L, include_ect=True):
    dln_wh = np.r_[np.nan, np.diff(df["ln_wh"].values)]
    cols = {f"dln_wh_L{lag}": pd.Series(dln_wh).shift(lag).values for lag in range(L + 1)}
    X = pd.DataFrame(cols, index=df.index)
    if include_ect and "ect" in df.columns:
        X["ect_l1"] = pd.Series(df["ect"].values).shift(1).values
    y = np.r_[np.nan, np.diff(df["ln_cif"].values)]
    D = pd.concat([pd.Series(y, index=df.index, name="dln_cif"), X], axis=1).dropna()
    return D

def fit_arima_grid(y, X=None, seasonal=False, max_p=3, max_d=1, max_q=3):
    best = None; records = []
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(y, order=(p, d, q), exog=X, enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit()
                    llf = res.llf; nobs = res.nobs; k_params = res.params.shape[0]
                    aic = -2 * llf + 2 * k_params
                    aicc = aic + (2 * k_params * (k_params + 1)) / max(nobs - k_params - 1, 1)
                    bic = res.bic
                    records.append({"p": p, "d": d, "q": q, "AICc": round(aicc, 2), "BIC": round(bic, 2), "n": int(nobs)})
                    if (best is None) or (aicc < best["AICc"]):
                        best = {"res": res, "order": (p, d, q), "AICc": aicc, "BIC": bic}
                except Exception:
                    continue
    tab = pd.DataFrame(records) if records else pd.DataFrame(columns=["p","d","q","AICc","BIC","n"])
    return best, tab.sort_values("AICc") if not tab.empty else tab

def align_xreg(fitted_res, Xnew):
    try:
        xr = fitted_res.model.exog_names
        xr = [c for c in xr if c != "const"] if xr else None
    except Exception:
        xr = None
    if xr is None or Xnew is None: return None
    X = Xnew.copy()
    for c in xr:
        if c not in X.columns: X[c] = 0.0
    return X[xr].values

def forecast_lnwh(df, h, vecm_res=None, method="vec"):
    last_ln_cif = df["ln_cif"].iloc[-1]; last_ln_wh = df["ln_wh"].iloc[-1]
    if method == "vec" and vecm_res is not None:
        try:
            fc = vecm_res.predict(steps=h)
            names = list(vecm_res.names)
            i_wh = names.index("ln_wh") if "ln_wh" in names else 1
            i_cif = names.index("ln_cif") if "ln_cif" in names else 0
            return fc[:, i_wh], fc[:, i_cif]
        except Exception:
            pass
    try:
        best, _ = fit_arima_grid(df["ln_wh"].values, X=None, seasonal=False)
        if best is not None: 
            pred = best["res"].get_forecast(steps=h)
            ln_wh_future = pred.predicted_mean.values
        else:
            ln_wh_future = np.repeat(df["ln_wh"].values[-1], h)

        return ln_wh_future, np.repeat(last_ln_cif, h)
    except Exception:
        ln_wh_future = np.repeat(last_ln_wh, h)
        return ln_wh_future, np.repeat(last_ln_cif, h)

def build_future_xreg(df, L, h, beta, mu, ln_wh_future, ln_cif0):
    d_hist = np.r_[np.nan, np.diff(df["ln_wh"].values)]
    d_fut = np.r_[np.nan, np.diff(np.r_[df["ln_wh"].values[-1], ln_wh_future])][1:]
    d_pad = np.r_[d_hist[-L:], d_fut]
    Xcols = {f"dln_wh_L{lag}": d_pad[(L - lag):(L - lag + h)] for lag in range(L + 1)}
    Xf = pd.DataFrame(Xcols)
    pair = np.column_stack([ln_cif0, ln_wh_future])
    ect_raw = pair @ beta
    ect_l1 = np.r_[df["ect"].values[-1], (ect_raw - mu).ravel()][0:h]
    Xf["ect_l1"] = ect_l1
    return Xf

def shock_scale(model_resid, var_ix, use_girf=False):
    if use_girf: return 0.01
    try:
        s = float(np.nanstd(model_resid[:, var_ix]))
        return 0.01 / s if s > 0 else 0.01
    except Exception:
        return 0.01

# ---------------------------
# Page Config & State init
# ---------------------------
st.set_page_config(page_title="CIF vs Wholesale (VECM/ARIMAX)", layout="wide")
st.title("CIF vs Wholesale: 클리닝·패널 → 검증 → 모델링")

for k in ("tx_clean","wh_clean","model_df","pair_title"):
    if k not in st.session_state:
        st.session_state[k] = None

# ===========================
# TABs
# ===========================
tabs = st.tabs([
    "0) 클리닝·패널 생성",
    "데이터",
    "검증A) 정상성",
    "검증B) 공적분·그랜저",
    "1) OLS",
    "2) 공적분·VECM & IRF",
    "3) ECM-ARIMAX",
    "4) Prophet",
    "5) 성능 비교",
])

# -------------------------------------------------------------------
# 0) 클리닝·패널 생성
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("① 원천 업로드")

    # 기본 디렉터리 (사용자 제공 경로)
    DEFAULT_DIR = r"C:\Users\KimByeolha\OneDrive - 트릿지\예측지계산_20250901"

    c1, c2 = st.columns(2)

    # ---------- Transaction 업로드/기본 불러오기 ----------
    with c1:
        tx_file = st.file_uploader("Transaction (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="tx_up")

        # 기본 불러오기 버튼
        if st.button("기본 불러오기: transaction_data_250902.csv", key="btn_tx_default"):
            tx_df = None
            for cand in [
                os.path.join(DEFAULT_DIR, "transaction_data_250902.csv"),
                "transaction_data_250902.csv",
                "/mnt/data/transaction_data_250902.csv",
            ]:
                if os.path.exists(cand):
                    tx_df = read_any(cand)
                    break
            if tx_df is None:
                st.warning("기본 파일을 찾지 못했습니다. 업로드를 사용하세요.")
        else:
            tx_df = read_any(tx_file, getattr(tx_file, "name", None)) if tx_file is not None else None

        if tx_df is not None:
            st.write("Transaction 원시(상위 8)")
            st.dataframe(tx_df.head(8), use_container_width=True)

    # ---------- Wholesale 업로드/기본 불러오기 ----------
    with c2:
        wh_file = st.file_uploader("Wholesale (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="wh_up")

        if st.button("기본 불러오기: wholesale_data_250902.csv", key="btn_wh_default"):
            wh_df = None
            for cand in [
                os.path.join(DEFAULT_DIR, "wholesale_data_250902.csv"),
                "wholesale_data_250902.csv",
                "/mnt/data/wholesale_data_250902.csv",
            ]:
                if os.path.exists(cand):
                    wh_df = read_any(cand)
                    break
            if wh_df is None:
                st.warning("기본 파일을 찾지 못했습니다. 업로드를 사용하세요.")
        else:
            wh_df = read_any(wh_file, getattr(wh_file, "name", None)) if wh_file is not None else None

        if wh_df is not None:
            st.write("Wholesale 원시(상위 8)")
            st.dataframe(wh_df.head(8), use_container_width=True)

    st.markdown("---")
    st.subheader("② 클리닝 & 월평균")

    tx_clean = None
    wh_clean = None
    if (tx_df is not None) and (wh_df is not None):
        try:
            tx_clean = clean_cif_v2(tx_df)
            wh_clean = clean_wholesale_v2(wh_df, aggregate_monthly=True)
            st.success("클리닝 완료")
        except Exception as e:
            st.error(f"클리닝 에러: {e}")

    if tx_clean is not None:
        st.session_state["tx_clean"] = tx_clean
    if wh_clean is not None:
        st.session_state["wh_clean"] = wh_clean

    if (tx_clean is not None) and (wh_clean is not None):
        c3, c4 = st.columns(2)
        with c3:
            st.write("CIF 요약")
            st.dataframe(pd.DataFrame({
                "rows": [len(tx_clean)],
                "기간": [f"{tx_clean['date'].min().date()} ~ {tx_clean['date'].max().date()}"],
                "제품 수": [tx_clean["product_code"].nunique()],
                "수출국 수": [tx_clean["export_iso2"].nunique()],
                "수입국 수": [tx_clean["import_iso2"].nunique()],
            }), use_container_width=True)
        with c4:
            st.write("Wholesale 요약")
            st.dataframe(pd.DataFrame({
                "rows": [len(wh_clean)],
                "기간": [f"{wh_clean['date'].min().date()} ~ {wh_clean['date'].max().date()}"],
                "제품 수": [wh_clean["product_code"].nunique()],
                "국가 수": [wh_clean["country_iso2"].nunique()],
            }), use_container_width=True)

    st.markdown("---")
    st.subheader("③ 패널 설정 & 생성 (쌍 선택)")

    key_side = st.selectbox(
        "매핑 키",
        options=["import", "export"],
        format_func=lambda x: "CIF IMPORT_ISO2 ↔ WH COUNTRY_CODE" if x == "import"
                              else "CIF EXPORT_ISO2 ↔ WH COUNTRY_CODE",
        index=1,
        key="key_side_sel",
    )
    join_type = st.selectbox("조인 방식", ["inner", "left_keep_CIF", "right_keep_Wholesale"], index=0, key="join_sel")

    tx_clean = st.session_state.get("tx_clean")
    wh_clean = st.session_state.get("wh_clean")

    if (tx_clean is None) or (wh_clean is None):
        st.info("먼저 ①~②에서 데이터를 불러와 클리닝을 끝내세요.")
    else:
        st.info("CIF 쪽과 Wholesale 쪽을 각각 선택해 한 쌍(pair) 시계열을 만들고, 날짜만 맞춰 머지합니다.")

        csel1, csel2 = st.columns(2)
        with csel1:
            st.markdown("**CIF 선택**")
            cif_prod = st.selectbox("CIF product_code", sorted(tx_clean["product_code"].unique()), key="cif_prod")
            if key_side == "export":
                cif_iso = st.selectbox(
                    "CIF export_iso2",
                    sorted(tx_clean.loc[tx_clean["product_code"] == cif_prod, "export_iso2"].unique()),
                    key="cif_iso"
                )
            else:
                cif_iso = st.selectbox(
                    "CIF import_iso2",
                    sorted(tx_clean.loc[tx_clean["product_code"] == cif_prod, "import_iso2"].unique()),
                    key="cif_iso"
                )

        with csel2:
            st.markdown("**Wholesale 선택**")
            wh_prod = st.selectbox("WH product_code", sorted(wh_clean["product_code"].unique()), key="wh_prod")
            wh_iso = st.selectbox(
                "WH country_iso2",
                sorted(wh_clean.loc[wh_clean["product_code"] == wh_prod, "country_iso2"].unique()),
                key="wh_iso"
            )

        if st.button("선택한 한 쌍으로 패널 생성", type="primary", key="btn_make_pair"):
            pair = build_pair_panel(
                tx_clean, wh_clean,
                key_side=key_side,
                cif_product=cif_prod, cif_iso=cif_iso,
                wh_product=wh_prod, wh_iso=wh_iso,
                join=join_type
            )
            if pair.empty:
                st.error("머지 결과 0행 — 날짜 겹침이 없습니다. 기간을 점검하세요.")
            else:
                st.success(f"쌍 패널 길이: {len(pair)}")
                st.dataframe(pair.head(30), use_container_width=True)

                # 로그 플롯
                fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
                ax.plot(pair["date"], pair["ln_cif"], label="ln_cif", linewidth=1.2)
                ax.plot(pair["date"], pair["ln_wh"], label="ln_wh", linewidth=1.2)
                title = f"CIF[{pair['pair_cif'].iloc[0]}]  vs  WH[{pair['pair_wh'].iloc[0]}]"
                pretty_ax(ax, title, "log")
                ax.legend(frameon=False, fontsize=10, loc="upper left")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

                # 모델용 데이터 저장
                dfm = pair[["date", "ln_cif", "ln_wh"]].dropna().copy()
                dfm = dfm.sort_values("date").drop_duplicates(subset="date", keep="last").set_index("date")
                st.session_state["model_df"] = dfm
                st.session_state["pair_title"] = title

                # 다운로드
                st.download_button(
                    "현재 쌍 패널 CSV 다운로드",
                    pair.to_csv(index=False).encode("utf-8"),
                    file_name="panel_pair.csv",
                    mime="text/csv",
                    key="dl_pair_csv"
                )


# -------------------------------------------------------------------
# 탭: 데이터
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("미리보기")
    df = st.session_state.get("model_df", None)
    if df is None or len(df) == 0:
        st.info("0) 탭(클리닝·패널)에서 한 쌍을 생성한 뒤 보세요.")
    else:
        if "date" in df.columns:
            show = df.copy()
        else:
            show = df.reset_index(names="date")
        st.dataframe(show.head(12), use_container_width=True)
        st.markdown("---")
        stat = pd.DataFrame({
            "start": [df.index.min().date()],
            "end":   [df.index.max().date()],
            "n":     [len(df)],
            "ln_cif_min": [round(float(df["ln_cif"].min()), 4)],
            "ln_cif_max": [round(float(df["ln_cif"].max()), 4)],
            "ln_wh_min":  [round(float(df["ln_wh"].min()), 4)],
            "ln_wh_max":  [round(float(df["ln_wh"].max()), 4)],
        })
        st.dataframe(stat, use_container_width=True)

# -------------------------------------------------------------------
# 검증A) 정상성
# -------------------------------------------------------------------
with tabs[2]:
    st.subheader("정상성 검증 (ADF + KPSS)")
    df = st.session_state.get("model_df")
    if df is None or df.empty:
        st.info("먼저 0) 탭에서 패널을 선택하세요.")
    else:
        d = pd.DataFrame({
            "dln_cif": np.r_[np.nan, np.diff(df["ln_cif"].values)],
            "dln_wh":  np.r_[np.nan, np.diff(df["ln_wh"].values)],
        }, index=df.index)
        st.write("**레벨**")
        st.write(adf_report(df["ln_cif"], "ln_cif", "c"))
        st.write(adf_report(df["ln_wh"],  "ln_wh",  "c"))
        st.write(kpss_report(df["ln_cif"], "ln_cif", "c"))
        st.write(kpss_report(df["ln_wh"],  "ln_wh",  "c"))
        st.write("---")
        st.write("**1차 차분**")
        st.write(adf_report(d["dln_cif"], "dln_cif", "n"))
        st.write(adf_report(d["dln_wh"],  "dln_wh",  "n"))
        st.write(kpss_report(d["dln_cif"], "dln_cif", "c"))
        st.write(kpss_report(d["dln_wh"],  "dln_wh",  "c"))

# -------------------------------------------------------------------
# 검증B) 공적분 · 그랜저
# -------------------------------------------------------------------
with tabs[3]:
    st.subheader("공적분 & 그랜저 인과성")
    df = st.session_state.get("model_df")
    if df is None or df.empty:
        st.info("먼저 0) 탭에서 패널을 선택하세요.")
    else:
        st.write(eg_coint_report(df["ln_cif"], df["ln_wh"], "ln_cif", "ln_wh"))
        try:
            st.write(johansen_report(df[["ln_cif","ln_wh"]]))
        except Exception:
            pass
        maxlag = st.slider("Granger max lag", 2, 12, 6, 1, key="gr_lag")
        dd = df.copy()
        dd["dln_cif"] = np.r_[np.nan, np.diff(dd["ln_cif"].values)]
        dd["dln_wh"]  = np.r_[np.nan, np.diff(dd["ln_wh"].values)]
        tbl = granger_table_dlog(dd, maxlag=maxlag)
        st.dataframe(tbl, use_container_width=True)

# -------------------------------------------------------------------
# 1) OLS
# -------------------------------------------------------------------
with tabs[4]:
    st.subheader("OLS: ln_cif ~ ln_wh(L)")
    df = st.session_state.get("model_df")
    if df is None or df.empty:
        st.info("먼저 0) 탭에서 패널을 선택하세요.")
    else:
        maxL = st.number_input("최대 시차(L) 탐색", min_value=0, max_value=12, value=3, step=1)
        if st.button("OLS 적합", key="ols_run"):
            rows = []; best = None
            for L in range(int(maxL) + 1):
                mdl, n, aic_val = ols_with_aic(df, L)
                rows.append({"L": L, "n": n, "AIC": round(aic_val, 2)})
                if (best is None) or (aic_val < best["aic"]):
                    best = {"model": mdl, "L": L, "aic": aic_val}
            st.write("라그별 AIC"); st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.write(f"**선택: L={best['L']} (AIC={best['aic']:.2f})**")
            st.text(best["model"].summary().as_text())

# -------------------------------------------------------------------
# 2) 공적분·VECM & IRF
# -------------------------------------------------------------------
def vecm_irf_plot(df, vec_lagmax, irf_h, girf=False):
    X = df[["ln_cif", "ln_wh"]]
    pick = pick_vecm(X, nmax=int(vec_lagmax))

    if pick is None:
        # ===== r=0: VAR(Δlog) IRF =====
        dd = pd.DataFrame({
            "dln_cif": np.r_[np.nan, np.diff(df["ln_cif"].values)],
            "dln_wh":  np.r_[np.nan, np.diff(df["ln_wh"].values)],
        }, index=df.index).dropna()

        if len(dd) < 8:
            st.error(f"표본 부족: Δlog 관측치 {len(dd)}개 → VAR IRF 생략(최소 8~12 권장).")
            return

        k = dd.shape[1]  # 2
        nobs = len(dd)
        user_max = int(vec_lagmax)

        # 표본 기반 가능한 최대 p (보수적으로 클램프)
        feasible_max = max(1, int((nobs - 1) // (k + 1)))
        max_p = max(1, min(int(user_max), int(feasible_max), max(1, nobs // 3), 12))



        # AIC로 p 선택 (실패 시 p=1~2 폴백)
        try:
            sel = VAR(dd).select_order(maxlags=max_p)
            choices = [getattr(sel, "aic", None), getattr(sel, "bic", None),
                       getattr(sel, "hqic", None), getattr(sel, "fpe", None)]
            p = next((int(v) for v in choices if isinstance(v, (int, float)) and v is not None), None)
            if p is None or p < 1 or p > max_p:
                p = max(1, min(2, max_p))
        except Exception:
            p = max(1, min(2, max_p))

        # 마지막 안전장치
        p = int(min(p, max_p, max(1, (nobs - 1) // (k + 1) - 1)))

        try:
            var_d = VAR(dd).fit(p)
        except Exception:
            p = 1
            var_d = VAR(dd).fit(p)

        st.info(f"공적분 r=0 → VAR(Δln) IRF 대체. 사용 p={p} (요청 {user_max}, 표본상한 {feasible_max}, 캡 {max_p}, n={nobs}).")

        # IRF (충격: dln_wh → 반응: dln_cif)
        irf = var_d.irf(int(irf_h))
        scale = shock_scale(var_d.resid, var_ix=dd.columns.get_loc("dln_wh"), use_girf=girf)
        r = irf.irfs[:, dd.columns.get_loc("dln_wh"), dd.columns.get_loc("dln_cif")] * scale

        # 레벨(로그) 효과 보고 싶으면 누적 → %로 변환
        cum_r   = np.cumsum(r)
        lvl_pct = (np.exp(cum_r) - 1) * 100.0

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.axhline(0, ls="--")
        ax.plot(range(len(lvl_pct)), lvl_pct, linewidth=1.2)
        ax.set_title("VAR(Δln) IRF (per +1% shock): Δln_wh → Δln_cif")
        ax.set_xlabel("months")
        ax.set_ylabel("response (% level, cumulative)")
        st.pyplot(fig, clear_figure=True)

    else:
        # ===== r=1: VECM IRF (레벨-로그 반응) =====
        st.success(f"VECM 채택 (k_ar_diff={pick['k']}, r=1)")
        res = pick["res"]
        irf = res.irf(int(irf_h))
        names = list(res.names)
        i_imp = names.index("ln_wh") if "ln_wh" in names else 1
        i_rsp = names.index("ln_cif") if "ln_cif" in names else 0

        arr = np.asarray(res.resid)
        resid_mat = arr if arr.ndim == 2 and arr.shape[1] == len(res.names) else np.column_stack(list(res.resid))
        scale = shock_scale(resid_mat, var_ix=i_imp, use_girf=girf)


        m = irf.irfs[:, i_imp, i_rsp] * scale
        lvl_pct = (np.exp(m) - 1) * 100.0

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.axhline(0, ls="--")
        ax.plot(range(len(lvl_pct)), lvl_pct, linewidth=1.2)
        ax.set_title("VECM IRF (per +1% shock): ln_wh → ln_cif")
        ax.set_xlabel("months")
        ax.set_ylabel("response (% level)")
        st.pyplot(fig, clear_figure=True)

def vecm_or_var_forecast(df, vec_lagmax, h):
    """
    df: index=월(MS), cols=['ln_cif','ln_wh']
    vec_lagmax: VECM/VAR 최대 라그 탐색 상한
    h: horizon (months)
    return: (DataFrame[date, ln_cif_fc, ln_wh_fc], msg)
    """
    X = df[['ln_cif', 'ln_wh']]
    pick = pick_vecm(X, nmax=int(vec_lagmax))
    dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=int(h), freq='MS')

    if pick is not None:
        # ---- r=1: VECM 레벨-로그 예측 ----
        res = pick['res']
        fut = res.predict(steps=int(h))  # shape (h, k)
        names = list(res.names)
        i_cif = names.index('ln_cif') if 'ln_cif' in names else 0
        i_wh  = names.index('ln_wh')  if 'ln_wh'  in names else 1
        ln_cif_fc = fut[:, i_cif]
        ln_wh_fc  = fut[:, i_wh]
        msg = f"VECM 예측 (r=1, k_ar_diff={pick['k']})"
    else:
        # ---- r=0: VAR(Δln) 폴백 → 차분 예측 후 적분 ----
        dd = pd.DataFrame({
            'dln_cif': np.r_[np.nan, np.diff(df['ln_cif'].values)],
            'dln_wh':  np.r_[np.nan, np.diff(df['ln_wh'].values)],
        }, index=df.index).dropna()

        if len(dd) < 8:
            raise ValueError(f"표본 부족: Δlog 관측치 {len(dd)}개 (최소 8~12 권장)")

        k = dd.shape[1]  # 2
        nobs = len(dd)
        user_max = int(vec_lagmax)
        feasible_max = max(1, int((nobs - 1) // (k + 1)))
        max_p = max(1, min(user_max, feasible_max, nobs // 3, 12))

        # AIC 기반 p 선택, 실패시 p=1~2 폴백
        try:
            sel = VAR(dd).select_order(maxlags=max_p)
            cand = [getattr(sel, 'aic', None), getattr(sel, 'bic', None),
                    getattr(sel, 'hqic', None), getattr(sel, 'fpe', None)]
            p = next((int(v) for v in cand if isinstance(v, (int, float)) and v is not None), None)
            if p is None or p < 1 or p > max_p:
                p = max(1, min(2, max_p))
        except Exception:
            p = max(1, min(2, max_p))

        p = int(min(p, max_p, max(1, (nobs - 1) // (k + 1) - 1)))

        var_d = VAR(dd).fit(p)
        fc_d = var_d.forecast(y=dd.values[-p:], steps=int(h))  # 예측된 Δlog
        d_cif = fc_d[:, dd.columns.get_loc('dln_cif')]
        d_wh  = fc_d[:, dd.columns.get_loc('dln_wh')]

        ln_cif_fc = df['ln_cif'].values[-1] + np.cumsum(d_cif)
        ln_wh_fc  = df['ln_wh'].values[-1]  + np.cumsum(d_wh)
        msg = f"VAR(Δln) 폴백 예측 (p={p}, n={nobs})"

    out = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'ln_cif_fc': np.round(ln_cif_fc, 4),
        'ln_wh_fc':  np.round(ln_wh_fc,  4),
    })
    return out, msg


with tabs[5]:
    st.subheader("공적분 판정 & IRF")
    df = st.session_state.get("model_df")
    if df is None or df.empty:
        st.info("먼저 0) 탭에서 패널을 선택하세요.")
    else:
        vec_lagmax = st.number_input("VAR lag.max (AIC 선정용)", min_value=4, max_value=24, value=12, step=1)
        irf_h = st.number_input("IRF horizon (months)", min_value=4, max_value=48, value=12, step=1)
        girf = st.checkbox("Orthogonal off (참고용)", value=False)
        if st.button("VECM/IRF 실행", key="vecm_run"):
            vecm_irf_plot(df, vec_lagmax, irf_h, girf)

# -------------------------------------------------------------------
# 3) ECM-ARIMAX
# -------------------------------------------------------------------
with tabs[6]:
    st.subheader("ECM-ARIMAX")
    df = st.session_state.get("model_df")
    if df is None or df.empty:
        st.info("먼저 0) 탭에서 패널을 선택하세요.")
    else:
        Lmax = st.number_input("Δln_wh 최대시차(Lmax)", min_value=0, max_value=12, value=3, step=1)
        scn = st.selectbox("ln_wh 시나리오", options=["vec","flat"],
                           format_func=lambda x: "VECM 경로" if x=="vec" else "flat(마지막값 고정)")
        h_steps = st.slider("예측 horizon", min_value=6, max_value=24, value=12, step=1)
        conf = st.slider("신뢰수준(%)", min_value=60, max_value=95, value=80, step=5)
        if st.button("ECM-ARIMAX 실행", key="ax_run"):
            vec_pick = pick_vecm(df[["ln_cif","ln_wh"]], nmax=12)
            has_coint = vec_pick is not None
            vec_res = vec_pick["res"] if has_coint else None
            ect, beta, mu = make_ect(df, vec_res if has_coint else None)
            dwork = df.copy(); dwork["ect"] = ect
            best = None; recs = []
            for L in range(int(Lmax)+1):
                D = design_matrix_for_ecm(dwork, L=L, include_ect=has_coint)
                if D.empty: continue
                y = D["dln_cif"].values; X = D.drop(columns=["dln_cif"]).values
                fit, _tab = fit_arima_grid(y, X=X, seasonal=False)
                if fit is None: continue
                aicc_val = fit["AICc"]; bic_val = fit["BIC"]
                recs.append({"L": L, "n": len(y), "AICc": round(aicc_val, 2), "BIC": round(bic_val, 2)})
                if (best is None) or (aicc_val < best["AICc"]):
                    best = {"fit": fit["res"], "L": L, "AICc": aicc_val, "BIC": bic_val}
            if best is None:
                st.error("ARIMAX 선택 실패")
            else:
                ln_wh_future, ln_cif0 = forecast_lnwh(df, h=int(h_steps), vecm_res=vec_res, method=scn)
                Xf = build_future_xreg(dwork, L=best["L"], h=int(h_steps), beta=beta, mu=mu,
                                       ln_wh_future=ln_wh_future, ln_cif0=ln_cif0)
                Xf_aligned = align_xreg(best["fit"], Xf)
                pred = best["fit"].get_forecast(steps=int(h_steps), exog=Xf_aligned)
                dln_fc = np.asarray(pred.predicted_mean)
                ci = pred.conf_int(alpha=1 - conf/100.0)
                if hasattr(ci, "to_numpy"):
                        ci_np = ci.to_numpy()
                else:
                        ci_np = np.asarray(ci)
                lo_arr = ci_np[:, 0]
                hi_arr = ci_np[:, 1]

                ln_last = df["ln_cif"].values[-1]
                ln_hat = ln_last + np.cumsum(dln_fc)
                ln_lo  = ln_last + np.cumsum(lo_arr)
                ln_hi  = ln_last + np.cumsum(hi_arr)
                dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=int(h_steps), freq="MS")
                tbl = pd.DataFrame({
                    "date": dates.strftime("%Y-%m-%d"),
                    "ln_cif_hat": np.round(ln_hat, 4),
                    "CIF_hat": np.round(np.exp(ln_hat), 4),
                    "ln_lo": np.round(ln_lo, 4),
                    "ln_hi": np.round(ln_hi, 4),
                })
                st.write("라그 선택(AICc)"); st.dataframe(pd.DataFrame(recs), use_container_width=True)
                st.write(f"**선택: L={best['L']} (AICc={best['AICc']:.2f}, BIC={best['BIC']:.2f}) | 시나리오: {scn} | ECT {'포함' if has_coint else '제외'}**")

                fig1, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
                hist_ln = df["ln_cif"]; ax1.plot(hist_ln.index, hist_ln.values, label="hist", linewidth=1.2)
                ax1.plot(dates, ln_hat, label="fc", linewidth=1.4)
                ax1.fill_between(dates, ln_lo, ln_hi, alpha=0.18)
                pretty_ax(ax1, "ECM-ARIMAX (log)", "ln(CIF)")
                ax1.legend(frameon=False, fontsize=10, loc="upper left")
                fig1.tight_layout(); st.pyplot(fig1, clear_figure=True)

                fig2, ax2 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
                ax2.plot(hist_ln.index, np.exp(hist_ln.values), label="hist", linewidth=1.2)
                ax2.plot(dates, np.exp(ln_hat), label="fc", linewidth=1.4)
                pretty_ax(ax2, "ECM-ARIMAX (level, mean only)", "CIF(level)")
                ax2.legend(frameon=False, fontsize=10, loc="upper left")
                fig2.tight_layout(); st.pyplot(fig2, clear_figure=True)

                st.subheader("예측표"); st.dataframe(tbl, use_container_width=True)

# -------------------------------------------------------------------
# 4) Prophet
# -------------------------------------------------------------------
with tabs[7]:
    st.subheader("Prophet (선택 패널 기본 사용)")
    base = st.session_state.get("model_df")

    # 업로더는 옵션(override)
    f = st.file_uploader("대신 업로드(옵션, cols: date, ln_cif, ln_wh)", type=["csv"], key="prop_csv")
    if f is not None:
        df_in = read_any(f, getattr(f,"name",None))
        if df_in is not None:
            df_in.columns = [c.lower() for c in df_in.columns]
            need = {"date","ln_cif","ln_wh"}
            if need.issubset(set(df_in.columns)):
                df_in["date"] = month_floor(df_in["date"])
                df_in["ln_cif"] = pd.to_numeric(df_in["ln_cif"], errors="coerce")
                df_in["ln_wh"]  = pd.to_numeric(df_in["ln_wh"],  errors="coerce")
                df_in = df_in.dropna(subset=["date","ln_cif","ln_wh"])

                df_in = df_in.sort_values("date")
                df_in.index = df_in["date"]
                base = df_in[["ln_cif","ln_wh"]]
            else:
                st.error("필수 컬럼: date, ln_cif, ln_wh")

    if base is None or base.empty:
        st.info("먼저 0) 탭에서 패널 한 쌍을 선택(생성)하거나 CSV를 업로드하세요.")
    else:
        use_reg = st.checkbox("ln_wh 리그레서 사용", value=True)
        align_start = st.checkbox("첫 점 보정(연속성)", value=True)
        h = st.slider("예측 horizon (months)", 6, 24, 12, 1)
        if not HAS_PROPHET:
            st.error("prophet 패키지가 설치되어 있지 않습니다.  pip install prophet") 
else:
    # Prophet 학습/예측 블록 (깨진 줄 싹 제거)
    data = pd.DataFrame({
        "ds": pd.to_datetime(base.index),
        "y": base["ln_cif"].values,
        "ln_wh": base["ln_wh"].values
    })

    m = Prophet(
        yearly_seasonality=True,   # 월간이면 연간 seasonality는 True면 충분
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        random_state=42
    )
    if use_reg:
        m.add_regressor("ln_wh", standardize=True)

    m.fit(data[["ds","y","ln_wh"]] if use_reg else data[["ds","y"]])

    future = m.make_future_dataframe(periods=int(h), freq="MS")
    if use_reg:
        # 과거는 관측치 사용, 미래 구간은 마지막 값으로 보간/연장
        future = future.merge(data[["ds","ln_wh"]], on="ds", how="left")
        future["ln_wh"] = future["ln_wh"].ffill()

    p = m.predict(future)
    fc = p.tail(int(h)).copy()

    # 첫 점 보정(연속성)
    try:
        last_fit = p[p["ds"] == data["ds"].iloc[-1]]
        bias = float(data["y"].iloc[-1] - last_fit["yhat"].iloc[0]) if (align_start and not last_fit.empty) else 0.0
    except Exception:
        bias = 0.0
    for col in ["yhat","yhat_lower","yhat_upper"]:
        fc[col] = fc[col] + bias

            fig1, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            ax1.plot(data["ds"], data["y"], label="hist", linewidth=1.2)
            ax1.plot(fc["ds"], fc["yhat"], label="fc", linewidth=1.4)
            ax1.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.18)
            pretty_ax(ax1, f"Prophet{' + ln_wh' if use_reg else ''} (log)", "ln(CIF)")
            ax1.legend(frameon=False, fontsize=10, loc="upper left")
            fig1.tight_layout(); st.pyplot(fig1, clear_figure=True)

            fig2, ax2 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            ax2.plot(data["ds"], np.exp(data["y"]), label="hist", linewidth=1.2)
            ax2.plot(fc["ds"], np.exp(fc["yhat"]), label="fc", linewidth=1.4)
            pretty_ax(ax2, f"Prophet{' + ln_wh' if use_reg else ''} (level, mean only)", "CIF(level)")
            ax2.legend(frameon=False, fontsize=10, loc="upper left")
            fig2.tight_layout(); st.pyplot(fig2, clear_figure=True)

            out_tbl = fc[["ds","yhat","yhat_lower","yhat_upper"]].copy()
            out_tbl["CIF_hat"] = np.exp(out_tbl["yhat"])
            st.subheader("예측표")
            st.dataframe(out_tbl.rename(columns={"ds":"date","yhat":"ln_cif_hat","yhat_lower":"ln_lo","yhat_upper":"ln_hi"}),
                         use_container_width=True)

# -------------------------------------------------------------------
# 5) 성능 비교
# -------------------------------------------------------------------
with tabs[8]:
    st.subheader("홀드아웃 성능 비교 (ECM-ARIMAX vs Prophet)")
    df = st.session_state.get("model_df")
    if df is None or df.empty:
        st.info("먼저 0) 탭에서 패널을 선택하세요.")
    else:
        h_eval = st.slider("홀드아웃 길이(h)", min_value=6, max_value=18, value=12, step=1, key="eval_h")
        pr_use_reg = st.checkbox("Prophet에 ln_wh 리그레서 사용", value=True)
        if st.button("평가 실행", key="eval_run"):
            if len(df) <= int(h_eval) + 12:
                st.error("표본이 너무 짧습니다. h를 줄이세요.")
            else:
                train = df.iloc[:-int(h_eval)]
                test  = df.iloc[-int(h_eval):]

                # ECM-ARIMAX
                vec_pick = pick_vecm(train[["ln_cif","ln_wh"]], nmax=12)
                has_coint = vec_pick is not None
                vec_res = vec_pick["res"] if has_coint else None
                ect, beta, mu = make_ect(train, vec_res if has_coint else None)
                tr = train.copy(); tr["ect"] = ect
                best = None
                for L in range(0, 4):
                    D = design_matrix_for_ecm(tr, L=L, include_ect=has_coint)
                    if D.empty: continue
                    y = D["dln_cif"].values; X = D.drop(columns=["dln_cif"]).values
                    fit, _tab = fit_arima_grid(y, X=X, seasonal=False)
                    if fit is None: continue
                    if (best is None) or (fit["AICc"] < best["AICc"]):
                        best = {"res": fit["res"], "L": L, "AICc": fit["AICc"]}
                ln_wh_future = test["ln_wh"].values
                ln_cif0 = np.repeat(train["ln_cif"].values[-1], len(test))
                Xf = build_future_xreg(tr, L=best["L"], h=len(test), beta=beta, mu=mu,
                                       ln_wh_future=ln_wh_future, ln_cif0=ln_cif0)
                Xf_aligned = align_xreg(best["res"], Xf)
                pred_ax = best["res"].get_forecast(steps=len(test), exog=Xf_aligned)
                dln_ax = np.asarray(pred_ax.predicted_mean)
                ln_ax  = train["ln_cif"].values[-1] + np.cumsum(dln_ax)

                # Prophet
                if not HAS_PROPHET:
                    st.warning("prophet 미설치 — Prophet 평가는 생략")
                    ln_pr = np.full(len(test), np.nan)
                else:
                    M = pd.DataFrame({"ds": pd.to_datetime(train.index), "y": train["ln_cif"].values, "ln_wh": train["ln_wh"].values})
                    m = Prophet(yearly_seasonality=12, weekly_seasonality=False, daily_seasonality=False,
                                seasonality_mode="additive", changepoint_prior_scale=0.05)
                    if pr_use_reg: m.add_regressor("ln_wh", standardize=True)
                    m.fit(M[["ds","y","ln_wh"]] if pr_use_reg else M[["ds","y"]])
                    fut = pd.DataFrame({"ds": pd.to_datetime(test.index)})
                    if pr_use_reg: fut["ln_wh"] = test["ln_wh"].values
                    p = m.predict(fut); ln_pr = p["yhat"].values

                y_true_lvl = np.exp(test["ln_cif"].values)
                ax_lvl = np.exp(ln_ax)
                pr_lvl = np.exp(ln_pr) if np.all(np.isfinite(ln_pr)) else np.full_like(y_true_lvl, np.nan)

                res_tab = pd.DataFrame({
                    "Model": [
                        f"ECM-ARIMAX{' (ECT 포함)' if has_coint else ''}",
                        f"Prophet{' + ln_wh' if pr_use_reg else ''}",
                    ],
                    "RMSE": [round(rmse(y_true_lvl, ax_lvl), 3),
                             round(rmse(y_true_lvl, pr_lvl), 3) if np.isfinite(pr_lvl).all() else np.nan],
                    "MAPE": [round(mape(y_true_lvl, ax_lvl), 3),
                             round(mape(y_true_lvl, pr_lvl), 3) if np.isfinite(pr_lvl).all() else np.nan],
                    "HitRatio(Δln, %)": [
                        round(hit_ratio(test["ln_cif"].values, ln_ax), 1),
                        round(hit_ratio(test["ln_cif"].values, ln_pr), 1) if np.isfinite(ln_pr).all() else np.nan,
                    ],
                })
                st.write(f"홀드아웃 구간: {test.index.min().date()} ~ {test.index.max().date()}")
                st.dataframe(res_tab, use_container_width=True)
