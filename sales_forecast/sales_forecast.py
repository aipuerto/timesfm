#!/usr/bin/env python3
"""TimesFM 2.5 ile satış serisi tahmini (bölge / müşteri / ürün bazında).

Girdi: CSV veya Excel (.xlsx), iki formattan biri:
  1) Uzun:  date,series,value
        2024-01-31,Latin America,125000
        2024-01-31,Africa,98000
     (`series` sütunu yoksa tek seri varsayılır)
  2) Geniş: ilk sütun seri/firma adı, diğer sütun başlıkları yıl veya tarih
        Firma,2023,2024,2025,2026
     'Genel Toplam' gibi özet satırlar otomatik çıkarılır.
  3) Ay bloğu: ay adı satırı (ay toplamı), altında o ayın firmaları; sütunlar yıl
        Ocak,   140051, 197791   <- ay toplamı (veriye katılmaz)
        EUROFLEX, 41499,  67288  <- o ayın firma satırı
     Dolu olan son ay otomatik bulunur; boş gelecek aylar sıfır sayılmaz.

Çıktı: forecast.csv  (series, date, p10, p50, p90, mean)
       backtest.csv  (--backtest N verilirse: seri başına MAPE / sMAPE)
       forecast.png  (matplotlib kuruluysa)

Örnek:
    python sales_forecast.py --csv sales.csv --horizon 6 --backtest 6
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
# forecast() çıktısı: quantiles[..., 0] = ortalama, 1..9 = P10..P90
Q_MEAN, Q10, Q50, Q90 = 0, 1, 5, 9


TOTAL_ROW = re.compile(r"^\s*(genel\s*toplam|toplam|grand\s*total|total)\s*$", re.I)


MONTHS = {m: i + 1 for i, m in enumerate(
    ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
     "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"])}
MONTHS.update({m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"])})


def _month_no(x) -> int | None:
    return MONTHS.get(str(x).strip().lower())


def block_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Ay bloğu düzeni: ay adı satırı (ay toplamı), altında o ayın firma satırları;
    sütunlar yıl. Ay satırları ve toplam satırları veriye katılmaz."""
    name_col = df.columns[0]
    years = {}
    for c in df.columns[1:]:
        p = _as_period_start(c)
        if p is not None:
            years[c] = p.year
    rows, cur = [], None
    for _, r in df.iterrows():
        nm = str(r[name_col]).strip()
        m = _month_no(nm)
        if m:
            cur = m
            continue
        if cur is None or TOTAL_ROW.match(nm):
            continue
        for c, y in years.items():
            rows.append((pd.Timestamp(year=y, month=cur, day=1), nm, r[c]))
    return pd.DataFrame(rows, columns=["date", "series", "value"])


def read_table(path: str, sheet: str | int) -> pd.DataFrame:
    if Path(path).suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def _as_period_start(col) -> pd.Timestamp | None:
    """Sütun başlığı yıl (2024 / 2024.0) veya tarih ise Timestamp döndür."""
    try:
        f = float(col)
        if f.is_integer() and 1900 <= f <= 2200:
            return pd.Timestamp(year=int(f), month=1, day=1)
        return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(col)
    except (ValueError, TypeError):
        return None


def to_long(df: pd.DataFrame, date_col: str, value_col: str,
            series_col: str) -> pd.DataFrame:
    """Uzun (date, series, value) veya geniş (ilk sütun = seri adı, diğer
    sütunlar = yıl/tarih) formatı uzun formata çevir."""
    if date_col in df.columns and value_col in df.columns:
        out = df.copy()
        if series_col not in out.columns:
            out[series_col] = "toplam"
        return out[[date_col, series_col, value_col]].rename(
            columns={date_col: "date", series_col: "series", value_col: "value"})

    name_col = df.columns[0]
    if df[name_col].map(_month_no).notna().sum() >= 6:
        return block_to_long(df)
    period_cols = {c: _as_period_start(c) for c in df.columns[1:]}
    period_cols = {c: p for c, p in period_cols.items() if p is not None}
    if not period_cols:
        sys.exit("Format tanınmadı. Ya '%s','%s' sütunları olmalı ya da ilk sütun "
                 "seri adı, diğerleri yıl/tarih başlıklı olmalı. Sütunlar: %s"
                 % (date_col, value_col, list(df.columns)))
    long = df.melt(id_vars=name_col, value_vars=list(period_cols),
                   var_name="date", value_name="value")
    long["date"] = long["date"].map(period_cols)
    return long.rename(columns={name_col: "series"})[["date", "series", "value"]]


def load_series(path: str, date_col: str, value_col: str, series_col: str,
                freq: str, sheet: str | int = 0, add_total: bool = False,
                last_fraction: float = 1.0) -> dict[str, pd.Series]:
    long = to_long(read_table(path, sheet), date_col, value_col, series_col)
    long["date"] = pd.to_datetime(long["date"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["series"] = long["series"].astype(str)

    # 'Genel Toplam' gibi özet satırlar çift saymaya yol açar
    is_total = long["series"].map(lambda x: bool(TOTAL_ROW.match(x)))
    if is_total.any():
        print("Özet satır(lar) çıkarıldı: %s" % sorted(long.loc[is_total, "series"].unique()))
    long = long[~is_total]

    # Verinin gerçekten dolu olduğu son dönem (boş hücreler dönem sayılmaz)
    last_obs = long.loc[long["value"].notna(), "date"].max()
    print("Verinin son dolu dönemi: %s" % last_obs.date())

    out = {}
    for name, g in long.groupby("series", sort=False):
        g = g.dropna(subset=["value"])      # boş hücre = veri yok, dönem sayılmaz
        if g.empty:
            continue
        s = g.set_index("date")["value"].resample(freq).sum(min_count=1)
        s = s.loc[s.first_valid_index():]   # ilk satıştan önceki boşluklar silinir
        # Satışı durmuş firma da ortak son döneme kadar 0 olarak devam etmeli
        idx = pd.date_range(s.index[0], max(last_obs, s.index[-1]), freq=freq)
        out[name] = s.reindex(idx).fillna(0.0)  # aradaki/sondaki boşluklar = satış yok

    if add_total:
        out["TOPLAM (hesaplanan)"] = pd.concat(out.values(), axis=1).sum(axis=1)

    if last_fraction < 1.0:  # son dönem eksikse (örn. yılın %85'i) yıllıklandır
        for s in out.values():
            s.iloc[-1] = s.iloc[-1] / last_fraction
    return out


def build_model(max_context: int, max_horizon: int):
    import timesfm  # geç import: --help torch olmadan da çalışsın

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID, torch_compile=False)
    model.compile(timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
        infer_is_positive=True,  # satış negatif olamaz
    ))
    return model


def forecast(model, arrays: list[np.ndarray], horizon: int):
    point, quant = model.forecast(horizon=horizon, inputs=arrays)
    return np.asarray(point), np.asarray(quant)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(pred)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(actual - pred)[mask] / denom[mask]) * 100) if mask.any() else float("nan")


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs(actual - pred)[mask] / np.abs(actual[mask])) * 100) if mask.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Girdi CSV yolu")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--series-col", default="series", help="Seri sütunu (yoksa tek seri varsayılır)")
    ap.add_argument("--freq", default="MS", help="Pandas frekansı: MS=ay başı, W=hafta, D=gün, QS=çeyrek")
    ap.add_argument("--horizon", type=int, default=6, help="Kaç dönem ileri tahmin")
    ap.add_argument("--backtest", type=int, default=0,
                    help="Son N dönemi saklayıp modeli bunlar üzerinde sına")
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--sheet", default=0, help="Excel sayfa adı veya sırası (varsayılan: ilk sayfa)")
    ap.add_argument("--add-total", action="store_true",
                    help="Tüm serilerin toplamını da ayrı seri olarak tahmin et")
    ap.add_argument("--last-fraction", type=float, default=1.0,
                    help="Son dönemin tamamlanma oranı (0-1). Örn. yıl %%85 bittiyse 0.85: "
                         "son değer yıllıklandırılır")
    ap.add_argument("--min-history", type=int, default=8,
                    help="Bundan kısa geçmişi olan seriler atlanır (TimesFM kısa seride güvenilmez)")
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    series = load_series(args.csv, args.date_col, args.value_col, args.series_col,
                         args.freq, sheet, args.add_total, args.last_fraction)

    skipped = {nm: len(s) for nm, s in series.items() if len(s) < args.min_history}
    series = {nm: s for nm, s in series.items() if len(s) >= args.min_history}
    if skipped:
        print("Atlandı (geçmiş < %d dönem): %d seri" % (args.min_history, len(skipped)))
    if not series:
        longest = max(skipped.values(), default=0)
        sys.exit("Tahmin edilecek seri kalmadı (en uzun geçmiş: %d dönem). Daha sık veri "
                 "(ör. aylık) kullan veya --min-history değerini düşür." % longest)
    names = list(series)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_h = max(args.horizon, args.backtest, 1)
    model = build_model(args.max_context, max_h)

    # --- Backtest ---
    if args.backtest:
        n = args.backtest
        short = [s.iloc[:-n].to_numpy(np.float32) for s in series.values()]
        too_short = [nm for nm, a in zip(names, short) if len(a) < 12]
        if too_short:
            print(f"UYARI: geçmişi çok kısa seriler (backtest güvenilmez): {too_short}")
        point, _ = forecast(model, short, n)
        rows = []
        for nm, p, s in zip(names, point, series.values()):
            actual = s.iloc[-n:].to_numpy(np.float64)
            pred = p[:n].astype(np.float64)
            rows.append({"series": nm, "mape_%": round(mape(actual, pred), 2),
                         "smape_%": round(smape(actual, pred), 2)})
        bt = pd.DataFrame(rows)
        bt.to_csv(out_dir / "backtest.csv", index=False)
        print("\nBacktest (son %d dönem):\n%s" % (n, bt.to_string(index=False)))

    # --- Asıl tahmin ---
    arrays = [s.to_numpy(np.float32) for s in series.values()]
    point, quant = forecast(model, arrays, args.horizon)

    rows = []
    for i, (nm, s) in enumerate(series.items()):
        future_idx = pd.date_range(s.index[-1], periods=args.horizon + 1, freq=args.freq)[1:]
        for h, d in enumerate(future_idx):
            rows.append({
                "series": nm, "date": d.date().isoformat(),
                "p10": round(float(quant[i, h, Q10]), 2),
                "p50": round(float(quant[i, h, Q50]), 2),
                "p90": round(float(quant[i, h, Q90]), 2),
                "mean": round(float(point[i, h]), 2),
            })
    fc = pd.DataFrame(rows)
    fc.to_csv(out_dir / "forecast.csv", index=False)
    print("\nTahmin:\n%s" % fc.to_string(index=False))

    # --- Grafik (opsiyonel) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib yok, grafik atlandı)")
        return

    fig, axes = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)), squeeze=False)
    for ax, nm in zip(axes[:, 0], names):
        hist = series[nm]
        f = fc[fc.series == nm]
        fdates = pd.to_datetime(f.date)
        ax.plot(hist.index, hist.values, label="Gerçekleşen")
        ax.plot(fdates, f.p50, label="Tahmin (P50)")
        ax.fill_between(fdates, f.p10, f.p90, alpha=0.25, label="P10–P90")
        ax.set_title(nm)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "forecast.png", dpi=150)
    print(f"\nÇıktılar: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
