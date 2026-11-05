#!/usr/bin/env python3
"""TimesFM 2.5 ile satış serisi tahmini (bölge / müşteri / ürün bazında).

Girdi CSV (uzun format):   date,series,value
    2024-01-31,Latin America,125000
    2024-01-31,Africa,98000
Tek seri ise `series` sütunu olmadan da çalışır:   date,value

Çıktı: forecast.csv  (series, date, p10, p50, p90, mean)
       backtest.csv  (--backtest N verilirse: seri başına MAPE / sMAPE)
       forecast.png  (matplotlib kuruluysa)

Örnek:
    python sales_forecast.py --csv sales.csv --horizon 6 --backtest 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
# forecast() çıktısı: quantiles[..., 0] = ortalama, 1..9 = P10..P90
Q_MEAN, Q10, Q50, Q90 = 0, 1, 5, 9


def load_series(path: str, date_col: str, value_col: str,
                series_col: str | None, freq: str) -> dict[str, pd.Series]:
    df = pd.read_csv(path)
    for col in (date_col, value_col):
        if col not in df.columns:
            sys.exit(f"CSV'de '{col}' sütunu yok. Mevcut sütunlar: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if series_col and series_col in df.columns:
        groups = {str(k): g for k, g in df.groupby(series_col)}
    else:
        groups = {"toplam": df}

    out = {}
    for name, g in groups.items():
        # Aynı döneme düşen satırları topla, eksik dönemleri 0 ile doldur
        s = (g.set_index(date_col)[value_col].resample(freq).sum(min_count=1))
        s = s.fillna(0.0)
        out[name] = s
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
    args = ap.parse_args()

    series = load_series(args.csv, args.date_col, args.value_col, args.series_col, args.freq)
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
