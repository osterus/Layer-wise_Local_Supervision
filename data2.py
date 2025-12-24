import pandas as pd
import numpy as np

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("ffv2_test.csv")

b = df["batch"].values
loss_bp = df["loss_bp"].values
loss_ff = df["loss_ff"].values
diff = loss_ff - loss_bp

n = len(df)

# =========================
# 2. Функции статистики
# =========================
def compute_stats(x):
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "final": x[-1],
        "auc": np.trapz(x)
    }

def print_stats(title, stats):
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        print(f"{k:>6}: {v:.6f}")

# =========================
# 3. Полный лог
# =========================
stats_bp = compute_stats(loss_bp)
stats_ff = compute_stats(loss_ff)
stats_diff = compute_stats(diff)

print_stats("BP (full log)", stats_bp)
print_stats("FF (full log)", stats_ff)
print_stats("DIFF (full log)", stats_diff)

# =========================
# 4. Хаотичность (rolling STD)
# =========================
window = max(5, n//30)
rolling_std_bp = pd.Series(loss_bp).rolling(window).std()
rolling_std_ff = pd.Series(loss_ff).rolling(window).std()

print(f"\nRolling STD window: {window}")
print(f"BP rolling STD mean: {rolling_std_bp.mean():.6f}, max: {rolling_std_bp.max():.6f}")
print(f"FF rolling STD mean: {rolling_std_ff.mean():.6f}, max: {rolling_std_ff.max():.6f}")

# =========================
# 5. Расчёт ускорения FF относительно BP
# =========================
bp_reach = []
for lf in loss_ff:
    idx = np.where(loss_bp <= lf)[0]
    if len(idx) == 0:
        bp_reach.append(np.nan)
    else:
        bp_reach.append(idx[0])
bp_reach = np.array(bp_reach)

speedup = bp_reach / b
valid = ~np.isnan(speedup)
mean_speedup = np.nanmean(speedup)
print(f"\nFF mean speedup over BP (full log): {mean_speedup:.2f}x")

# =========================
# 6. Разделение на 10 частей с локальным speedup
# =========================
print("\n=== STATS BY 10% PARTS ===")
n_parts = 10
part_len = n // n_parts
part_speedups = []

for i in range(n_parts):
    start = i*part_len
    end = (i+1)*part_len if i < n_parts-1 else n
    part_slice = slice(start, end)

    part_bp = loss_bp[part_slice]
    part_ff = loss_ff[part_slice]
    part_diff = diff[part_slice]
    part_b = b[part_slice]

    print(f"\n--- Part {i+1} ({start}-{end}) ---")
    print_stats("BP", compute_stats(part_bp))
    print_stats("FF", compute_stats(part_ff))
    print_stats("DIFF", compute_stats(part_diff))

    # Rolling STD по части
    roll_bp = pd.Series(part_bp).rolling(window).std()
    roll_ff = pd.Series(part_ff).rolling(window).std()
    print(f"BP rolling STD mean: {roll_bp.mean():.6f}, max: {roll_bp.max():.6f}")
    print(f"FF rolling STD mean: {roll_ff.mean():.6f}, max: {roll_ff.max():.6f}")

    # Speedup по части
    part_bp_reach = []
    for lf in part_ff:
        idx = np.where(part_bp <= lf)[0]
        if len(idx) == 0:
            part_bp_reach.append(np.nan)
        else:
            part_bp_reach.append(idx[0])
    part_bp_reach = np.array(part_bp_reach)
    part_speedup = part_bp_reach / part_b
    part_valid = ~np.isnan(part_speedup)
    mean_part_speedup = np.nanmean(part_speedup)
    part_speedups.append(mean_part_speedup)
    print(f"FF mean speedup over BP (this part): {mean_part_speedup:.2f}x")

# =========================
# 7. Поиск 3–4 пиков ускорения
# =========================
speedup_smooth = pd.Series(speedup).rolling(window=max(1, n//50), min_periods=1).mean()
# Находим локальные максимумы
from scipy.signal import find_peaks

peaks, _ = find_peaks(speedup_smooth)
# сортируем по величине speedup
top_peaks_idx = peaks[np.argsort(speedup_smooth.iloc[peaks])[::-1][:10]]

print("\n=== TOP 3-4 PEAKS OF FF SPEEDUP ===")
for idx in top_peaks_idx:
    start_batch = b[max(0, idx-1)]
    end_batch = b[min(n-1, idx+1)]
    print(f"Peak at batch {b[idx]}, speedup={speedup_smooth.iloc[idx]:.2f}x, period ~ {start_batch}-{end_batch}")
