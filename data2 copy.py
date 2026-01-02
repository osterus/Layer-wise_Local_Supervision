import pandas as pd
import numpy as np

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("ffv2_test64_.csv")

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
# Сглаживание loss (EMA / SMA)
# =========================
smooth_mode = "ema"   # "ema" или "sma"

window_smooth = max(1, n//50)

if smooth_mode == "sma":
    smoothed_bp = (
        pd.Series(loss_bp)
        .rolling(window=window_smooth, min_periods=1)
        .mean()
        .values
    )
    smoothed_ff = (
        pd.Series(loss_ff)
        .rolling(window=window_smooth, min_periods=1)
        .mean()
        .values
    )

elif smooth_mode == "ema":
    # span ~ window_smooth, adjust=False = классическая EMA
    smoothed_bp = (
        pd.Series(loss_bp)
        .ewm(span=window_smooth, adjust=False)
        .mean()
        .values
    )
    smoothed_ff = (
        pd.Series(loss_ff)
        .ewm(span=window_smooth, adjust=False)
        .mean()
        .values
    )

else:
    raise ValueError("Unknown smooth_mode")

# =========================
# 5. Расчёт ускорения FF относительно BP через производные (с учетом сглаживания)
# =========================
epsilon = 1e-9

slope_bp = np.gradient(smoothed_bp)
slope_ff = np.gradient(smoothed_ff)

# Используем модуль наклона для корректной интерпретации ускорения
abs_slope_bp = np.abs(slope_bp)
abs_slope_ff = np.abs(slope_ff)

# Фильтруем случаи, где наклон BP >= 0, заменяя на np.inf
abs_slope_bp_filtered = np.where(slope_bp < 0, abs_slope_bp, np.inf)

with np.errstate(divide='ignore', invalid='ignore'):
    speedup_by_slope = abs_slope_ff / (abs_slope_bp_filtered + epsilon)
    speedup_by_slope = np.where(abs_slope_bp_filtered == np.inf, np.inf, speedup_by_slope)

if smooth_mode == "ema":
    speedup_by_slope_smooth = pd.Series(speedup_by_slope).ewm(
        span=window_smooth, adjust=False
    ).mean()
else:
    speedup_by_slope_smooth = pd.Series(speedup_by_slope).rolling(
        window=window_smooth, min_periods=1
    ).mean()

valid = ~np.isnan(speedup_by_slope)
mean_speedup = np.nanmean(speedup_by_slope)
print(f"\nFF mean speedup over BP (by derivative): {mean_speedup:.2f}x")

# =========================
# Расчёт ускорения по частям
# =========================
n_parts = 15
part_size = n // n_parts

for i in range(n_parts):
    start_idx = i * part_size
    end_idx = (i + 1) * part_size if i < n_parts - 1 else n

    part_loss_bp = loss_bp[start_idx:end_idx]
    part_loss_ff = loss_ff[start_idx:end_idx]
    part_diff = diff[start_idx:end_idx]

    part_stats_bp = compute_stats(part_loss_bp)
    part_stats_ff = compute_stats(part_loss_ff)
    part_stats_diff = compute_stats(part_diff)

    print_stats(f"BP (part {i+1})", part_stats_bp)
    print_stats(f"FF (part {i+1})", part_stats_ff)
    print_stats(f"DIFF (part {i+1})", part_stats_diff)

    part_rolling_std_bp = pd.Series(part_loss_bp).rolling(window).std()
    part_rolling_std_ff = pd.Series(part_loss_ff).rolling(window).std()

    print(f"\nRolling STD window: {window} (part {i+1})")
    print(f"BP rolling STD mean: {part_rolling_std_bp.mean():.6f}, max: {part_rolling_std_bp.max():.6f}")
    print(f"FF rolling STD mean: {part_rolling_std_ff.mean():.6f}, max: {part_rolling_std_ff.max():.6f}")

    part_speedup = speedup_by_slope_smooth.iloc[start_idx:end_idx]
    mean_part_speedup = part_speedup.mean()
    print(f"Mean FF speedup over BP (part {i+1}): {mean_part_speedup:.2f}x")

# =========================
# Удалён старый код расчёта bp_reach и speedup по частям
# =========================


# Находим локальные максимумы сглаженного speedup по производным
from scipy.signal import find_peaks

peaks, _ = find_peaks(speedup_by_slope_smooth)
# сортируем по величине speedup
top_peaks_idx = peaks[np.argsort(speedup_by_slope_smooth.iloc[peaks])[::-1][:10]]

print("\n=== TOP10 PEAKS OF FF SPEEDUP BY DERIVATIVE (smoothed) ===")
for idx in top_peaks_idx:
    start_batch = b[max(0, idx-1)]
    end_batch = b[min(n-1, idx+1)]
    print(f"Peak at batch {b[idx]}, speedup={speedup_by_slope_smooth.iloc[idx]:.2f}x, period ~ {start_batch}-{end_batch}")
