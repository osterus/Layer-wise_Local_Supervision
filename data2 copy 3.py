import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Чтение данных
# --------------------------
df = pd.read_csv("ow128.csv")  # колонки: batch, loss_bp, loss_ff, loss_ff_all

loss_bp = df['loss_bp'].values
loss_ff = df['loss_lls'].values
loss_ff_all = df['loss_lls_all'].values
batches = df['batch'].values

# --------------------------
# 2. Сглаживание через rolling mean
# --------------------------
window_smooth = 200

smoothed_bp = pd.Series(loss_bp).rolling(window=window_smooth, min_periods=1).mean().values
smoothed_ff = pd.Series(loss_ff).rolling(window=window_smooth, min_periods=1).mean().values
smoothed_ff_all = pd.Series(loss_ff_all).rolling(window=window_smooth, min_periods=1).mean().values

# --------------------------
# 3. Расчёт speedup через производные сглаженных кривых (без модуля)
# --------------------------
epsilon = 1e-9

slope_bp = np.gradient(smoothed_bp)
slope_ff = np.gradient(smoothed_ff)
slope_ff_all = np.gradient(smoothed_ff_all)

# Ускорение с учётом направления изменения
speedup_ff = slope_ff / (slope_bp + epsilon)
speedup_ff_all = slope_ff_all / (slope_bp + epsilon)

# Клинг: ограничиваем слишком большие значения для визуализации
speedup_ff_clipped = np.clip(speedup_ff, -10, 10)
speedup_ff_all_clipped = np.clip(speedup_ff_all, -10, 10)

# EMA сглаживание с span = 120
speedup_ff_smooth = pd.Series(speedup_ff_clipped).ewm(span=120, adjust=False).mean().values
speedup_ff_all_smooth = pd.Series(speedup_ff_all_clipped).ewm(span=120, adjust=False).mean().values

mean_speedup_ff = np.nanmean(speedup_ff_smooth)
mean_speedup_ff_all = np.nanmean(speedup_ff_all_smooth)
print(f"Mean FF speedup over BP (with sign, clipped & EMA smoothed): {mean_speedup_ff:.2f}x")
print(f"Mean FF_all speedup over BP (with sign, clipped & EMA smoothed): {mean_speedup_ff_all:.2f}x")

# --------------------------
# 4. Визуализация кривых loss
# --------------------------
plt.figure(figsize=(12,5))
plt.plot(batches, smoothed_bp, label="BP (smoothed)")
plt.plot(batches, smoothed_ff, label="FF (smoothed)")
plt.plot(batches, smoothed_ff_all, label="FF_all (smoothed)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Smoothed Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 5. Визуализация ускорения FF
# --------------------------
plt.figure(figsize=(12,5))
plt.plot(batches, speedup_ff_smooth, label="Speedup FF over BP (with sign)")
plt.xlabel("Batch")
plt.ylabel("Speedup")
plt.title("FF Speedup over BP per Batch (with sign)")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 5b. Визуализация ускорения FF_all
# --------------------------
plt.figure(figsize=(12,5))
plt.plot(batches, speedup_ff_all_smooth, label="Speedup FF_all over BP (with sign)", color='orange')
plt.xlabel("Batch")
plt.ylabel("Speedup")
plt.title("FF_all Speedup over BP per Batch (with sign)")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 6. График: ускорение vs разница loss
# --------------------------
diff_ff = smoothed_bp - smoothed_ff
diff_ff_all = smoothed_bp - smoothed_ff_all

plt.figure(figsize=(8,6))

# scatter
plt.scatter(diff_ff, speedup_ff_smooth, c='purple', alpha=0.6, label='FF points')
plt.scatter(diff_ff_all, speedup_ff_all_smooth, c='orange', alpha=0.6, label='FF_all points')

# линия по батчам
plt.plot(diff_ff, speedup_ff_smooth, color='yellow', alpha=1.0, linewidth=3, label='FF progression')
plt.plot(diff_ff_all, speedup_ff_all_smooth, color='red', alpha=1.0, linewidth=2, label='FF_all progression')

# зеленая точка — начало
plt.scatter(diff_ff[0], speedup_ff_smooth[0], color='green', s=100, label='Start FF')
plt.scatter(diff_ff_all[0], speedup_ff_all_smooth[0], color='blue', s=100, label='Start FF_all')

# красная точка — конец
plt.scatter(diff_ff[-1], speedup_ff_smooth[-1], color='red', s=100, label='End FF')
plt.scatter(diff_ff_all[-1], speedup_ff_all_smooth[-1], color='black', s=100, label='End FF_all')

plt.xlabel('BP - FF / FF_all Loss (smoothed)')
plt.ylabel('Speedup (clipped & EMA smoothed)')
plt.title('FF / FF_all Speedup vs Loss Difference per Batch')
plt.grid(True)
plt.legend()
plt.show()