import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Чтение данных
# --------------------------
df = pd.read_csv("ffv2_test64_o.csv")  # колонки: batch, loss_bp, loss_ff

loss_bp = df['loss_bp'].values
loss_ff = df['loss_ff'].values
batches = df['batch'].values

# --------------------------
# 2. Сглаживание через rolling mean
# --------------------------
window_smooth = 200

smoothed_bp = pd.Series(loss_bp).rolling(window=window_smooth, min_periods=1).mean().values
smoothed_ff = pd.Series(loss_ff).rolling(window=window_smooth, min_periods=1).mean().values

# --------------------------
# 3. Расчёт speedup через производные сглаженных кривых
# --------------------------
epsilon = 1e-9

slope_bp = np.gradient(smoothed_bp)
slope_ff = np.gradient(smoothed_ff)

# Модуль наклона для корректной интерпретации ускорения
abs_slope_bp = np.abs(slope_bp)
abs_slope_ff = np.abs(slope_ff)

# Расчёт моментного ускорения FF относительно BP
speedup = abs_slope_ff / (abs_slope_bp + epsilon)

#
# --------------------------
# Клинг и EMA сглаживание
# --------------------------
# Клиппинг: все значения speedup > 10 обрезаем до 10
speedup_clipped = np.minimum(speedup, 10)

# EMA сглаживание с span = 20
speedup_smooth = pd.Series(speedup_clipped).ewm(span=120, adjust=False).mean().values

mean_speedup = np.nanmean(speedup_smooth)
print(f"Mean FF speedup over BP (clipped & EMA smoothed): {mean_speedup:.2f}x")

# Используем сглаженный для графика
speedup = speedup_smooth

# --------------------------
# 4. Визуализация
# --------------------------
plt.figure(figsize=(12,5))
plt.plot(batches, smoothed_bp, label="BP (smoothed)")
plt.plot(batches, smoothed_ff, label="FF (smoothed)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Smoothed Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(batches, speedup, label="Speedup FF over BP")
plt.xlabel("Batch")
plt.ylabel("Speedup")
plt.title("FF Speedup over BP per Batch")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 5. График: ускорение vs разница loss
# --------------------------
diff = smoothed_bp - smoothed_ff

plt.figure(figsize=(8,6))
plt.scatter(diff, speedup, c='purple', alpha=0.6)
plt.xlabel('BP - FF Loss (smoothed)')
plt.ylabel('FF Speedup (clipped & EMA smoothed)')
plt.title('FF Speedup vs Loss Difference per Batch')
plt.grid(True)
plt.show()