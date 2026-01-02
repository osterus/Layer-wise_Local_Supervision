import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Переключатель для 3D графика
# True  - использовать сглаженные и клиппированные данные
# False - использовать сырые данные
# --------------------------
use_smoothed_for_3d = True

# --------------------------
# 1. Чтение данных
# --------------------------
df = pd.read_csv("ffv2_test64_.csv")  # колонки: batch, loss_bp, loss_ff

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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --------------------------
# 5. 3D график: батч, ускорение, разница loss
# --------------------------
if use_smoothed_for_3d:
    X = batches
    Y = speedup       # уже сглаженный и клиппированный
    Z = smoothed_bp - smoothed_ff
else:
    X = batches
    # Мгновенный speedup по сырым данным
    slope_bp_raw = np.gradient(loss_bp)
    slope_ff_raw = np.gradient(loss_ff)
    speedup_raw = np.abs(slope_ff_raw) / (np.abs(slope_bp_raw) + 1e-9)
    Y = speedup_raw
    Z = loss_bp - loss_ff

# Цвет по ускорению для наглядности
colors = cm.viridis((Y - np.min(Y)) / (np.max(Y) - np.min(Y)))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, c=colors, s=20, alpha=0.8)

ax.set_xlabel('Batch')
ax.set_ylabel('FF Speedup (clipped & EMA smoothed)')
ax.set_zlabel('BP - FF Loss (smoothed)')
ax.set_title('3D view: Batch vs Speedup vs Loss Difference')

# Кубическое соотношение осей
ax.set_box_aspect([1,1,1])

plt.show()