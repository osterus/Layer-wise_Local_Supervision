import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. ЗАГРУЗКА
# =========================
df = pd.read_csv("ffv2_test.csv")

b = df["batch"].values
loss_bp = df["loss_bp"].values
loss_ff = df["loss_ff"].values
diff = loss_ff - loss_bp

# =========================
# 2. БАЗОВАЯ СТАТИСТИКА
# =========================
def stats(name, x):
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "final": x[-1]
    }

print("\n=== BASIC STATS ===")
print("BP:", stats("bp", loss_bp))
print("FF:", stats("ff", loss_ff))

# =========================
# 3. ХАОТИЧНОСТЬ
# =========================
window = max(5, len(df)//30)

std_bp_roll = pd.Series(loss_bp).rolling(window).std()
std_ff_roll = pd.Series(loss_ff).rolling(window).std()

# производная (шум)
d_bp = np.diff(loss_bp)
d_ff = np.diff(loss_ff)

# =========================
# 4. AUC (ПЛОЩАДЬ ПОД КРИВОЙ)
# =========================
auc_bp = np.trapz(loss_bp)
auc_ff = np.trapz(loss_ff)

print("\n=== LEARNING EFFICIENCY ===")
print(f"AUC BP: {auc_bp:.2f}")
print(f"AUC FF: {auc_ff:.2f}")
print(f"FF better by {(auc_bp/auc_ff):.2f}x (lower is better)")

# =========================
# 5. ГРАФИКИ
# =========================

# ---- 1. LOSS ----
plt.figure()
plt.plot(b, loss_bp, label="BP")
plt.plot(b, loss_ff, label="FF")
plt.title("Loss (linear)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# ---- 2. LOG LOSS ----
plt.figure()
plt.semilogy(b, loss_bp, label="BP")
plt.semilogy(b, loss_ff, label="FF")
plt.title("Loss (log scale)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# ---- 3. DIFF ----
plt.figure()
plt.plot(b, diff)
plt.axhline(0, linestyle="--")
plt.title("Loss difference (FF - BP)")
plt.xlabel("Batch")
plt.ylabel("ΔLoss")
plt.grid()
plt.show()

# ---- 4. CHAOS ----
plt.figure()
plt.plot(b, std_bp_roll, label="BP chaos")
plt.plot(b, std_ff_roll, label="FF chaos")
plt.title(f"Rolling STD (window={window})")
plt.xlabel("Batch")
plt.ylabel("STD")
plt.legend()
plt.grid()
plt.show()

# ---- 5. GRADIENT NOISE ----
plt.figure()
plt.plot(d_bp, label="BP dLoss")
plt.plot(d_ff, label="FF dLoss")
plt.title("Loss derivative (noise)")
plt.xlabel("Batch")
plt.ylabel("ΔLoss")
plt.legend()
plt.grid()
plt.show()

# =========================
# 6. FF ОПЕРЕЖЕНИЕ BP
# =========================
bp_reach = []

for i, lf in enumerate(loss_ff):
    idx = np.where(loss_bp <= lf)[0]
    if len(idx) == 0:
        bp_reach.append(np.nan)
    else:
        bp_reach.append(idx[0])

bp_reach = np.array(bp_reach)

# speedup
speedup = bp_reach / b
valid = ~np.isnan(speedup)
mean_speedup = np.nanmean(speedup)

print("\n=== SPEEDUP ===")
print(f"FF faster than BP by ~{mean_speedup:.2f}x")

# ---- 6.1 график достижения ----
plt.figure()
plt.plot(b, bp_reach)
plt.plot(b, b, linestyle="--", label="Equal speed")
plt.title("How many BP batches to reach FF loss")
plt.xlabel("FF batch")
plt.ylabel("BP batch")
plt.legend()
plt.grid()
plt.show()

# ---- 6.2 speedup ----
plt.figure()
plt.plot(b[valid], speedup[valid])
plt.axhline(1, linestyle="--")
plt.title("FF speedup over BP")
plt.xlabel("FF batch")
plt.ylabel("Speedup (BP/FF)")
plt.grid()
plt.show()
