import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import sentencepiece as spm
print("i am alive")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Используем MPS (M1 GPU acceleration)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Используем CUDA")
else:
    device = torch.device("cpu")
    print("⚠️ Используем CPU")

# -------------------------
# 1. Подготовка данных
# -------------------------
SEQ_LEN = 64

with open("text8", "r") as f:
    text = f.read()

# --- Subword tokenization (SentencePiece) ---
sp = spm.SentencePieceProcessor()
sp.load("sp_text8.model")

data = sp.encode(text)
vocab_size = sp.get_piece_size()

print("dataset s (subword)")
print("vocab size:", vocab_size)
print("sample:", sp.decode(sp.encode("this is a test")))

class TokenDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

dataset = TokenDataset(data, SEQ_LEN)
loader = DataLoader(dataset, batch_size=12, shuffle=True)
print("dataset done")

# -------------------------
# 2. Модель Transformer
# -------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # ОПТИМИЗАЦИЯ: Используем batch_first=True для лучшей производительности
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                batch_first=True,  # Изменено на True
                norm_first=True,  # Pre-LN более стабилен
                dim_feedforward=d_model * 4
            ) 
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, return_latents=False):
        # x: (B, seq_len)
        x = self.embed(x)  # (B, seq_len, d_model)
        latents = []
        out = x  # Теперь batch_first=True, не нужен transpose
        
        for layer in self.layers:
            out = layer(out)
            if return_latents:
                latents.append(out)
        
        out = self.ln(out)  # (B, seq_len, d_model)
        logits = self.fc_out(out)
        
        if return_latents:
            return logits, latents
        return logits

# -------------------------
# 3. Инициализация двух сетей одинаково
# -------------------------
print("init models")
torch.manual_seed(42)
net_bp = SimpleTransformer(vocab_size).to(device)
torch.manual_seed(42)
net_ff = SimpleTransformer(vocab_size).to(device)
print("start optimizer")
optimizer_bp = optim.Adam(net_bp.parameters(), lr=1e-3)
# Для FF создаём отдельные оптимизаторы для каждого слоя
optimizer_ff_layers = [
    optim.Adam(layer.parameters(), lr=1e-3) 
    for layer in net_ff.layers
]
optimizer_ff_head = optim.Adam(
    list(net_ff.ln.parameters()) + list(net_ff.fc_out.parameters()), 
    lr=1e-3
)
print("end optimizer")

# -------------------------
# 4. Функция локального обучения (УЛУЧШЕННАЯ ВЕРСИЯ)
# -------------------------
def l_step_v2(model, x, y, optimizers_layers, optimizer_head, alpha=1.0):
    """
    Forward-Forward Layer-wise Training v2 (OPTIMIZED)
    """
    
    # ===== ШАГ 1: Первый проход - получаем глобальную ошибку =====
    with torch.no_grad():
        logits_final = model(x)
        global_loss = F.cross_entropy(logits_final.view(-1, vocab_size), y.view(-1))
    
    # ===== ШАГ 2: Локальное обучение каждого слоя =====
    out = model.embed(x)  # (B, seq_len, d_model) - теперь batch_first

    for i, (layer, opt) in enumerate(zip(model.layers, optimizers_layers)):
        # ----- 1. Первый прогон (обучение слоя) -----
        out = out.detach().requires_grad_(True)
        out_tmp = layer(out)
        h = out_tmp

        h_norm = model.ln(h)
        temp_logits = model.fc_out(h_norm)

        local_loss = F.cross_entropy(
            temp_logits.view(-1, vocab_size),
            y.view(-1)
        )
        local_loss = local_loss * local_loss.detach() * alpha

        opt.zero_grad()
        local_loss.backward()
        opt.step()

        # ----- 2. Второй прогон слоя (уже обученного) -----
        with torch.no_grad():
            out = layer(out.detach())
    
    # ===== ШАГ 3: Обновляем голову сети =====
    out = out.detach()
    out = model.ln(out)  # (B, seq_len, d_model)
    logits = model.fc_out(out)
    head_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer_head.zero_grad()
    head_loss.backward()
    optimizer_head.step()
    
    return global_loss.item()

# -------------------------
# 5. Логирование и график в реальном времени
# -------------------------
losses_bp = []
losses_ff = []

plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line1, = ax.plot(losses_bp, label="Backprop", linewidth=2, color='blue')
line2, = ax.plot(losses_ff, label="FF-Layer v2 (Cross-Entropy)", linewidth=2, color='red')
ax.set_ylim(0, 10)
ax.set_xlabel("Batch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Backprop vs FF-Layer: Training Comparison", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

csv_data = []

# -------------------------
# 6. Основной цикл обучения
# -------------------------
if hasattr(torch, 'compile') and device.type != "mps":
    print("🔥 Компилируем модели для ускорения...")
    try:
        net_bp = torch.compile(net_bp)
        net_ff = torch.compile(net_ff)
    except:
        print("⚠️ Компиляция недоступна, продолжаем без неё")


for epoch in range(1):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        # ===== Backprop сеть =====
        optimizer_bp.zero_grad()
        logits_bp = net_bp(x)
        loss_bp = F.cross_entropy(logits_bp.view(-1, vocab_size), y.view(-1))
        loss_bp.backward()
        optimizer_bp.step()
        
        # ===== FF сеть (версия 2 с cross-entropy) =====
        loss_ff = l_step_v2(net_ff, x, y, optimizer_ff_layers, optimizer_ff_head, alpha=1.0)
        
        # Логирование
        losses_bp.append(loss_bp.item())
        losses_ff.append(loss_ff)
        
        # Обновляем tqdm
        diff = loss_ff - loss_bp.item()
        loop.set_postfix({
            "loss_bp": f"{loss_bp.item():.4f}",
            "loss_ff": f"{loss_ff:.4f}",
            "diff": f"{diff:+.4f}"
        })
        
        # Обновляем график каждые 50 батчей
        if batch_idx % 2 == 0:
            line1.set_ydata(losses_bp)
            line1.set_xdata(range(len(losses_bp)))
            line2.set_ydata(losses_ff)
            line2.set_xdata(range(len(losses_ff)))
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        if batch_idx % 2 == 0:
            csv_row = {
                "batch": len(losses_bp),
                "loss_bp": loss_bp.item(),
                "loss_ff": loss_ff,
                "diff": loss_ff - loss_bp.item()
            }
            csv_data.append(csv_row)
            df = pd.DataFrame([csv_row])
            df.to_csv(
                "ffv2_test64_o.csv",
                mode="a",
                index=False,
                header=not os.path.exists("ffv2_test64_o.csv")
            )

plt.ioff()
plt.show()

# -------------------------
# 7. Сохранение лосса в CSV
# -------------------------
df_batches = pd.DataFrame(csv_data)
df_batches.to_csv("losses_batches.csv", index=False)
print("📁 Batch-лоссы сохранены в losses_batches.csv")

df = pd.DataFrame({
    "backprop": losses_bp, 
    "ff_layer_v2": losses_ff,
    "diff": [ff - bp for bp, ff in zip(losses_bp, losses_ff)]
})
df.to_csv("losses_v2.csv", index=False)
print("\n✅ Обучение завершено! Результаты сохранены в losses_v2.csv")
print(f"\n📊 Финальная разница: {df['diff'].iloc[-1]:.4f}")
print(f"   Средняя разница: {df['diff'].mean():.4f}")