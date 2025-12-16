import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 1. Подготовка данных
# -------------------------
SEQ_LEN = 164

with open("text8", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

data = [stoi[ch] for ch in text]

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

dataset = CharDataset(data, SEQ_LEN)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------
# 2. Модель Transformer
# -------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, return_latents=False):
        # x: (B, seq_len)
        x = self.embed(x)  # (B, seq_len, d_model)
        latents = []
        out = x.transpose(0,1)  # (seq_len, B, d_model) для Transformer
        for layer in self.layers:
            out = layer(out)
            latents.append(out.transpose(0,1))  # сохраняем латенты перед fc
        out = self.ln(out.transpose(0,1))  # (B, seq_len, d_model)
        logits = self.fc_out(out)
        if return_latents:
            return logits, latents
        return logits

# -------------------------
# 3. Инициализация двух сетей одинаково
# -------------------------
torch.manual_seed(42)
net_bp = SimpleTransformer(vocab_size)
torch.manual_seed(42)
net_ff = SimpleTransformer(vocab_size)

optimizer_bp = optim.Adam(net_bp.parameters(), lr=1e-3)
optimizer_ff = optim.Adam(net_ff.parameters(), lr=1e-3)  # локальные обновления будем применять сами

# -------------------------
# 4. Функция локального FF-обучения
# -------------------------
def ff_step(model, x, y, alpha=1.0):
    # Forward 1: получить итоговый выход
    logits, latents = model(x, return_latents=True)
    loss_global = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    # Forward 2: по слоям
    input_layer = model.embed(x)
    out = input_layer.transpose(0,1)
    
    for i, layer in enumerate(model.layers):
        out = layer(out)
        h = out.transpose(0,1)  # (B, seq_len, d_model)
        # локальная цель: приблизиться к логитам финального выхода
        target = logits.detach()  # фиксируем глобальный выход
        local_loss = F.mse_loss(h, target.detach()) * alpha
        layer.zero_grad()
        local_loss.backward()
        # прямое локальное обновление весов слоя
        for p in layer.parameters():
            p.data -= 1e-3 * p.grad
        out = out.detach().requires_grad_(True)  # отсоединяем для следующего слоя

    return loss_global.item()

# -------------------------
# 5. Логирование и график в реальном времени
# -------------------------
losses_bp = []
losses_ff = []

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot(losses_bp, label="Backprop")
line2, = ax.plot(losses_ff, label="FF-Layer")
ax.set_ylim(0, 5)
ax.legend()
plt.show()

# -------------------------
# 6. Основной цикл обучения
# -------------------------
for epoch in range(1):
    for batch_idx, (x, y) in enumerate(loader):
        # Backprop сеть
        optimizer_bp.zero_grad()
        logits_bp = net_bp(x)
        loss_bp = F.cross_entropy(logits_bp.view(-1, vocab_size), y.view(-1))
        loss_bp.backward()
        optimizer_bp.step()
        
        # FF сеть
        loss_ff = ff_step(net_ff, x, y)
        
        # Логирование
        losses_bp.append(loss_bp.item())
        losses_ff.append(loss_ff)
        
        if batch_idx % 50 == 0:
            line1.set_ydata(losses_bp)
            line1.set_xdata(range(len(losses_bp)))
            line2.set_ydata(losses_ff)
            line2.set_xdata(range(len(losses_ff)))
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

# -------------------------
# 7. Сохранение лосса в CSV
# -------------------------
df = pd.DataFrame({"backprop": losses_bp, "ff_layer": losses_ff})
df.to_csv("losses.csv", index=False)
