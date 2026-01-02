import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
import os

import sentencepiece as spm

print("🚀 Запуск FF-обучения с мониторингом слоёв")

# -------------------------
# 1. Подготовка данных
# -------------------------
SEQ_LEN = 128

with open("text8", "r") as f:
    text = f.read()

# --- Subword tokenization (SentencePiece) ---
sp = spm.SentencePieceProcessor()
sp.load("sp_text8.model")

data = sp.encode(text)
vocab_size = sp.get_piece_size()

print("✅ Dataset загружен (Subword, SentencePiece)")
print("📦 Vocab size:", vocab_size)
print("🔍 Sample tokens:", sp.encode("this is a test")[:10])
print("🔁 Decode check:", sp.decode(sp.encode("this is a test")))

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
loader = DataLoader(dataset, batch_size=8, shuffle=True)
print("✅ DataLoader готов")

# -------------------------
# 2. Модель Transformer
# -------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False) 
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)  # (B, seq_len, d_model)
        out = x.transpose(0,1)  # (seq_len, B, d_model)
        for layer in self.layers:
            out = layer(out)
        out = self.ln(out.transpose(0,1))  # (B, seq_len, d_model)
        logits = self.fc_out(out)
        return logits

# -------------------------
# 3. Инициализация модели
# -------------------------
print("🔧 Инициализация модели...")
torch.manual_seed(42)
net_ff = SimpleTransformer(vocab_size)
NUM_LAYERS = len(net_ff.layers)

# Оптимизаторы для каждого слоя
optimizer_ff_layers = [
    optim.Adam(layer.parameters(), lr=1e-3) 
    for layer in net_ff.layers
]
optimizer_ff_head = optim.Adam(
    list(net_ff.ln.parameters()) + list(net_ff.fc_out.parameters()), 
    lr=1e-3
)

# Для отслеживания изменений весов
previous_weights = []
for layer in net_ff.layers:
    # Сохраняем копию весов каждого слоя
    layer_weights = []
    for param in layer.parameters():
        layer_weights.append(param.data.clone())
    previous_weights.append(layer_weights)

print(f"✅ Модель инициализирована ({NUM_LAYERS} слоёв)")

# -------------------------
# 4. Функция FF-обучения с отслеживанием
# -------------------------
def ff_step_with_tracking(model, x, y, optimizers_layers, optimizer_head, prev_weights, alpha=1.0):
    """
    FF обучение с отслеживанием loss и изменений весов по слоям
    
    Returns:
        losses_per_layer: list of loss values for each layer
        weight_changes: list of weight change magnitudes for each layer
    """
    losses_per_layer = []
    weight_changes = []
    
    # Глобальная ошибка
    with torch.no_grad():
        logits_final = model(x)
        global_loss = F.cross_entropy(logits_final.view(-1, vocab_size), y.view(-1))
    
    # Локальное обучение каждого слоя
    out = model.embed(x).transpose(0,1)  # (seq_len, B, d_model)
    for i, (layer, opt) in enumerate(zip(model.layers, optimizers_layers)):
        # ----- 1. Первый прогон (для обучения слоя) -----
        out = out.detach().requires_grad_(True)
        out_tmp = layer(out)
        h = out_tmp.transpose(0,1)  # (B, seq_len, d_model)

        h_norm = model.ln(h)
        temp_logits = model.fc_out(h_norm)

        raw_ce = F.cross_entropy(
            temp_logits.view(-1, vocab_size),
            y.view(-1)
        )
        local_loss = raw_ce * raw_ce.detach() * alpha

        losses_per_layer.append(raw_ce.item())

        opt.zero_grad()
        local_loss.backward()
        opt.step()

        # ----- 2. Подсчёт изменения весов -----
        total_change = 0.0
        num_params = 0
        for j, param in enumerate(layer.parameters()):
            change = torch.norm(param.data - prev_weights[i][j]).item()
            total_change += change
            num_params += 1
            prev_weights[i][j] = param.data.clone()

        avg_change = total_change / num_params if num_params > 0 else 0.0
        weight_changes.append(avg_change)

        # ----- 3. ВТОРОЙ прогон слоя (уже ОБУЧЕННОГО) -----
        with torch.no_grad():
            out = layer(out.detach())
    
    # Обновляем голову сети
    out = out.detach()
    out = model.ln(out.transpose(0,1))
    logits = model.fc_out(out)
    head_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer_head.zero_grad()
    head_loss.backward()
    optimizer_head.step()
    
    return losses_per_layer, weight_changes, global_loss.item()

# -------------------------
# 5. Логирование и графики
# -------------------------
# Структуры для хранения истории
layer_losses_history = [[] for _ in range(NUM_LAYERS)]  # loss каждого слоя
layer_weight_changes_history = [[] for _ in range(NUM_LAYERS)]  # изменения весов
global_losses = []

# CSV лог
csv_data = []

# Настройка графиков
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# График 1: Loss по слоям
colors = plt.cm.viridis(range(0, 256, 256 // NUM_LAYERS))
lines_loss = []
for i in range(NUM_LAYERS):
    line, = ax1.plot([], [], label=f"Layer {i+1}", linewidth=1.5, color=colors[i], alpha=0.8)
    lines_loss.append(line)

ax1.set_xlabel("Batch", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.set_title("FF Training: Loss по каждому слою", fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# График 2: Изменение весов по слоям
lines_weights = []
for i in range(NUM_LAYERS):
    line, = ax2.plot([], [], label=f"Layer {i+1}", linewidth=1.5, color=colors[i], alpha=0.8)
    lines_weights.append(line)

ax2.set_xlabel("Batch", fontsize=11)
ax2.set_ylabel("Weight Change (L2 norm)", fontsize=11)
ax2.set_title("FF Training: Изменение весов по каждому слою", fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 Графики инициализированы")

# -------------------------
# 6. Основной цикл обучения
# -------------------------
batch_counter = 0
for epoch in range(1):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
    for batch_idx, (x, y) in enumerate(loop):
        # FF обучение с отслеживанием
        losses, weight_changes, global_loss = ff_step_with_tracking(
            net_ff, x, y, optimizer_ff_layers, optimizer_ff_head, previous_weights, alpha=1.0
        )
        
        # Сохраняем историю
        for i in range(NUM_LAYERS):
            layer_losses_history[i].append(losses[i])
            layer_weight_changes_history[i].append(weight_changes[i])
        global_losses.append(global_loss)
        
        # Обновляем tqdm
        avg_loss = sum(losses) / len(losses)
        avg_weight_change = sum(weight_changes) / len(weight_changes)
        loop.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "global_loss": f"{global_loss:.4f}",
            "avg_Δw": f"{avg_weight_change:.6f}"
        })
        
        # Обновляем графики каждые 2 батча
        if batch_idx % 2 == 0:
            x_data = range(len(layer_losses_history[0]))
            
            # График 1: Loss
            for i, line in enumerate(lines_loss):
                line.set_data(x_data, layer_losses_history[i])
            ax1.relim()
            ax1.autoscale_view()
            
            # График 2: Weight changes
            for i, line in enumerate(lines_weights):
                line.set_data(x_data, layer_weight_changes_history[i])
            ax2.relim()
            ax2.autoscale_view()
            
            plt.pause(0.01)
        
        # Сохраняем в CSV каждые 10 шагов
        if batch_idx % 2 == 0:
            csv_row = {
                "batch": batch_idx,
                "global_loss": global_loss,
                "avg_layer_loss": avg_loss,
                "avg_weight_change": avg_weight_change
            }
        
            for i in range(NUM_LAYERS):
                csv_row[f"layer_{i+1}_loss"] = losses[i]
                csv_row[f"layer_{i+1}_weight_change"] = weight_changes[i]
        
            csv_data.append(csv_row)
        
            df = pd.DataFrame([csv_row])
            df.to_csv(
                "ff_training_results.csv",
                mode="a",
                index=False,
                header=not os.path.exists("ff_training_results.csv")
            )

plt.ioff()
plt.show()

# -------------------------
# 7. Сохранение результатов в CSV
# -------------------------
df = pd.DataFrame(csv_data)
df.to_csv("ff_training_results.csv", index=False)
print("\n✅ Обучение завершено!")
print(f"📁 Результаты сохранены в ff_training_results.csv")
print(f"📊 Записано {len(csv_data)} строк данных (каждые 10 батчей)")
print(f"\n📈 Финальная средняя loss: {df['avg_layer_loss'].iloc[-1]:.4f}")
print(f"⚖️  Финальное среднее изменение весов: {df['avg_weight_change'].iloc[-1]:.6f}")

# Статистика по слоям
print("\n" + "="*60)
print("📊 СТАТИСТИКА ПО СЛОЯМ (финальные значения):")
print("="*60)
for i in range(NUM_LAYERS):
    final_loss = df[f"layer_{i+1}_loss"].iloc[-1]
    final_weight_change = df[f"layer_{i+1}_weight_change"].iloc[-1]
    print(f"Layer {i+1}: Loss = {final_loss:.4f}, Weight Change = {final_weight_change:.6f}")