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
from gensim.models import Word2Vec
import numpy as np
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

# --- Предобучение эмбеддинга или загрузка ---
embedding_path = "embedding.pt"
embedding_dim = 128

if not os.path.exists(embedding_path):
    print("Предобучение эмбеддинга с помощью Word2Vec...")
    # Подготовка данных для Word2Vec: разбиваем data на последовательности для обучения
    # Используем скользящее окно по токенам длиной 5 для контекста
    sentences = []
    window_size = 5
    data_str = [str(token) for token in data]
    for i in tqdm(range(len(data_str) - window_size + 1)):
        sentences.append(data_str[i:i+window_size])
    # Обучаем Word2Vec
    w2v_model = Word2Vec(sentences, vector_size=embedding_dim, window=window_size, min_count=1, workers=4, sg=1, epochs=5)
    # Создаём матрицу эмбеддингов для всех токенов
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for idx in tqdm(range(vocab_size)):
        token_str = str(idx)
        if token_str in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[token_str]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    embedding_tensor = torch.tensor(embedding_matrix)
    torch.save(embedding_tensor, embedding_path)
    print("Эмбеддинг сохранён в embedding.pt")
else:
    print("Загрузка предобученного эмбеддинга из embedding.pt")
    embedding_tensor = torch.load(embedding_path)

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

# Ограничиваем датасет 100 случайными примерами для экспресс-теста
random_indices = np.random.choice(len(data) - SEQ_LEN, size=100, replace=False)
subset_data = [data[i:i+SEQ_LEN+1] for i in random_indices]

class SmallTokenDataset(Dataset):
    def __init__(self, subset_data):
        self.subset_data = subset_data
    def __len__(self):
        return len(self.subset_data)
    def __getitem__(self, idx):
        x = torch.tensor(self.subset_data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.subset_data[idx][1:], dtype=torch.long)
        return x, y

dataset = SmallTokenDataset(subset_data)
loader = DataLoader(dataset, batch_size=12, shuffle=True)
print("dataset done (100 samples)")

# -------------------------
# 2. Модель Transformer
# -------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=64, pretrained_embedding=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        if pretrained_embedding is not None:
            self.embed.weight = nn.Parameter(pretrained_embedding)
        self.embed.weight.requires_grad_(False)
        
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
        # Обучаемые alpha для каждого слоя (локально-глобальный баланс)
        self.layer_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in self.layers
        ])
    
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
# 3. Инициализация трёх сетей одинаково, включая обучаемый embedding
# -------------------------
print("init models")
torch.manual_seed(42)
net_bp = SimpleTransformer(vocab_size, pretrained_embedding=embedding_tensor).to(device)
torch.manual_seed(42)
net_lls = SimpleTransformer(vocab_size, pretrained_embedding=embedding_tensor).to(device)
torch.manual_seed(42)
net_lls_all = SimpleTransformer(vocab_size, pretrained_embedding=embedding_tensor).to(device)
print("start optimizer")

optimizer_bp = optim.Adam(net_bp.parameters(), lr=1e-3)

# Для LLS с окном: отдельные оптимизаторы для каждого слоя и embedding
optimizer_lls_layers = [
    optim.Adam(
        list(layer.parameters()) + [net_lls.layer_alphas[i]],
        lr=1e-3
    )
    for i, layer in enumerate(net_lls.layers)
]
optimizer_lls_head = optim.Adam(
    list(net_lls.ln.parameters()) + list(net_lls.fc_out.parameters()), 
    lr=1e-3
)

# Для LLS с одной головой (вся сеть)
optimizer_lls_all = optim.Adam(net_lls_all.parameters(), lr=1e-3)

print("end optimizer")

# -------------------------
# 4. Новые функции локального обучения
# -------------------------
#
## Задаём фиксированные приоритеты для каждого слоя (НЕ обучаемые)
layer_priority_weights = [0.75 for _ in net_lls.layers]
net_lls.layer_priorities = layer_priority_weights  # for logging compatibility

def lls_step_window(model, x, y, optimizers_layers, optimizer_head):
    """
    LLS: объединение текущего слоя и следующих двух слоёв в один общий градиент
    (без штрафов/наград, просто суммируем локальные лоссы)
    """
    out = model.embed(x)
    layer_losses = []
    layer_priorities_values = []

    num_layers = len(model.layers)
    for i in range(num_layers):
        # проход через предыдущие слои без градиента
        h = out.detach()
        #for wl in model.layers[:i]:
        #    with torch.no_grad():
        #        h = wl(h)

        # формируем окно: текущий слой + следующие два
        window_layers = []
        h_window = h.detach().requires_grad_(True)
        h_layers = []
        for j in range(4):
            if i + j < num_layers:
                h_window = model.layers[i + j](h_window)
                h_layers.append(h_window)

        # считаем локальные лоссы и суммируем
        total_loss = 0
        for h_layer in h_layers:
            h_norm = model.ln(h_layer)
            logits = model.fc_out(h_norm)
            total_loss = total_loss + F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # backward один раз для всех слоёв окна
        opt_layers = [optimizers_layers[i + j] for j in range(len(h_layers))]
        for opt in opt_layers:
            opt.zero_grad()
        total_loss.backward()
        for opt in opt_layers:
            opt.step()

        # Для логирования: средний локальный лосс по окну (или первый, если хотите)
        if len(h_layers) > 0:
            h_norm = model.ln(h_layers[0])
            logits = model.fc_out(h_norm)
            local_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            layer_losses.append(local_loss.item())
        else:
            layer_losses.append(0.0)
        layer_priorities_values.append(1.0)

        # обновление out для следующего слоя
        with torch.no_grad():
            out = h_window.clone()

    # обновление головы
    out = out.detach()
    out = model.ln(out)
    logits = model.fc_out(out)
    head_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer_head.zero_grad()
    head_loss.backward()
    optimizer_head.step()

    return head_loss.item(), layer_losses, layer_priorities_values

def lls_step_all(model, x, y, optimizer):
    """
    Layer-wise Local Supervision (LLS) обучение всей сети с одной головой,
    включая обучаемый embedding
    """
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

def l_step_bp(model, x, y, optimizer):
    """
    Стандартное backprop обучение всей сети
    """
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


# -------------------------
# 5. Логирование и график в реальном времени
# -------------------------
losses_bp = []
losses_lls = []
losses_lls_all = []

plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line1, = ax.plot(losses_bp, label="Backprop", linewidth=2, color='blue')
line2, = ax.plot(losses_lls, label="LLS-Layer v2 (Cross-Entropy)", linewidth=2, color='red')
line3, = ax.plot(losses_lls_all, label="LLS-All Layers (One Head)", linewidth=2, color='green')
ax.set_ylim(0, 10)
ax.set_xlabel("Batch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Backprop vs LLS-Layer vs LLS-All: Training Comparison", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

csv_data = []
csv_metrics_data = []

# -------------------------
# 6. Основной цикл обучения
# -------------------------
if hasattr(torch, 'compile') and device.type != "mps":
    print("🔥 Компилируем модели для ускорения...")
    try:
        net_bp = torch.compile(net_bp)
        net_lls = torch.compile(net_lls)
        net_lls_all = torch.compile(net_lls_all)
    except:
        print("⚠️ Компиляция недоступна, продолжаем без неё")


for epoch in range(500):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        # ===== Backprop сеть =====
        loss_bp = l_step_bp(net_bp, x, y, optimizer_bp)
        
        # ===== LLS сеть (версия 2 с cross-entropy, окно) =====
        loss_lls, layer_losses, layer_priorities = lls_step_window(
            net_lls, x, y,
            optimizer_lls_layers,
            optimizer_lls_head,
        )

        # ===== LLS сеть с одной головой (вся сеть) =====
        loss_lls_all = lls_step_all(net_lls_all, x, y, optimizer_lls_all)
        
        # Вычисляем similarity активаций соседних слоёв (среднее косинусное сходство)
        with torch.no_grad():
            _, latents = net_lls(x, return_latents=True)
            _, latents_bp = net_bp(x, return_latents=True)
            similarities = []
            for i in range(len(latents)-1):
                a = latents[i].flatten(1)
                b = latents[i+1].flatten(1)
                a_norm = F.normalize(a, dim=1)
                b_norm = F.normalize(b, dim=1)
                sim = (a_norm * b_norm).sum(dim=1).mean().item()
                similarities.append(sim)
            activation_similarity = np.mean(similarities) if similarities else 0.0

            # Новое: косинусное сходство латентов LLS с латентами BP для каждого слоя
            similarities_bp = []
            for i in range(len(latents)):
                a = latents[i].flatten(1)
                b = latents_bp[i].flatten(1)
                a_norm = F.normalize(a, dim=1)
                b_norm = F.normalize(b, dim=1)
                sim_bp = (a_norm * b_norm).sum(dim=1).mean().item()
                similarities_bp.append(sim_bp)
            activation_similarity_bp = np.mean(similarities_bp) if similarities_bp else 0.0

            # BP: self-similarity across layers
            similarities_bp_self = []
            for i in range(len(latents_bp)-1):
                a = latents_bp[i].flatten(1)
                b = latents_bp[i+1].flatten(1)
                a_norm = F.normalize(a, dim=1)
                b_norm = F.normalize(b, dim=1)
                sim_self = (a_norm * b_norm).sum(dim=1).mean().item()
                similarities_bp_self.append(sim_self)
            activation_similarity_bp_self = np.mean(similarities_bp_self) if similarities_bp_self else 0.0

        # Логирование
        losses_bp.append(loss_bp)
        losses_lls.append(loss_lls)
        losses_lls_all.append(loss_lls_all)
        
        # Обновляем tqdm
        loop.set_postfix({
            "loss_bp": f"{loss_bp:.4f}",
            "loss_lls": f"{loss_lls:.4f}",
            "loss_lls_all": f"{loss_lls_all:.4f}",
            "activation_sim": f"{activation_similarity:.4f}"
        })
        
        # Обновляем график каждые 50 батчей
        if batch_idx % 2 == 0:
            line1.set_ydata(losses_bp)
            line1.set_xdata(range(len(losses_bp)))
            line2.set_ydata(losses_lls)
            line2.set_xdata(range(len(losses_lls)))
            line3.set_ydata(losses_lls_all)
            line3.set_xdata(range(len(losses_lls_all)))
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        if batch_idx % 2 == 0:
            csv_row = {
                "batch": len(losses_bp),
                "loss_bp": loss_bp,
                "loss_lls": loss_lls,
                "loss_lls_all": loss_lls_all,
                "diff": loss_lls - loss_bp
            }
            csv_data.append(csv_row)
            df = pd.DataFrame([csv_row])
            df.to_csv(
                "ow1.csv",
                mode="a",
                index=False,
                header=not os.path.exists("ow1.csv")
            )
            # Сохраняем метрики в отдельный CSV
            metrics_row = {
                "batch": len(losses_bp),
                "activation_similarity": activation_similarity,
                "activation_similarity_bp": activation_similarity_bp,
                "activation_similarity_bp_self": activation_similarity_bp_self,
            }
            # Добавляем priority каждого слоя и локальный лосс каждого слоя
            for idx, (priority, l_loss) in enumerate(zip(layer_priorities, layer_losses)):
                metrics_row[f"priority_layer_{idx}"] = priority
                metrics_row[f"local_loss_layer_{idx}"] = l_loss
            csv_metrics_data.append(metrics_row)
            df_metrics = pd.DataFrame([metrics_row])
            df_metrics.to_csv(
                "ics.csv",
                mode="a",
                index=False,
                header=not os.path.exists("ics.csv")
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
    "lls_layer_v2": losses_lls,
    "lls_all": losses_lls_all,
    "diff": [lls - bp for bp, lls in zip(losses_bp, losses_lls)]
})
df.to_csv("losses_v2.csv", index=False)
print("\n✅ Обучение завершено! Результаты сохранены в losses_v2.csv")
print(f"\n📊 Финальная разница: {df['diff'].iloc[-1]:.4f}")
print(f"   Средняя разница: {df['diff'].mean():.4f}")