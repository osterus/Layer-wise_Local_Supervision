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

dataset = TokenDataset(data, SEQ_LEN)
loader = DataLoader(dataset, batch_size=12, shuffle=True)
print(f"dataset done ({len(dataset)} samples)")

# -------------------------
# 2. Модель Transformer
# -------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, pretrained_embedding=None):
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

class RecursiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=4, num_layers=2, pretrained_embedding=None, max_steps=40, confidence_threshold=0.9):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        if pretrained_embedding is not None:
            self.embed.weight = nn.Parameter(pretrained_embedding)
        self.embed.weight.requires_grad_(False)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                batch_first=True, 
                norm_first=True,
                dim_feedforward=d_model * 4
            ) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold

    def forward(self, x):
        # x: (B, seq_len)
        x = self.embed(x)
        out = x
        step = 0
        done_mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)

        while step < self.max_steps:
            for layer in self.layers:
                out = self.ln(out)
                out = layer(out)
            logits = self.fc_out(self.ln(out))
            probs = F.softmax(logits, dim=-1)
            confidence, pred = probs.max(dim=-1)
            done_mask |= (confidence > self.confidence_threshold)
            if done_mask.all():
                break
            step += 1

        return logits

# -------------------------
# 3. Training Recursive Transformer
# -------------------------
model = RecursiveTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, pretrained_embedding=embedding_tensor)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 500

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

        # формируем окно: текущий слой + следующие два
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

# -------------------------
# 4. Second model with LLS training
# -------------------------
model_lls = SimpleTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, pretrained_embedding=embedding_tensor)
model_lls.to(device)

# Создаём отдельные оптимизаторы для каждого слоя
optimizers_layers = [optim.Adam(layer.parameters(), lr=1e-3) for layer in model_lls.layers]
optimizer_head = optim.Adam(list(model_lls.fc_out.parameters()) + list(model_lls.ln.parameters()), lr=1e-3)

plt.ion()
fig, ax = plt.subplots(figsize=(8,5))
line1, = ax.plot([], [], marker='o', label='Recursive Transformer')
line2, = ax.plot([], [], marker='x', label='LLS Transformer')
ax.set_title("Training Loss - Real-time")
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

losses_recursive = []
losses_lls = []
batch_count = 0

for epoch in range(epochs):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        # ---- Train Recursive Transformer ----
        optimizer.zero_grad()
        logits_rec = model(x)
        loss_rec = criterion(logits_rec.view(-1, vocab_size), y.view(-1))
        loss_rec.backward()
        optimizer.step()
        losses_recursive.append(loss_rec.item())

        # ---- Train LLS Transformer in real-time (layer by layer) ----
        out = model_lls.embed(x)
        num_layers = len(model_lls.layers)
        for i in range(num_layers):
            # prepare window: current layer + next two
            h_window = out.detach().requires_grad_(True)
            h_layers = []
            for j in range(3):
                if i + j < num_layers:
                    h_window = model_lls.layers[i + j](h_window)
                    h_layers.append(h_window)
            
            total_loss = 0
            for h_layer in h_layers:
                h_norm = model_lls.ln(h_layer)
                logits = model_lls.fc_out(h_norm)
                total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            opt_layers = [optimizers_layers[i + j] for j in range(len(h_layers))]
            for opt in opt_layers:
                opt.zero_grad()
            total_loss.backward()
            for opt in opt_layers:
                opt.step()
            
            with torch.no_grad():
                out = h_window.clone()
        
        # update head
        out = model_lls.ln(out)
        logits_head = model_lls.fc_out(out)
        optimizer_head.zero_grad()
        head_loss_val = F.cross_entropy(logits_head.view(-1, vocab_size), y.view(-1))
        head_loss_val.backward()
        optimizer_head.step()
        losses_lls.append(head_loss_val.item())

        batch_count += 1

        # Update progress bar with both losses
        loop.set_postfix(rec_loss=loss_rec.item(), lls_loss=head_loss_val)

        # Update real-time plot every 2 batches
        if batch_count % 2 == 0:
            line1.set_xdata(range(len(losses_recursive)))
            line1.set_ydata(losses_recursive)
            line2.set_xdata(range(len(losses_lls)))
            line2.set_ydata(losses_lls)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

plt.ioff()
plt.show()