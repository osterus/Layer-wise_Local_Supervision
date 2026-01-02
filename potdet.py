import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend для ускорения
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import sentencepiece as spm
from torch.cuda.amp import autocast, GradScaler
import multiprocessing as mp

# -------------------------
# 0. Настройка устройства (M1 MPS)
# -------------------------
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Используется Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Используется CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Используется CPU")
    return device

# -------------------------
# 1. Dataset класс
# -------------------------
class TokenDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        # Предвычисляем длину для ускорения
        self._len = len(self.data) - self.seq_len
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        # Используем slice для быстрого доступа
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# -------------------------
# 2. ОПТИМИЗИРОВАННАЯ Модель Transformer
# -------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Используем nn.Sequential для возможной оптимизации
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=False,
                norm_first=False  # Post-norm для стабильности
            ) 
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Инициализация весов для лучшей сходимости
        self._init_weights()
    
    def _init_weights(self):
        # Xavier/Glorot инициализация
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x):
        x = self.embed(x)  # (B, seq_len, d_model)
        out = x.transpose(0, 1)  # (seq_len, B, d_model)
        
        for layer in self.layers:
            out = layer(out)
        
        out = self.ln(out.transpose(0, 1))  # (B, seq_len, d_model)
        logits = self.fc_out(out)
        return logits

# -------------------------
# 3. Вспомогательные функции
# -------------------------
@torch.no_grad()  # Изменено с @torch.inference_mode() на @torch.no_grad()
def compute_weight_changes(layer, prev_weights_layer):
    """Вычисление изменений весов (вынесено для оптимизации)"""
    total_change = 0.0
    num_params = 0
    
    for j, param in enumerate(layer.parameters()):
        # Переносим на CPU только для сравнения
        param_cpu = param.data.cpu()
        change = torch.norm(param_cpu - prev_weights_layer[j]).item()
        total_change += change
        num_params += 1
        prev_weights_layer[j] = param_cpu.clone()
    
    return total_change / num_params if num_params > 0 else 0.0

def ff_step_optimized(model, x, y, optimizers_layers, optimizer_head, prev_weights, vocab_size, device, alpha=1.0):
    """
    ОПТИМИЗИРОВАННОЕ FF обучение
    - Минимум переносов данных между CPU/GPU
    - Использование in-place операций где возможно
    - Батчированные вычисления
    """
    losses_per_layer = []
    weight_changes = []
    
    # Переносим данные на устройство один раз
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    
    # Глобальная ошибка
    with torch.no_grad():  # Изменено с inference_mode на no_grad
        logits_final = model(x)
        global_loss = F.cross_entropy(
            logits_final.view(-1, vocab_size), 
            y.view(-1)
        ).item()
    
    # Локальное обучение каждого слоя
    out = model.embed(x).transpose(0, 1)  # (seq_len, B, d_model)
    
    for i, (layer, opt) in enumerate(zip(model.layers, optimizers_layers)):
        # ----- Обучение слоя -----
        out_detached = out.detach()
        out_detached.requires_grad_(True)
        
        out_tmp = layer(out_detached)
        h = out_tmp.transpose(0, 1)  # (B, seq_len, d_model)
        
        h_norm = model.ln(h)
        temp_logits = model.fc_out(h_norm)
        
        raw_ce = F.cross_entropy(
            temp_logits.view(-1, vocab_size),
            y.view(-1)
        )
        
        # Сохраняем loss (на CPU)
        losses_per_layer.append(raw_ce.item())
        
        # Локальная функция потерь
        local_loss = raw_ce * raw_ce.detach() * alpha
        
        opt.zero_grad(set_to_none=True)  # Быстрее, чем обычный zero_grad()
        local_loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
        
        opt.step()
        
        # ----- Подсчёт изменения весов (на CPU) -----
        avg_change = compute_weight_changes(layer, prev_weights[i])
        weight_changes.append(avg_change)
        
        # ----- Второй прогон (inference) -----
        with torch.no_grad():  # Изменено с inference_mode на no_grad
            out = layer(out.detach())
    
    # Обновляем голову сети
    out = out.detach()
    out.requires_grad_(True)
    out_head = model.ln(out.transpose(0, 1))
    logits = model.fc_out(out_head)
    
    head_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer_head.zero_grad(set_to_none=True)
    torch.nn.utils.clip_grad_norm_(
        list(model.ln.parameters()) + list(model.fc_out.parameters()), 
        max_norm=1.0
    )
    optimizer_head.step()
    
    return losses_per_layer, weight_changes, global_loss

def save_plots(layer_losses_history, layer_weight_changes_history, batch_counter, NUM_LAYERS):
    """Сохранение графиков в файл"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.viridis(range(0, 256, 256 // NUM_LAYERS))
    x_data = range(len(layer_losses_history[0]))
    
    # График 1: Loss
    for i in range(NUM_LAYERS):
        ax1.plot(x_data, layer_losses_history[i], 
                label=f"Layer {i+1}", linewidth=1.5, 
                color=colors[i], alpha=0.8)
    
    ax1.set_xlabel("Batch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("FF Training: Loss по каждому слою", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Weight changes
    for i in range(NUM_LAYERS):
        ax2.plot(x_data, layer_weight_changes_history[i], 
                label=f"Layer {i+1}", linewidth=1.5, 
                color=colors[i], alpha=0.8)
    
    ax2.set_xlabel("Batch", fontsize=11)
    ax2.set_ylabel("Weight Change (L2 norm)", fontsize=11)
    ax2.set_title("FF Training: Изменение весов по каждому слою", 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'ff_training_plot_batch_{batch_counter}.png', dpi=100)
    plt.close()

# -------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# -------------------------
def main():
    print("🚀 Запуск ОПТИМИЗИРОВАННОГО FF-обучения для M1")
    
    # Настройка
    device = get_device()
    SEQ_LEN = 128
    BATCH_SIZE = 32
    NUM_WORKERS = min(4, mp.cpu_count())
    PIN_MEMORY = True if device.type == "cuda" else False
    
    # Загрузка данных
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
    
    # Dataset и DataLoader
    dataset = TokenDataset(data, SEQ_LEN)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    print(f"✅ DataLoader готов (batch_size={BATCH_SIZE}, workers={NUM_WORKERS})")
    
    # Инициализация модели
    print("🔧 Инициализация модели...")
    torch.manual_seed(42)
    net_ff = SimpleTransformer(vocab_size).to(device)
    NUM_LAYERS = len(net_ff.layers)
    print(f"✅ Модель перенесена на {device}")
    
    # Оптимизаторы
    optimizer_ff_layers = [
        optim.AdamW(
            layer.parameters(), 
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        ) 
        for layer in net_ff.layers
    ]
    optimizer_ff_head = optim.AdamW(
        list(net_ff.ln.parameters()) + list(net_ff.fc_out.parameters()), 
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Для отслеживания изменений весов
    previous_weights = []
    for layer in net_ff.layers:
        layer_weights = []
        for param in layer.parameters():
            layer_weights.append(param.data.cpu().clone())
        previous_weights.append(layer_weights)
    
    print(f"✅ Модель инициализирована ({NUM_LAYERS} слоёв)")
    print(f"📊 Параметров в модели: {sum(p.numel() for p in net_ff.parameters()):,}")
    
    # Логирование
    layer_losses_history = [[] for _ in range(NUM_LAYERS)]
    layer_weight_changes_history = [[] for _ in range(NUM_LAYERS)]
    global_losses = []
    csv_data = []
    
    SAVE_PLOT_EVERY = 100
    LOG_EVERY = 10
    print("📊 Логирование настроено")
    
    # Основной цикл обучения
    batch_counter = 0
    print("\n🏃 Начинаем обучение...\n")
    
    for epoch in range(1):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
        
        for batch_idx, (x, y) in enumerate(loop):
            # FF обучение
            losses, weight_changes, global_loss = ff_step_optimized(
                net_ff, x, y, optimizer_ff_layers, optimizer_ff_head, 
                previous_weights, vocab_size, device, alpha=1.0
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
            
            # Сохраняем в CSV периодически
            if batch_idx % LOG_EVERY == 0:
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
            
            # Сохраняем график периодически
            if batch_idx > 0 and batch_idx % SAVE_PLOT_EVERY == 0:
                save_plots(layer_losses_history, layer_weight_changes_history, batch_idx, NUM_LAYERS)
                print(f"\n📊 График сохранён: ff_training_plot_batch_{batch_idx}.png")
            
            batch_counter += 1
    
    # Финальный график
    save_plots(layer_losses_history, layer_weight_changes_history, batch_counter, NUM_LAYERS)
    print(f"\n📊 Финальный график сохранён: ff_training_plot_batch_{batch_counter}.png")
    
    # Сохранение результатов
    df = pd.DataFrame(csv_data)
    df.to_csv("ff_training_results_optimized.csv", index=False)
    print("\n" + "="*60)
    print("✅ Обучение завершено!")
    print("="*60)
    print(f"📁 Результаты сохранены в ff_training_results_optimized.csv")
    print(f"📊 Записано {len(csv_data)} строк данных")
    print(f"\n📈 Финальная средняя loss: {df['avg_layer_loss'].iloc[-1]:.4f}")
    print(f"⚖️  Финальное среднее изменение весов: {df['avg_weight_change'].iloc[-1]:.6f}")
    
    # Статистика по слоям
    print("\n" + "="*60)
    print("📊 СТАТИСТИКА ПО СЛОЯМ (финальные значения):")
    print("="*60)
    for i in range(NUM_LAYERS):
        final_loss = df[f"layer_{i+1}_loss"].iloc[-1]
        final_weight_change = df[f"layer_{i+1}_weight_change"].iloc[-1]
        print(f"Layer {i+1:3d}: Loss = {final_loss:.4f}, ΔW = {final_weight_change:.6f}")
    
    # Сохраняем модель
    torch.save({
        'model_state_dict': net_ff.state_dict(),
        'epoch': epoch,
        'final_loss': df['avg_layer_loss'].iloc[-1],
    }, 'ff_model_checkpoint.pt')
    print("\n💾 Модель сохранена: ff_model_checkpoint.pt")

# -------------------------
# ТОЧКА ВХОДА
# -------------------------
if __name__ == '__main__':
    # Для многопроцессности на macOS/Windows
    mp.set_start_method('spawn', force=True)
    main()