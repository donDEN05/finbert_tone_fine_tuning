import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from collections import defaultdict
import inspect

# Настройка для красивого вывода
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 9

MODEL_ID = "yiyanghkust/finbert-tone"

print("="*80)
print(f"🔍 АНАЛИЗ АРХИТЕКТУРЫ МОДЕЛИ {MODEL_ID.upper()}")
print("="*80)

# Загрузка модели
print(f"\n📥 Загрузка модели {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

print("✅ Модель загружена\n")

# Анализ структуры модели
def analyze_model_structure(model, prefix=""):
    """Рекурсивный анализ структуры модели"""
    layers_info = []
    total_params = 0
    
    for name, module in model.named_children():
        module_type = type(module).__name__
        num_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        layer_info = {
            'name': name,
            'type': module_type,
            'params': num_params,
            'trainable_params': trainable_params,
            'full_path': f"{prefix}.{name}" if prefix else name
        }
        
        # Получаем методы модуля
        methods = [method for method in dir(module) 
                  if not method.startswith('_') and callable(getattr(module, method, None))]
        layer_info['methods'] = methods[:10]  # Первые 10 методов
        
        layers_info.append(layer_info)
        total_params += num_params
        
        # Рекурсивно анализируем вложенные модули
        if len(list(module.children())) > 0:
            sub_layers = analyze_model_structure(module, f"{prefix}.{name}" if prefix else name)
            layers_info.extend(sub_layers)
    
    return layers_info

# Сбор информации о модели
print("📊 Анализ структуры модели...")
layers_info = analyze_model_structure(model)

# Подсчет статистики
total_params = sum(l['params'] for l in layers_info)
trainable_params = sum(l['trainable_params'] for l in layers_info)
layer_types = defaultdict(int)
for layer in layers_info:
    layer_types[layer['type']] += 1

print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
print(f"   Всего параметров: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   Обучаемых параметров: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print(f"   Всего слоев: {len(layers_info)}")
print(f"   Уникальных типов слоев: {len(layer_types)}")

print(f"\n📋 ТИПЫ СЛОЕВ:")
for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1]):
    print(f"   {layer_type}: {count}")

# Детальная информация о слоях
print(f"\n{'='*80}")
print("🔬 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О СЛОЯХ")
print(f"{'='*80}")

# Группировка по основным компонентам
bert_layers = [l for l in layers_info if 'bert' in l['name'].lower() or 'encoder' in l['name'].lower()]
classifier_layers = [l for l in layers_info if 'classifier' in l['name'].lower() or 'dropout' in l['name'].lower()]
embedding_layers = [l for l in layers_info if 'embedding' in l['name'].lower()]

print(f"\n🧠 BERT Encoder слои: {len(bert_layers)}")
print(f"📊 Classifier слои: {len(classifier_layers)}")
print(f"🔤 Embedding слои: {len(embedding_layers)}")

# Вывод топ-10 самых больших слоев
print(f"\n📊 ТОП-10 СЛОЕВ ПО КОЛИЧЕСТВУ ПАРАМЕТРОВ:")
sorted_layers = sorted(layers_info, key=lambda x: -x['params'])[:10]
for i, layer in enumerate(sorted_layers, 1):
    print(f"   {i:2d}. {layer['full_path']:50s} | {layer['type']:30s} | {layer['params']:>12,} params")

# Визуализация 1: Иерархическая структура
print(f"\n🎨 Создание визуализаций...")

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# График 1: Распределение параметров по типам слоев
ax1 = fig.add_subplot(gs[0, 0])
layer_params = defaultdict(int)
for layer in layers_info:
    layer_params[layer['type']] += layer['params']

sorted_types = sorted(layer_params.items(), key=lambda x: -x[1])[:15]
types_names = [t[0] for t in sorted_types]
types_params = [t[1] for t in sorted_types]

colors = plt.cm.viridis(np.linspace(0, 1, len(types_names)))
bars = ax1.barh(range(len(types_names)), [p/1e6 for p in types_params], color=colors)
ax1.set_yticks(range(len(types_names)))
ax1.set_yticklabels(types_names, fontsize=8)
ax1.set_xlabel('Параметры (миллионы)', fontsize=10, fontweight='bold')
ax1.set_title('Распределение параметров по типам слоев', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Добавляем значения на столбцы
for i, (bar, params) in enumerate(zip(bars, types_params)):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{params/1e6:.2f}M', va='center', fontsize=8)

# График 2: Количество слоев по типам
ax2 = fig.add_subplot(gs[0, 1])
type_counts = sorted(layer_types.items(), key=lambda x: -x[1])[:15]
type_names = [t[0] for t in type_counts]
type_counts_vals = [t[1] for t in type_counts]

colors2 = plt.cm.plasma(np.linspace(0, 1, len(type_names)))
bars2 = ax2.bar(range(len(type_names)), type_counts_vals, color=colors2)
ax2.set_xticks(range(len(type_names)))
ax2.set_xticklabels(type_names, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Количество слоев', fontsize=10, fontweight='bold')
ax2.set_title('Количество слоев по типам', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for bar, count in zip(bars2, type_counts_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontsize=8)

# График 3: Архитектурная схема (упрощенная)
ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 8)
ax3.axis('off')
ax3.set_title('Архитектурная схема FinBERT-Tone', fontsize=14, fontweight='bold', pad=20)

# Определяем основные компоненты
components = [
    {'name': 'Input\n(Text)', 'x': 1, 'y': 4, 'color': '#FF6B6B', 'width': 0.8, 'height': 1},
    {'name': 'Tokenizer', 'x': 2.5, 'y': 4, 'color': '#4ECDC4', 'width': 0.8, 'height': 1},
    {'name': 'Embeddings\n(Word + Position + Token)', 'x': 4, 'y': 4, 'color': '#95E1D3', 'width': 1.2, 'height': 1},
    {'name': 'BERT Encoder\n(12 Transformer Layers)', 'x': 6, 'y': 4, 'color': '#F38181', 'width': 1.5, 'height': 2},
    {'name': 'Pooler\n(CLS Token)', 'x': 8, 'y': 5, 'color': '#AA96DA', 'width': 0.8, 'height': 0.6},
    {'name': 'Dropout', 'x': 8, 'y': 3.5, 'color': '#FCBAD3', 'width': 0.8, 'height': 0.6},
    {'name': 'Classifier\n(Linear Layer)', 'x': 9.5, 'y': 4, 'color': '#FFD93D', 'width': 0.8, 'height': 1},
]

# Рисуем компоненты
for comp in components:
    box = FancyBboxPatch(
        (comp['x'], comp['y'] - comp['height']/2),
        comp['width'], comp['height'],
        boxstyle="round,pad=0.1",
        facecolor=comp['color'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7
    )
    ax3.add_patch(box)
    ax3.text(comp['x'] + comp['width']/2, comp['y'], comp['name'],
             ha='center', va='center', fontsize=9, fontweight='bold')

# Рисуем стрелки
arrows = [
    (1.8, 4, 2.5, 4),
    (3.3, 4, 4, 4),
    (5.2, 4, 6, 4),
    (7.5, 4.3, 8, 4.3),
    (7.5, 3.7, 8, 3.7),
    (8.8, 4, 9.5, 4),
]

for x1, y1, x2, y2 in arrows:
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', lw=2, color='black', alpha=0.6
    )
    ax3.add_patch(arrow)

# График 4: Детальная структура BERT Encoder
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title('Детальная структура Transformer Layer', fontsize=12, fontweight='bold', pad=15)

# Компоненты Transformer слоя
transformer_components = [
    {'name': 'Multi-Head\nAttention', 'x': 2, 'y': 4.5, 'color': '#FF6B9D', 'w': 1.2, 'h': 0.8},
    {'name': 'Add & Norm', 'x': 3.5, 'y': 4.5, 'color': '#C44569', 'w': 0.8, 'h': 0.8},
    {'name': 'Feed Forward\n(2 Linear)', 'x': 5, 'y': 4.5, 'color': '#F8B500', 'w': 1.2, 'h': 0.8},
    {'name': 'Add & Norm', 'x': 6.5, 'y': 4.5, 'color': '#C44569', 'w': 0.8, 'h': 0.8},
    {'name': 'Input', 'x': 0.5, 'y': 4.5, 'color': '#95E1D3', 'w': 0.8, 'h': 0.8},
    {'name': 'Output', 'x': 7.8, 'y': 4.5, 'color': '#95E1D3', 'w': 0.8, 'h': 0.8},
]

for comp in transformer_components:
    box = FancyBboxPatch(
        (comp['x'], comp['y'] - comp['h']/2),
        comp['w'], comp['h'],
        boxstyle="round,pad=0.05",
        facecolor=comp['color'],
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7
    )
    ax4.add_patch(box)
    ax4.text(comp['x'] + comp['w']/2, comp['y'], comp['name'],
             ha='center', va='center', fontsize=8, fontweight='bold')

# Стрелки для transformer
transformer_arrows = [
    (1.3, 4.5, 2, 4.5),
    (3.2, 4.5, 3.5, 4.5),
    (4.3, 4.5, 5, 4.5),
    (6.2, 4.5, 6.5, 4.5),
    (7.3, 4.5, 7.8, 4.5),
    # Residual connections
    (0.9, 4.5, 0.9, 3.5), (0.9, 3.5, 3.5, 3.5), (3.5, 3.5, 3.5, 4.1),  # First residual
    (4.7, 4.5, 4.7, 2.5), (4.7, 2.5, 6.5, 2.5), (6.5, 2.5, 6.5, 4.1),  # Second residual
]

for coords in transformer_arrows:
    if len(coords) == 4:
        arrow = FancyArrowPatch(
            (coords[0], coords[1]), (coords[2], coords[3]),
            arrowstyle='->', lw=1.5, color='#2C3E50', alpha=0.5, 
            connectionstyle="arc3,rad=0.1" if abs(coords[1] - coords[3]) > 0.1 else None
        )
        ax4.add_patch(arrow)

# График 5: Методы основных компонентов
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
ax5.set_title('Основные методы модели', fontsize=12, fontweight='bold', pad=15)

# Получаем методы модели
model_methods = [method for method in dir(model) 
                if not method.startswith('_') and callable(getattr(model, method, None))]

# Группируем методы по категориям
forward_methods = [m for m in model_methods if 'forward' in m.lower() or 'call' in m.lower()]
get_methods = [m for m in model_methods if m.startswith('get')]
set_methods = [m for m in model_methods if m.startswith('set')]
other_methods = [m for m in model_methods if m not in forward_methods + get_methods + set_methods][:15]

methods_text = "ОСНОВНЫЕ МЕТОДЫ:\n\n"
methods_text += "🔹 Forward методы:\n"
for m in forward_methods[:5]:
    methods_text += f"   • {m}\n"

methods_text += "\n🔹 Get методы:\n"
for m in get_methods[:5]:
    methods_text += f"   • {m}\n"

methods_text += "\n🔹 Set методы:\n"
for m in set_methods[:5]:
    methods_text += f"   • {m}\n"

methods_text += "\n🔹 Другие важные методы:\n"
for m in other_methods[:10]:
    methods_text += f"   • {m}\n"

ax5.text(0.05, 0.95, methods_text, transform=ax5.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Сохранение (используем относительные пути)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(BASE_DIR, "model_visualization.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Визуализация сохранена: {output_path}")

# Дополнительная информация в консоль
print(f"\n{'='*80}")
print("📝 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О КОМПОНЕНТАХ")
print(f"{'='*80}")

# Анализ BERT encoder
bert_model = model.bert if hasattr(model, 'bert') else None
if bert_model:
    print(f"\n🧠 BERT Encoder:")
    if hasattr(bert_model, 'encoder'):
        encoder = bert_model.encoder
        if hasattr(encoder, 'layer'):
            print(f"   Количество Transformer слоев: {len(encoder.layer)}")
            if len(encoder.layer) > 0:
                first_layer = encoder.layer[0]
                print(f"   Структура одного слоя:")
                for name, module in first_layer.named_children():
                    print(f"      - {name}: {type(module).__name__}")

# Анализ Embeddings
if bert_model and hasattr(bert_model, 'embeddings'):
    embeddings = bert_model.embeddings
    print(f"\n🔤 Embeddings:")
    for name, module in embeddings.named_children():
        print(f"   - {name}: {type(module).__name__}")

# Анализ Classifier
if hasattr(model, 'classifier'):
    classifier = model.classifier
    print(f"\n📊 Classifier:")
    if isinstance(classifier, nn.Sequential):
        for i, module in enumerate(classifier):
            print(f"   Layer {i}: {type(module).__name__}")
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                print(f"      Input: {module.in_features}, Output: {module.out_features}")
    else:
        print(f"   Type: {type(classifier).__name__}")
        if hasattr(classifier, 'in_features') and hasattr(classifier, 'out_features'):
            print(f"      Input: {classifier.in_features}, Output: {classifier.out_features}")

# Информация о размерах
print(f"\n📏 РАЗМЕРЫ МОДЕЛИ:")
if bert_model:
    if hasattr(bert_model, 'config'):
        config = bert_model.config
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Number of attention heads: {config.num_attention_heads}")
        print(f"   Number of hidden layers: {config.num_hidden_layers}")
        print(f"   Intermediate size: {config.intermediate_size}")
        print(f"   Max position embeddings: {config.max_position_embeddings}")
        print(f"   Vocabulary size: {config.vocab_size}")

print(f"\n{'='*80}")
print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
print(f"{'='*80}")
print(f"\n📁 Файл сохранен: {output_path}")
print(f"📊 Всего слоев проанализировано: {len(layers_info)}")
print(f"💾 Общий размер модели: {total_params/1e6:.2f}M параметров")

plt.show()

