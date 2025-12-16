import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import mlflow
import mlflow.pytorch
import numpy as np
from datetime import datetime
import pickle
import numpy as np

# Настройка путей (используем относительные пути для кроссплатформенности)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - путь сохранения модели
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "finbert_tone_finetuned")
# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - название эксперимента в MLFlow
MLFLOW_EXPERIMENT_NAME = "finbert_tone_emotion_classification"
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")

# Проверка и загрузка данных
print("="*60)
print("📊 ЗАГРУЗКА ДАННЫХ")
print("="*60)
try:
    data = pd.read_csv(DATA_PATH)
    print(f'✅ Датасет загружен: {data.shape[0]} записей, {data.shape[1]} столбцов')
    
    # Проверка наличия необходимых колонок
    required_columns = ['text', 'emotion']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ Ошибка: Отсутствуют необходимые колонки: {missing_columns}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Ошибка при чтении файла: {str(e)}")
    sys.exit(1)

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Используемое устройство: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# Подготовка данных
print("\n" + "="*60)
print("🔧 ПОДГОТОВКА ДАННЫХ")
print("="*60)

# Удаление пустых значений
data = data.dropna(subset=['text', 'emotion'])
print(f"✅ Данные после удаления пустых значений: {len(data)} записей")

# Кодирование меток
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['emotion'])
num_labels = len(label_encoder.classes_)
print(f"✅ Количество классов: {num_labels}")
print(f"   Классы: {list(label_encoder.classes_)}")

# Разделение на train/val/test
train_df, temp_df = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"✅ Разделение данных:")
print(f"   Train: {len(train_df)} записей ({len(train_df)/len(data)*100:.1f}%)")
print(f"   Validation: {len(val_df)} записей ({len(val_df)/len(data)*100:.1f}%)")
print(f"   Test: {len(test_df)} записей ({len(test_df)/len(data)*100:.1f}%)")

# Создание Dataset класса
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Загрузка токенизатора и модели
print("\n" + "="*60)
print("🤖 ЗАГРУЗКА МОДЕЛИ")
print("="*60)
# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - название модели в print
print("Загрузка yiyanghkust/finbert-tone...")

# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - название модели в AutoTokenizer.from_pretrained
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - обновить комментарии о специфике модели
# Используем ignore_mismatched_sizes=True, чтобы загрузить базовые веса FinBERT-Tone
# и инициализировать новый classifier head с правильным количеством классов
# Оригинальная модель finbert-tone обучена на задачу анализа тональности,
# но нам нужно адаптировать её для классификации эмоций с нужным количеством классов
print(f"⚠️  Адаптируем модель для {num_labels} классов эмоций")
# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - название модели в AutoModelForSequenceClassification.from_pretrained
model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone",
    num_labels=num_labels,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True  # Игнорируем несоответствие размера classifier
)
model.to(device)

print(f"✅ Модель загружена и настроена для {num_labels} классов")

# Создание датасетов
train_dataset = EmotionDataset(train_df['text'], train_df['label'], tokenizer)
val_dataset = EmotionDataset(val_df['text'], val_df['label'], tokenizer)
test_dataset = EmotionDataset(test_df['text'], test_df['label'], tokenizer)

# Функция для вычисления метрик
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Callback для логирования в MLFlow по батчам
class MLFlowLoggingCallback(TrainerCallback):
    """Callback для логирования метрик в MLFlow на каждом шаге и батче"""
    
    def __init__(self):
        self.step = 0
        self.epoch = 0
        
    def on_train_step_end(self, args, state, control, model=None, **kwargs):
        """Вызывается после каждого шага обучения (каждого батча)"""
        # Пытаемся получить loss из разных источников
        batch_loss = None
        
        # Способ 1: из log_history (если уже записан)
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                batch_loss = last_log['loss']
        
        # Способ 2: из kwargs (если передан напрямую)
        if batch_loss is None and 'loss' in kwargs:
            batch_loss = kwargs['loss']
        
        # Логируем loss каждого батча в MLFlow
        if batch_loss is not None:
            try:
                mlflow.log_metric('batch_loss', float(batch_loss), step=state.global_step)
            except Exception as e:
                # Игнорируем ошибки логирования, чтобы не прерывать обучение
                pass
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при каждом логировании (каждый logging_steps)"""
        if logs is not None:
            # Логируем метрики в MLFlow
            metrics_to_log = {}
            
            # Логируем усредненный loss (это среднее за последние logging_steps батчей)
            if 'loss' in logs:
                metrics_to_log['train_loss_avg'] = logs['loss']
                
            # Логируем learning rate если доступен
            if 'learning_rate' in logs:
                metrics_to_log['learning_rate'] = logs['learning_rate']
                
            # Логируем другие метрики если есть
            for key, value in logs.items():
                if key not in ['loss', 'learning_rate', 'epoch'] and isinstance(value, (int, float)):
                    metrics_to_log[f'train_{key}'] = value
            
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log, step=state.global_step)
                
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце каждой эпохи"""
        self.epoch = state.epoch

# Настройка MLFlow
print("\n" + "="*60)
print("📈 НАСТРОЙКА MLFLOW")
print("="*60)

# Устанавливаем tracking URI для использования локальной директории
# Используем правильный формат для всех платформ
if os.name == 'nt':  # Windows
    # На Windows используем формат file:///D:/path/to/mlruns (с заглавной буквой диска)
    abs_path = os.path.abspath(MLFLOW_TRACKING_URI)
    # Преобразуем путь: D:\path\to\mlruns -> D:/path/to/mlruns
    uri_path = abs_path.replace('\\', '/')
    # Убеждаемся, что буква диска заглавная
    if len(uri_path) > 1 and uri_path[1] == ':':
        uri_path = uri_path[0].upper() + uri_path[1:]
    tracking_uri = f"file:///{uri_path}"
else:  # Unix/Linux/Mac
    tracking_uri = f"file://{os.path.abspath(MLFLOW_TRACKING_URI)}"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"✅ Эксперимент MLFlow: {MLFLOW_EXPERIMENT_NAME}")
print(f"✅ MLFlow tracking URI: {tracking_uri}")

# Параметры обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=500,
    weight_decay=0.02,
    logging_dir='./logs',
    logging_steps=5,  # Логирование каждые 10 батчей для детального отслеживания loss
                      # Для логирования каждого батча установите logging_steps=1
    eval_strategy="epoch",  # Исправлено: evaluation_strategy -> eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),  # Использовать mixed precision если доступна GPU
    report_to="none",  # Отключаем wandb/tensorboard, используем только MLFlow
)

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        MLFlowLoggingCallback()  # Добавляем callback для логирования по батчам
    ]
)

# Начало MLFlow run
print("\n" + "="*60)
print("🚀 НАЧАЛО ОБУЧЕНИЯ")
print("="*60)

# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - название run в MLFlow
with mlflow.start_run(run_name=f"finbert_tone_emotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Логирование параметров
    mlflow.log_params({
        # TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - название модели в mlflow.log_params
        "model_name": "yiyanghkust/finbert-tone",
        "num_labels": num_labels,
        "num_train_samples": len(train_df),
        "num_val_samples": len(val_df),
        "num_test_samples": len(test_df),
        "max_length": 128,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "num_epochs": training_args.num_train_epochs,
        "weight_decay": training_args.weight_decay,
        "warmup_steps": training_args.warmup_steps,
    })
    
    # Логирование классов
    mlflow.log_dict(
        {str(i): label for i, label in enumerate(label_encoder.classes_)},
        "label_mapping.json"
    )
    
    # Обучение
    print("Начало обучения...")
    train_result = trainer.train()
    
    # Логирование метрик обучения
    mlflow.log_metrics({
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get('train_runtime', 0),
        "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
    })
    
    # Оценка на валидационном наборе
    print("\nОценка на валидационном наборе...")
    val_metrics = trainer.evaluate()
    
    # Логирование метрик валидации
    mlflow.log_metrics({
        "val_loss": val_metrics['eval_loss'],
        "val_accuracy": val_metrics['eval_accuracy'],
        "val_f1": val_metrics['eval_f1'],
        "val_precision": val_metrics['eval_precision'],
        "val_recall": val_metrics['eval_recall'],
    })
    
    print(f"\n📊 Метрики валидации:")
    print(f"   Loss: {val_metrics['eval_loss']:.4f}")
    print(f"   Accuracy: {val_metrics['eval_accuracy']:.4f}")
    print(f"   F1: {val_metrics['eval_f1']:.4f}")
    print(f"   Precision: {val_metrics['eval_precision']:.4f}")
    print(f"   Recall: {val_metrics['eval_recall']:.4f}")
    
    # Оценка на тестовом наборе
    print("\nОценка на тестовом наборе...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    # Логирование метрик теста
    mlflow.log_metrics({
        "test_loss": test_metrics['eval_loss'],
        "test_accuracy": test_metrics['eval_accuracy'],
        "test_f1": test_metrics['eval_f1'],
        "test_precision": test_metrics['eval_precision'],
        "test_recall": test_metrics['eval_recall'],
    })
    
    print(f"\n📊 Метрики теста:")
    print(f"   Loss: {test_metrics['eval_loss']:.4f}")
    print(f"   Accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"   F1: {test_metrics['eval_f1']:.4f}")
    print(f"   Precision: {test_metrics['eval_precision']:.4f}")
    print(f"   Recall: {test_metrics['eval_recall']:.4f}")
    
    # Детальный отчет по классам на тестовом наборе
    print("\nГенерация детального отчета по классам...")
    test_predictions = trainer.predict(test_dataset)
    test_pred_labels = np.argmax(test_predictions.predictions, axis=1)
    test_true_labels = test_predictions.label_ids
    
    class_report = classification_report(
        test_true_labels,
        test_pred_labels,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # Логирование метрик по классам
    for emotion, metrics in class_report.items():
        if isinstance(metrics, dict):
            mlflow.log_metrics({
                f"test_{emotion}_precision": metrics.get('precision', 0),
                f"test_{emotion}_recall": metrics.get('recall', 0),
                f"test_{emotion}_f1": metrics.get('f1-score', 0),
                f"test_{emotion}_support": metrics.get('support', 0),
            })
    
    print("\n" + classification_report(
        test_true_labels,
        test_pred_labels,
        target_names=label_encoder.classes_
    ))
    
    # Сохранение модели
    print("\n" + "="*60)
    print("💾 СОХРАНЕНИЕ МОДЕЛИ")
    print("="*60)
    
    # Исправление проблемы с pickle и fp16: unwrap модель перед сохранением
    try:
        from accelerate import unwrap_model
        # Unwrap модель для корректного сохранения (убирает обертку от mixed precision)
        model_to_save = unwrap_model(trainer.model)
        print("✅ Модель unwrapped с помощью accelerate")
    except (ImportError, AttributeError):
        # Если accelerate не установлен или модель не обернута, используем модель напрямую
        model_to_save = trainer.model
        # Если модель обернута в DataParallel или DistributedDataParallel
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module
        # Если модель обернута в accelerate (но accelerate не импортирован)
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod
        print("✅ Модель обработана без accelerate")
    
    # Сохранение через MLFlow (используем unwrapped модель)
    try:
        # TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - зарегистрированное имя модели в MLFlow
        mlflow.pytorch.log_model(
            model_to_save,
            "model",
            registered_model_name="finbert_tone_emotion_classifier"
        )
        print("✅ Модель сохранена в MLFlow")
    except Exception as e:
        print(f"⚠️  Предупреждение при сохранении в MLFlow: {e}")
        print("   Продолжаем с локальным сохранением...")
    
    # Сохранение локально (trainer.save_model автоматически обрабатывает unwrap)
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"✅ Модель сохранена в: {MODEL_SAVE_PATH}")
    
    # Сохранение label encoder
    with open(os.path.join(MODEL_SAVE_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder сохранен")
    
    # Логирование артефактов
    mlflow.log_artifacts(MODEL_SAVE_PATH, "model_files")
    
    print("\n" + "="*60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"Модель сохранена в: {MODEL_SAVE_PATH}")
    print(f"MLFlow эксперимент: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Для просмотра метрик запустите: mlflow ui")

