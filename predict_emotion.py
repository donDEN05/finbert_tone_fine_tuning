import os
import sys
import time
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Настройка путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TODO: ИЗМЕНИТЬ ПРИ СМЕНЕ МОДЕЛИ - путь к сохраненной модели
MODEL_PATH = os.path.join(BASE_DIR, "finbert_tone_finetuned")
LABEL_ENCODER_PATH = os.path.join(MODEL_PATH, "label_encoder.pkl")

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Используемое устройство: {device}")


class EmotionClassifier:
    """Класс для классификации эмоций в тексте с помощью обученной модели FinBERT-Tone"""
    
    def __init__(self, model_path=MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH, device=device):
        """
        Инициализация классификатора эмоций
        
        Args:
            model_path: путь к сохраненной модели
            label_encoder_path: путь к сохраненному label encoder
            device: устройство для выполнения (cuda/cpu)
        """
        self.device = device
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        
        # Загрузка модели и токенизатора
        print(f"📥 Загрузка модели из {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Загрузка label encoder
        print(f"📥 Загрузка label encoder из {label_encoder_path}...")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Получаем список классов
        self.classes = self.label_encoder.classes_
        self.num_labels = len(self.classes)
        
        print(f"✅ Модель загружена успешно!")
        print(f"   Количество классов: {self.num_labels}")
        print(f"   Классы: {list(self.classes)}")
    
    def predict(self, text, return_probabilities=False, max_length=128):
        """
        Предсказание эмоции для текста
        
        Args:
            text: текст на английском языке для классификации
            return_probabilities: если True, возвращает вероятности всех классов
            max_length: максимальная длина токенизированного текста
        
        Returns:
            Если return_probabilities=False: строка с названием класса
            Если return_probabilities=True: словарь с классом, вероятностями и временем выполнения
        """
        # Начало измерения времени
        start_time = time.time()
        
        # Токенизация текста
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Перенос на устройство
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Получаем предсказанный класс
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        predicted_class = self.label_encoder.inverse_transform([predicted_class_id])[0]
        
        # Получаем вероятность предсказанного класса
        predicted_probability = probabilities[0][predicted_class_id].item()
        
        # Конец измерения времени
        elapsed_time = time.time() - start_time
        
        if return_probabilities:
            # Возвращаем все вероятности
            all_probabilities = probabilities[0].cpu().numpy()
            class_probabilities = {
                self.classes[i]: float(all_probabilities[i])
                for i in range(len(self.classes))
            }
            return {
                'predicted_class': predicted_class,
                'probability': float(predicted_probability),
                'all_probabilities': class_probabilities,
                'inference_time': elapsed_time  # Время выполнения в секундах
            }
        else:
            return predicted_class
    
    def predict_batch(self, texts, return_probabilities=False, max_length=128):
        """
        Предсказание эмоций для списка текстов
        
        Args:
            texts: список текстов для классификации
            return_probabilities: если True, возвращает вероятности всех классов
            max_length: максимальная длина токенизированного текста
        
        Returns:
            Если return_probabilities=True: словарь с результатами и общей статистикой времени
            Иначе: список предсказаний
        """
        start_time = time.time()
        results = []
        
        for text in texts:
            result = self.predict(text, return_probabilities=return_probabilities, max_length=max_length)
            results.append(result)
        
        total_time = time.time() - start_time
        
        if return_probabilities:
            # Вычисляем среднее время на один текст
            avg_time = total_time / len(texts) if len(texts) > 0 else 0
            return {
                'results': results,
                'total_time': total_time,
                'avg_time_per_text': avg_time,
                'num_texts': len(texts)
            }
        else:
            return results


def main():
    """Основная функция для интерактивного использования"""
    print("="*60)
    print("🎭 КЛАССИФИКАЦИЯ ЭМОЦИЙ В ТЕКСТЕ")
    print("="*60)
    
    # Проверка существования модели
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Ошибка: Модель не найдена по пути {MODEL_PATH}")
        print("   Сначала обучите модель, запустив train_finbert.py")
        return
    
    if not os.path.exists(LABEL_ENCODER_PATH):
        print(f"❌ Ошибка: Label encoder не найден по пути {LABEL_ENCODER_PATH}")
        print("   Сначала обучите модель, запустив train_finbert.py")
        return
    
    # Инициализация классификатора
    try:
        classifier = EmotionClassifier()
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return
    
    print("\n" + "="*60)
    print("💬 ВВОД ТЕКСТА ДЛЯ КЛАССИФИКАЦИИ")
    print("="*60)
    print("Введите текст на английском языке (или 'quit' для выхода):\n")
    
    # Интерактивный режим
    while True:
        try:
            text = input("Текст: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if not text:
                print("⚠️  Введите непустой текст")
                continue
            
            # Предсказание
            result = classifier.predict(text, return_probabilities=True)
            
            # Вывод результата
            print(f"\n📊 Результат классификации:")
            print(f"   Предсказанная эмоция: {result['predicted_class']}")
            print(f"   Уверенность: {result['probability']:.2%}")
            print(f"   ⏱️  Время выполнения: {result['inference_time']:.4f} сек")
            
            # Показываем топ-3 наиболее вероятных класса
            sorted_probs = sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            print(f"\n   Топ-3 наиболее вероятных эмоции:")
            for i, (emotion, prob) in enumerate(sorted_probs, 1):
                print(f"   {i}. {emotion}: {prob:.2%}")
            
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка при обработке текста: {e}\n")


if __name__ == "__main__":
    # Пример использования в коде
    if len(sys.argv) > 1:
        # Если передан текст как аргумент командной строки
        text = " ".join(sys.argv[1:])
        
        print("="*60)
        print("🎭 КЛАССИФИКАЦИЯ ЭМОЦИЙ В ТЕКСТЕ")
        print("="*60)
        
        try:
            classifier = EmotionClassifier()
            result = classifier.predict(text, return_probabilities=True)
            
            print(f"\n📝 Текст: {text}")
            print(f"\n📊 Результат:")
            print(f"   Предсказанная эмоция: {result['predicted_class']}")
            print(f"   Уверенность: {result['probability']:.2%}")
            print(f"   ⏱️  Время выполнения: {result['inference_time']:.4f} сек")
            
            # Показываем все вероятности
            print(f"\n   Все вероятности:")
            sorted_probs = sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for emotion, prob in sorted_probs:
                print(f"   - {emotion}: {prob:.2%}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    else:
        # Интерактивный режим
        main()

