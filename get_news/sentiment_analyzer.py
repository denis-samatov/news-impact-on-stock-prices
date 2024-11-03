from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd





class NewsSentimentAnalyzer:
    """
    Класс для анализа тональности новостей. Выполняет перевод текста с русского на английский
    и определяет тональность текста (положительная, отрицательная, нейтральная) с помощью модели FinBERT.
    """

    def __init__(self, translation_model_name="Helsinki-NLP/opus-mt-ru-en", finbert_model_name="ProsusAI/finbert"):
        """
        Инициализация NewsSentimentAnalyzer с моделями перевода и анализа тональности.

        Args:
            translation_model_name (str): Название модели для перевода текста.
            finbert_model_name (str): Название модели FinBERT для анализа тональности.
        """
        self.translation_model_name = translation_model_name
        self.finbert_model_name = finbert_model_name


        self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

        self.finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

        self.df = None

    def translate_text(self, text):
        """
        Переводит текст с русского на английский с использованием модели MarianMT.

        Args:
            text (list): Список строк текста на русском языке.

        Returns:
            list: Список переведенных строк текста на английский.
        """
        inputs = self.translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.translation_model.generate(**inputs)

        translated_text = self.translation_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated_text

    def analyze_sentiment(self, financial_text_russian):
        """
        Анализирует тональность финансовых новостей на русском языке. Сначала переводит текст, 
        затем использует модель FinBERT для определения тональности.

        Args:
            financial_text_russian (list): Список строк текста новостей на русском языке.

        Returns:
            pd.DataFrame: DataFrame с результатами анализа тональности, включая переведенные тексты, 
            вероятность для каждой тональности (Positive, Negative, Neutral) и оценку `real_score`.
        """
        translated_text = self.translate_text(financial_text_russian)

        inputs = self.finbert_tokenizer(translated_text, padding=True, truncation=True, return_tensors='pt')

        outputs = self.finbert_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        positive = predictions[:, 0].tolist()
        negative = predictions[:, 1].tolist()
        neutral = predictions[:, 2].tolist()

        table = {
            'Headline': financial_text_russian,
            'En Headline': translated_text,
            "Positive": positive,
            "Negative": negative,
            "Neutral": neutral
        }
        self.df = pd.DataFrame(table, columns=["Headline", 'En Headline', "Positive", "Negative", "Neutral"])
        
        self.df['real_score'] = self.df['Positive'] - self.df['Negative']
        return self.df