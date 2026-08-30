from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd




class NewsSentimentAnalyzer:
    """
    Class for analyzing news sentiment. Translates text from Russian to English
    and determines the text's sentiment (positive, negative, neutral) using the FinBERT model.
    """

    def __init__(self, translation_model_name="Helsinki-NLP/opus-mt-ru-en", finbert_model_name="ProsusAI/finbert"):
        """
        Initializes NewsSentimentAnalyzer with the translation and sentiment-analysis models.

        Args:
            translation_model_name (str): Name of the text-translation model.
            finbert_model_name (str): Name of the FinBERT sentiment-analysis model.
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
        Translates text from Russian to English using the MarianMT model.

        Args:
            text (list): List of Russian-language text strings.

        Returns:
            list: List of translated English text strings.
        """
        inputs = self.translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.translation_model.generate(**inputs)

        translated_text = self.translation_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated_text

    def analyze_sentiment(self, financial_text_russian):
        """
        Analyzes the sentiment of Russian-language financial news. First translates
        the text, then uses the FinBERT model to determine sentiment.

        Args:
            financial_text_russian (list): List of Russian-language news text strings.

        Returns:
            pd.DataFrame: DataFrame with the sentiment-analysis results, including the
            translated text, the probability for each sentiment (Positive, Negative,
            Neutral), and the `real_score` score.
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
