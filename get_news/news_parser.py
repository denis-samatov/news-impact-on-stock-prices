import os
import time
import locale
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentiment_analyzer import NewsSentimentAnalyzer

import nltk
nltk.download('punkt')


# Класс для сбора новостей по заданному URL и фильтрации по ключевым словам.
class NewsParser:
    """
    Класс для парсинга новостей с заданного URL. Сохраняет результаты парсинга и создает промежуточные сохранения для обработки большого объема данных.
    """
    
    def __init__(self, url, keywords, ticker):
        self.__url = url
        self.__keywords = keywords
        self.__driver = None
        self.__news = []
        self.ticker = ticker
        self.min_date = '2028-01-01'

    def __extract_sentences(self, content):
        """
        Фильтрует предложения, содержащие ключевые слова, из текста новости.
        
        Args:
            content (str): Текст новости.
        
        Returns:
            list: Список отфильтрованных предложений.
        """
        sentences = sent_tokenize(content)
        res_sentences = [sentence for sentence in sentences if any(keyword.lower() in sentence.lower() for keyword in self.__keywords)]
        return res_sentences

    def __format_date(self, date):
        """
        Преобразует текстовую дату в стандартный формат `YYYY-MM-DD`, учитывая относительные даты.
        
        Args:
            date (str): Дата в текстовом формате.
        
        Returns:
            str: Дата в формате `YYYY-MM-DD`.
        """
        if 'Сегодня' in date:
            formatted_date = (datetime.now()).strftime('%Y-%m-%d')
        elif 'Вчера' in date:
            formatted_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        elif ' в ' in date:
            locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
            formatted_date = datetime.strptime(date, '%d %B в %H:%M').strftime('%Y-%m-%d')
        else:
            locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
            formatted_date = datetime.strptime(date, '%d %B %Y').strftime('%Y-%m-%d')
        return formatted_date
    
    def __format_time(self, date):
        """
        Извлекает время из текстовой даты, если указано.
        
        Args:
            date (str): Дата в текстовом формате.
        
        Returns:
            str: Время в формате `HH:MM`.
        """
        if ' в ' in date:
            time_str = date.split(' в ')[1]
            time = datetime.strptime(time_str, '%H:%M').strftime('%H:%M')
        else:
            time = None
        
        return time

    def __parse_news(self, news_block):
        """
        Извлекает данные о конкретной новости из HTML-блока.
        
        Args:
            news_block (BeautifulSoup): HTML-блок с данными о новости.
        
        Returns:
            list: Список данных [дата, время, заголовок, ссылка, тип новости].
        """
        try:
            date = news_block.find("time").text.strip()
            title = news_block.find("a", class_="iKzE").text.strip()
            href = news_block.find("a", class_="iKzE")["href"]
            news_type_element = news_block.find("a", class_="ZIkT")
            news_type = news_type_element.text.strip() if news_type_element else None
            return [self.__format_date(date), self.__format_time(date), title, href, news_type]
        except Exception as e:
            print("An error occurred while parsing news:", e)
            return None

    def parse_news(self, len_dataset=500, stop_date=None):
        """
        Последовательно парсит новости с сайта, подгружая новые данные и проверяя условия остановки.
        
        Args:
            len_dataset (int): Целевое количество новостей для парсинга.
            stop_date (str): Дата, после которой прекращается парсинг новостей.
        """
        self.__driver = webdriver.Chrome()
        self.__driver.get(self.__url)
        
        while len(self.__news) < len_dataset:
            show_more_button = WebDriverWait(self.__driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[@class='nlJ0' and contains(text(), 'Показать еще новости')]"))
            )
            self.__driver.execute_script("arguments[0].click();", show_more_button)
            updated_html = BeautifulSoup(self.__driver.page_source, 'html.parser')
            news_blocks = updated_html.find_all("div", class_="TjB6 KSLV Ncpb E6j8")

            if not show_more_button.is_displayed():
                break

            for news_block in news_blocks[len(self.__news):]:
                date = news_block.find("time").text.strip()
                if date > self.min_date:
                    self.min_date = date
                parsed_data = self.__parse_news(news_block)
                if parsed_data:
                    self.__news.append(parsed_data)

            if len(self.__news) >= len_dataset or (stop_date and self.min_date < stop_date):
                break

            if len(self.__news) % 100 == 0:
                print("Загружено:", len(self.__news))
                self.save_to_csv('news_data', f'{self.ticker}_in_progress.csv')

        self.__driver.quit()

    def get_news_list(self):
        """
        Возвращает список собранных новостей.
        
        Returns:
            list: Список собранных новостей.
        """
        return self.__news

    def save_to_csv(self, output_dir, output_filename):
        """
        Сохраняет данные о новостях в CSV-файл с созданием директории `news_data`, если она отсутствует.
        
        Args:
            output_filename (str): Название CSV-файла для сохранения.
        """
        
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        
        if self.__news:
            df = pd.DataFrame(self.__news, columns=['Date', 'Time', 'Title', 'URL', 'Type'])
            df.to_csv(output_path, index=False)
        else:
            print("No news data to save.")


# Класс для анализа новостного контента, включая расчет тональности новостей.
class HelperParser:
    """
    Класс для анализа новостей, включая расчет тональности и фильтрацию по ключевым словам.
    """
    
    def __init__(self, keywords, ticker):
        self.__keywords = keywords[ticker]
        self.__driver = webdriver.Chrome()
        self.__analyzer = NewsSentimentAnalyzer()

    def parse_dataset(self, dataframe):
        """
        Обрабатывает набор данных новостей из DataFrame, выполняет анализ каждого URL.
        
        Args:
            dataframe (pd.DataFrame): DataFrame с новостями.
        
        Returns:
            pd.DataFrame: DataFrame с результатами анализа.
        """
        parsed_data = []
        for _, row in dataframe.iterrows():
            try:
                ticker = row['Ticker']
                title = row['Title']
                date = row['Date']
                time = row['Time']
                news_type = row['Type']
                url = row['URL']
                parsed_data.append(self.parse_url(ticker, title, date, time, news_type, url))
                if len(parsed_data) % 10 == 0:
                    print("Загружено", len(parsed_data))
                    pd.DataFrame(parsed_data).to_csv(f"news_data/{ticker}_news_in_progress.csv", index=False)
            except:
                pass
        return pd.DataFrame(parsed_data)

    def parse_url(self, ticker, title, date, time, news_type, url):
        """
        Загружает контент страницы по URL и выполняет анализ тональности.
        
        Args:
            ticker (str): Тикер компании.
            title (str): Заголовок новости.
            date (str): Дата новости.
            time (str): Время публикации.
            news_type (str): Тип новости.
            url (str): URL-адрес новости.
        
        Returns:
            dict: Результаты анализа тональности.
        """
        self.__driver.get(url)
        content_html = BeautifulSoup(self.__driver.page_source, 'html.parser')
        content = content_html.find("div", class_="YjHz UBOr RkGZ").text.strip()
        filtered_content = self.__extract_sentences(content)
        sentiment_df = self.__analyzer.analyze_sentiment(filtered_content)
        
        res_dict = {
            'ticker': ticker,
            'date': date,
            'time': time,
            'title': title,
            'news_type': news_type,
            'content': content,
            'filtered_content': filtered_content,
            'EN_filtered_content': list(sentiment_df['En Headline'])[0],
            'Positive': list(sentiment_df['Positive'])[0],
            'Negative': list(sentiment_df['Negative'])[0],
            'Neutral': list(sentiment_df['Neutral'])[0],
            'real_score': list(sentiment_df['real_score'])[0]
        }
        return res_dict

    def __extract_sentences(self, content):
        """
        Извлекает предложения с ключевыми словами.
        
        Args:
            content (str): Текст для фильтрации.
        
        Returns:
            str: Текст с предложениями, содержащими ключевые слова.
        """
        sentences = sent_tokenize(content)
        res_sentences = [sentence.replace("\n", "").replace("•", "").replace("\\x", "").replace("\\xa0", " ").replace("Краткосрочная картина", "") for sentence in sentences if any(keyword.lower() in sentence.lower() for keyword in self.__keywords)]
        return " ".join(res_sentences)

    def close_driver(self):
        """
        Завершает работу драйвера Selenium.
        """
        self.__driver.quit()
