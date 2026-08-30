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


# Class for collecting news from a given URL and filtering by keywords.
class NewsParser:
    """
    Class for parsing news from a given URL. Saves parsing results and creates
    intermediate saves to handle large volumes of data.
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
        Filters sentences containing keywords out of the news text.

        Args:
            content (str): News text.

        Returns:
            list: List of filtered sentences.
        """
        sentences = sent_tokenize(content)
        res_sentences = [sentence for sentence in sentences if any(keyword.lower() in sentence.lower() for keyword in self.__keywords)]
        return res_sentences

    def __format_date(self, date):
        """
        Converts a text date into the standard `YYYY-MM-DD` format, accounting for
        relative dates.

        Note: the source site publishes dates in Russian (e.g. "Сегодня" = "Today",
        "Вчера" = "Yesterday", and Russian month names), so these string literals
        match real site content and are intentionally not translated.

        Args:
            date (str): Date as text.

        Returns:
            str: Date in `YYYY-MM-DD` format.
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
        Extracts the time from a text date, if present.

        Args:
            date (str): Date as text.

        Returns:
            str: Time in `HH:MM` format.
        """
        if ' в ' in date:
            time_str = date.split(' в ')[1]
            time = datetime.strptime(time_str, '%H:%M').strftime('%H:%M')
        else:
            time = None

        return time

    def __parse_news(self, news_block):
        """
        Extracts data for one news item from an HTML block.

        Args:
            news_block (BeautifulSoup): HTML block with the news item's data.

        Returns:
            list: [date, time, title, link, news type].
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
        Sequentially parses news from the site, loading more data and checking
        stop conditions.

        Args:
            len_dataset (int): Target number of news items to parse.
            stop_date (str): Date after which parsing stops.
        """
        self.__driver = webdriver.Chrome()
        self.__driver.get(self.__url)

        while len(self.__news) < len_dataset:
            # XPath text below matches the site's actual Russian-language
            # "Show more news" button label, so it's left untranslated.
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
                print("Loaded:", len(self.__news))
                self.save_to_csv('news_data', f'{self.ticker}_in_progress.csv')

        self.__driver.quit()

    def get_news_list(self):
        """
        Returns the list of collected news items.

        Returns:
            list: List of collected news items.
        """
        return self.__news

    def save_to_csv(self, output_dir, output_filename):
        """
        Saves the news data to a CSV file, creating the `news_data` directory if
        it doesn't already exist.

        Args:
            output_filename (str): Name of the CSV file to save.
        """

        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)

        if self.__news:
            df = pd.DataFrame(self.__news, columns=['Date', 'Time', 'Title', 'URL', 'Type'])
            df.to_csv(output_path, index=False)
        else:
            print("No news data to save.")


# Class for analyzing news content, including computing news sentiment.
class HelperParser:
    """
    Class for analyzing news, including sentiment scoring and keyword filtering.
    """

    def __init__(self, keywords, ticker):
        self.__keywords = keywords[ticker]
        self.__driver = webdriver.Chrome()
        self.__analyzer = NewsSentimentAnalyzer()

    def parse_dataset(self, dataframe):
        """
        Processes a news dataset from a DataFrame, analyzing each URL.

        Args:
            dataframe (pd.DataFrame): DataFrame of news items.

        Returns:
            pd.DataFrame: DataFrame of analysis results.
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
                    print("Loaded", len(parsed_data))
                    pd.DataFrame(parsed_data).to_csv(f"news_data/{ticker}_news_in_progress.csv", index=False)
            except:
                pass
        return pd.DataFrame(parsed_data)

    def parse_url(self, ticker, title, date, time, news_type, url):
        """
        Loads the page content at a URL and runs sentiment analysis on it.

        Args:
            ticker (str): Company ticker.
            title (str): News headline.
            date (str): News date.
            time (str): Publication time.
            news_type (str): News type.
            url (str): News URL.

        Returns:
            dict: Sentiment analysis results.
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
        Extracts sentences containing keywords.

        Note: "Краткосрочная картина" ("Short-term picture") below is a
        boilerplate heading the source site inserts into articles; it's
        stripped out as noise, not translated, since it's matched against
        real Russian-language page content.

        Args:
            content (str): Text to filter.

        Returns:
            str: Text made up of the sentences containing keywords.
        """
        sentences = sent_tokenize(content)
        res_sentences = [sentence.replace("\n", "").replace("•", "").replace("\\x", "").replace("\\xa0", " ").replace("Краткосрочная картина", "") for sentence in sentences if any(keyword.lower() in sentence.lower() for keyword in self.__keywords)]
        return " ".join(res_sentences)

    def close_driver(self):
        """
        Shuts down the Selenium driver.
        """
        self.__driver.quit()
