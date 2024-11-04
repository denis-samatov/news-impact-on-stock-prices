import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.request import urlopen
from io import StringIO




# Словарь с идентификаторами акций на сайте для запроса данных
stocks_url_dict = {
    'SBER': 3,
    'GAZP': 16842,
    'LKOH': 8,
    'MTSS': 15523
}

class StockParser:
    """
    Класс для парсинга данных по акциям с сайта, построения графиков цен закрытия и сохранения данных в CSV файлы.
    """

    def __init__(self, tickers: list, start_date=None, end_date=None):
        """
        Инициализация класса StockParser.
        Аргументы:
            tickers (list): Список тикеров акций для анализа.
            start_date (str): Начальная дата в формате 'YYYY-MM-DD'. Если не указана, используется текущая дата.
            end_date (str): Конечная дата в формате 'YYYY-MM-DD'. Если не указана, используется текущая дата.
        """
        self.__tickers = tickers
        self.__year_start = start_date[0:4] if start_date is not None else str(datetime.now().year)
        self.__month_start = start_date[5:7] if start_date is not None else str(datetime.now().month).zfill(2)
        self.__day_start = start_date[8:10] if start_date is not None else str(datetime.now().day).zfill(2)
        self.__year_end = end_date[0:4] if end_date is not None else str(datetime.now().year)
        self.__month_end = end_date[5:7] if end_date is not None else str(datetime.now().month).zfill(2)
        self.__day_end = end_date[8:10] if end_date is not None else str(datetime.now().day).zfill(2)
        self.__data = {}

    def generate_url(self, ticker):
        """
        Генерирует URL для загрузки данных по заданному тикеру.
        Аргументы:
            ticker (str): Тикер акции для построения URL.
        Возвращает:
            str: Сформированный URL.
        """
        if ticker not in stocks_url_dict:
            raise ValueError(f"Тикер {ticker} не найден в словаре stocks_url_dict")

        url_template = ('https://export.finam.ru/export9.out?apply=0&p=8&e=.csv&dtf=2&tmf=1&MSOR=0&mstime=on&'
                        'mstimever=1&sep=3&sep2=1&datf=1&at=0&from={start_date}&to={end_date}&market=0&em={em}&'
                        'code={ticker}&f={ticker}&cn={ticker}&yf={year_start}&yt={year_end}&df={day_start}&dt={day_end}&mf=0&mt=0')

        url = url_template.format(
            start_date=f"{self.__day_start}.{self.__month_start}.{self.__year_start}",
            end_date=f"{self.__day_end}.{self.__month_end}.{self.__year_end}",
            em=stocks_url_dict[ticker],
            ticker=ticker,
            year_start=self.__year_start,
            year_end=self.__year_end,
            day_start=self.__day_start,
            day_end=self.__day_end
        )
        return url

    def parse_stocks(self):
        """
        Загружает данные по каждому тикеру из списка, парсит CSV, преобразует дату и сохраняет в словарь.
        """
        for ticker in self.__tickers:
            url = self.generate_url(ticker)
            try:
                with urlopen(url) as page:
                    content = page.read().decode('utf-8')
                    df = pd.read_csv(StringIO(content), delimiter=';', names=['ticker', 'per', 'date', 'time', 'open', 'high', 'low', 'close', 'vol'])
                    
                    df['date'] = pd.to_datetime(df['date'], format='%y%m%d')
                    self.__data[ticker] = df
            except Exception as e:
                print(f"Ошибка при загрузке данных для {ticker}: {e}")

    def plot_stock_data(self):
        """
        Строит график цен закрытия для каждого тикера в self.__tickers.
        """
        if not self.__data:
            print("Пожалуйста, сначала выполните parse_stocks().")
            return

        plt.figure(figsize=(10, 6))
        
        for ticker in self.__tickers:
            if ticker in self.__data:
                plt.plot(self.__data[ticker]['date'], self.__data[ticker]['close'], label=ticker, linewidth=2)

        plt.title('Цены закрытия акций', fontsize=16)
        plt.xlabel('Дата', fontsize=14)
        plt.ylabel('Цена закрытия', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.style.use('seaborn-darkgrid')
        plt.show()

    def get_stocks_dataframe(self):
        """
        Возвращает словарь с данными по акциям.
        Возвращает:
            dict: Словарь с DataFrame для каждого тикера.
        """
        if self.__data:
            return self.__data
        else:
            print("Пожалуйста, сначала выполните parse_stocks().")

    def save_to_csv(self, output_folder):
        """
        Сохраняет данные по акциям в CSV файлы.
        Аргументы:
            output_folder (str): Папка для сохранения файлов.
        """
        if not self.__data:
            print("Пожалуйста, сначала выполните parse_stocks().")
            return

        os.makedirs(output_folder, exist_ok=True)
        for ticker, df in self.__data.items():
            output_filename = f'{output_folder}/{ticker}_stock_data.csv'
            try:
                df.to_csv(output_filename, index=False)
                print(f"Данные для {ticker} сохранены в {output_filename}")
            except Exception as e:
                print(f"Ошибка при сохранении данных для {ticker}: {e}")
