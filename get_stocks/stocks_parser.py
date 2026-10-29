import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.request import urlopen
from io import StringIO




# Dictionary of stock IDs on the site, used to request data
stocks_url_dict = {
    'SBER': 3,
    'GAZP': 16842,
    'LKOH': 8,
    'MTSS': 15523
}

class StockParser:
    """
    Class for parsing stock data from the site, plotting closing prices, and saving data to CSV files.
    """

    def __init__(self, tickers: list, start_date=None, end_date=None):
        """
        Initialize the StockParser class.
        Args:
            tickers (list): List of stock tickers to analyze.
            start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to the current date if not given.
            end_date (str): End date in 'YYYY-MM-DD' format. Defaults to the current date if not given.
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
        Generates the download URL for the given ticker's data.
        Args:
            ticker (str): The stock ticker to build the URL for.
        Returns:
            str: The resulting URL.
        """
        if ticker not in stocks_url_dict:
            raise ValueError(f"Ticker {ticker} not found in stocks_url_dict")

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
        Downloads data for each ticker in the list, parses the CSV, converts the date, and stores it in a dict.
        """
        for ticker in self.__tickers:
            url = self.generate_url(ticker)
            try:
                with urlopen(url, timeout=30) as page:
                    payload = page.read(16 * 1024 * 1024 + 1)
                    if len(payload) > 16 * 1024 * 1024:
                        raise ValueError('stock response exceeds download limit')
                    content = payload.decode('utf-8')
                    df = pd.read_csv(StringIO(content), delimiter=';', names=['ticker', 'per', 'date', 'time', 'open', 'high', 'low', 'close', 'vol'])

                    df['date'] = pd.to_datetime(df['date'], format='%y%m%d')
                    self.__data[ticker] = df
            except Exception as e:
                print(f"Error loading data for {ticker}: {e}")

    def plot_stock_data(self):
        """
        Plots closing prices for each ticker in self.__tickers.
        """
        if not self.__data:
            print("Please run parse_stocks() first.")
            return

        plt.figure(figsize=(10, 6))

        for ticker in self.__tickers:
            if ticker in self.__data:
                plt.plot(self.__data[ticker]['date'], self.__data[ticker]['close'], label=ticker, linewidth=2)

        plt.title('Stock closing prices', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Closing price', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.style.use('seaborn-darkgrid')
        plt.show()

    def get_stocks_dataframe(self):
        """
        Returns the dict of stock data.
        Returns:
            dict: Dict of DataFrames, one per ticker.
        """
        if self.__data:
            return self.__data
        else:
            print("Please run parse_stocks() first.")

    def save_to_csv(self, output_folder):
        """
        Saves stock data to CSV files.
        Args:
            output_folder (str): Folder to save the files to.
        """
        if not self.__data:
            print("Please run parse_stocks() first.")
            return

        os.makedirs(output_folder, exist_ok=True)
        for ticker, df in self.__data.items():
            output_filename = f'{output_folder}/{ticker}_stock_data.csv'
            try:
                df.to_csv(output_filename, index=False)
                print(f"Data for {ticker} saved to {output_filename}")
            except Exception as e:
                print(f"Error saving data for {ticker}: {e}")
