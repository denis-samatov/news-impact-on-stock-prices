import os
from stocks_parser import StockParser  # Импортируем StockParser из файла, где он определен

def main():
    output_folder = "stock_data"
    
    tickers = ["SBER", "GAZP", "LKOH", "MTSS"]
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    stock_parser = StockParser(tickers=tickers, start_date=start_date, end_date=end_date)

    stock_parser.parse_stocks()
    
    stock_parser.save_to_csv(output_folder)
     
    print(f"Сбор данных завершен. Данные сохранены в папке {output_folder}.")

if __name__ == "__main__":
    main()
