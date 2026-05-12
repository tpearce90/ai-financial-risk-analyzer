import yfinance as yf
import pandas as pd

from config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, RAW_DATA_PATH


def download_price_data():
    """
    Downloads historical price data for selected ETFs using yfinance.
    Saves the closing prices to a CSV file.
    """

    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )

    prices = data[PRICE_FIELD]

    prices.to_csv(RAW_DATA_PATH)

    return prices


if __name__ == "__main__":
    prices = download_price_data()
    print(prices.head())
    print("Market data downloaded successfully.")