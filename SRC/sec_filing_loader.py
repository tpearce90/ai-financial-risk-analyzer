from sec_edgar_downloader import Downloader
from pathlib import Path


def download_latest_10k(ticker):

    download_path = "sec_filings"

    dl = Downloader(
        "AI Financial Risk Analyzer",
        "tpearce111590@gmail.com",
        download_path
    )

    dl.get(
        "10-K",
        ticker,
        limit=1,
        download_details=False
    )

    company_folder = Path(download_path) / "sec-edgar-filings"

    return company_folder