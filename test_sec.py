import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from sec_filing_loader import download_latest_10k
from filing_parser import (
    find_latest_filing_file,
    extract_text_from_filing,
    extract_risk_related_text
)
from rag_pipeline import (
    store_filing_text,
    retrieve_context
)

ticker = "AAPL"

download_latest_10k(ticker)

latest_file = find_latest_filing_file()

filing_text = extract_text_from_filing(
    latest_file
)

risk_text = extract_risk_related_text(
    filing_text
)

chunk_count = store_filing_text(
    ticker,
    risk_text
)

print(f"Stored {chunk_count} chunks for {ticker}")

query = "What are the main risk factors for Apple?"

context = retrieve_context(
    query
)

print(context[:3000])