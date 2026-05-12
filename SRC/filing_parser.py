from pathlib import Path
from bs4 import BeautifulSoup


RISK_KEYWORDS = [
    "risk factors",
    "item 1a",
    "market risk",
    "competition",
    "supply chain",
    "macroeconomic",
    "interest rates",
    "inflation",
    "cybersecurity",
    "regulatory",
    "liquidity",
    "credit risk"
]


def find_latest_filing_file(base_path="sec_filings"):

    base = Path(base_path)

    filing_files = list(
        base.rglob("*.txt")
    )

    if not filing_files:
        return None

    latest_file = max(
        filing_files,
        key=lambda file: file.stat().st_mtime
    )

    return latest_file


def extract_text_from_filing(file_path):

    file_path = Path(file_path)

    raw_text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    soup = BeautifulSoup(
        raw_text,
        "lxml"
    )

    clean_text = soup.get_text(
        separator=" "
    )

    clean_text = " ".join(
        clean_text.split()
    )

    return clean_text


def extract_risk_related_text(text, window_size=1200):

    text_lower = text.lower()

    snippets = []

    for keyword in RISK_KEYWORDS:

        start = text_lower.find(keyword)

        if start != -1:

            snippet_start = max(
                start - 300,
                0
            )

            snippet_end = min(
                start + window_size,
                len(text)
            )

            snippet = text[
                snippet_start:snippet_end
            ]

            snippets.append(snippet)

    combined_text = "\n\n---\n\n".join(
        snippets
    )

    return combined_text