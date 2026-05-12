# AI Financial Risk Analyzer

A deployed AI-powered portfolio risk analytics app built with Python, Streamlit, Plotly, Yahoo Finance, SEC EDGAR filings, ChromaDB, and OpenAI.

Live App: https://ai-financial-risk-analyzer.streamlit.app/

## Overview

This project analyzes custom stock and ETF portfolios using quantitative risk metrics, interactive visualizations, benchmark comparison, future scenario modeling, and AI-generated explanations supported by SEC filing context.

The goal is to simulate how an AI-assisted financial risk analyst could combine market data, portfolio analytics, and company risk disclosures into a single dashboard.

## Features

- Custom stock and ETF ticker input
- Portfolio weight editing
- Historical return analysis
- Annualized volatility / risk level
- Worst historical decline / drawdown analysis
- Sharpe ratio
- Risk score
- Portfolio vs benchmark comparison
- Interactive Plotly charts
- Correlation analysis
- Monte Carlo future scenario simulation
- SEC 10-K filing retrieval
- RAG pipeline using ChromaDB
- AI-generated portfolio risk analysis
- Public Streamlit deployment

## Tech Stack

- Python
- Streamlit
- pandas
- numpy
- Plotly
- yfinance
- SEC EDGAR
- ChromaDB
- OpenAI API
- BeautifulSoup
- sec-edgar-downloader

## Architecture

## Architecture

```text
User-selected stock / ETF tickers
        ↓
Yahoo Finance market data retrieval
        ↓
Portfolio risk metrics and analytics
        ↓
SEC 10-K filing retrieval
        ↓
Risk-section extraction and text parsing
        ↓
ChromaDB vector storage and retrieval
        ↓
OpenAI-generated risk analysis using retrieved context
        ↓
Interactive Streamlit dashboard
```

## Dashboard Preview

### Main Dashboard

![Main Dashboard](screenshots/Main Dashboard.png)

### Portfolio Allocation

![Where Your Money Is Invested](screenshots/Where Your Money Is Invested.png)

### Portfolio Growth

![How Your Portfolio Grew Over Time](screenshots/How Your Portfolio Grew Over Time.png)

### Investment Risk Breakdown

![Investment Risk Breakdown](screenshots/Investment Risk Breakdown.png)

### Biggest Historical Declines

![Biggest Drops From Previous Highs](screenshots/Biggest Drops From Previous Highs.png)

### Rolling Risk Trends

![How Risk Changed Over Time](screenshots/How Risk Changed Over Time.png)

### Correlation Heatmap

![How Investments Move Together](screenshots/How Investments Move Together.png)

### Portfolio vs Benchmark

![Portfolio vs Market Benchmark](screenshots/Portfolio vs Market Benchmark.png)

### AI Risk Analysis

![AI Portfolio Analysis](screenshots/AI Portfolio Analysis.png)

### Future Portfolio Simulations

![Possible Future Portfolio Paths](screenshots/Possible Future Portfolio Paths.png)

### Future Portfolio Outcomes

![Possible Future Portfolio Outcomes](screenshots/Possible Future Portfolio Outcomes.png)