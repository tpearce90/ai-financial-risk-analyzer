import numpy as np
import pandas as pd


def calculate_returns(prices):
    return prices.pct_change().dropna()


def calculate_risk_summary(returns):
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)

    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    risk_summary = pd.DataFrame({
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Max Drawdown": max_drawdown
    })

    risk_summary["Return / Volatility"] = (
        risk_summary["Annual Return"] / risk_summary["Annual Volatility"]
    )

    return risk_summary, drawdowns

