import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go

from risk_metrics import calculate_returns, calculate_risk_summary

from sec_filing_loader import download_latest_10k
from filing_parser import (
    find_latest_filing_file,
    extract_text_from_filing,
    extract_risk_related_text
)

from rag_pipeline import retrieve_context, store_filing_text
from ai_explainer import generate_ai_analysis


st.set_page_config(
    page_title="AI Financial Risk Analyzer",
    page_icon="📈",
    layout="wide"
)


st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #F1F5F9;
            border-right: 1px solid #E5E7EB;
        }

        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 18px;
            color: #6B7280;
            margin-bottom: 28px;
            line-height: 1.5;
        }

        .footer {
            color: #6B7280;
            font-size: 13px;
            text-align: center;
            margin-top: 40px;
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def download_price_data(tickers, start_date, end_date):

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices


def get_risk_outlook(
    portfolio_annual_return,
    portfolio_annual_volatility,
    portfolio_max_drawdown,
    portfolio_sharpe_ratio,
    latest_volatility
):

    score = 0

    if portfolio_annual_return > 0.10:
        score += 1

    if portfolio_sharpe_ratio > 0.75:
        score += 1

    if portfolio_annual_volatility > 0.25:
        score -= 1

    if abs(portfolio_max_drawdown) > 0.35:
        score -= 1

    if latest_volatility > 0.20:
        score -= 1

    if score >= 2:
        return "Bullish / Constructive"

    elif score >= 0:
        return "Neutral / Mixed"

    else:
        return "Bearish / Cautious"


def run_monte_carlo_simulation(
    portfolio_returns,
    initial_value=10000,
    days=252,
    simulations=1000
):

    daily_mean = portfolio_returns.mean()
    daily_std = portfolio_returns.std()

    simulation_results = []

    for _ in range(simulations):

        simulated_daily_returns = np.random.normal(
            daily_mean,
            daily_std,
            days
        )

        simulated_growth = initial_value * (
            1 + simulated_daily_returns
        ).cumprod()

        simulation_results.append(simulated_growth)

    simulation_df = pd.DataFrame(simulation_results).T

    simulation_df.index = range(1, days + 1)

    return simulation_df


# HERO
st.markdown(
    """
    <div class="main-title">📈 AI Financial Risk Analyzer</div>
    <div class="subtitle">
        AI-powered portfolio risk intelligence using market data,
        volatility, diversification, benchmark comparison,
        future scenario modeling, SEC filing context, and AI-generated insights.
    </div>
    """,
    unsafe_allow_html=True
)


# SIDEBAR
st.sidebar.markdown("## Portfolio Setup")

ticker_input = st.sidebar.text_input(
    "Tickers",
    "AAPL, MSFT, NVDA, SPY"
)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark",
    "SPY"
).strip().upper()

selected_assets = [
    ticker.strip().upper()
    for ticker in ticker_input.split(",")
    if ticker.strip()
]

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2015-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.Timestamp.today()
)


# DOWNLOAD DATA
with st.spinner("Downloading market data..."):

    prices = download_price_data(
        selected_assets,
        start_date,
        end_date
    )

if prices.empty:
    st.error("No price data found for selected tickers.")
    st.stop()

selected_assets = list(prices.columns)


# PORTFOLIO WEIGHTS
st.sidebar.markdown("---")
st.sidebar.markdown("### Portfolio Weights")

default_weights = [
    round(1 / len(selected_assets), 2)
    for _ in selected_assets
]

weights_df = pd.DataFrame({
    "Ticker": selected_assets,
    "Weight": default_weights
})

weights_df.index = range(
    1,
    len(weights_df) + 1
)

weights_df = st.sidebar.data_editor(
    weights_df,
    num_rows="fixed",
    use_container_width=True
)

weights_array = np.array(
    weights_df["Weight"]
)

weights_array = (
    weights_array / weights_array.sum()
)


# CALCULATIONS
returns = calculate_returns(prices)

risk_summary, drawdowns = calculate_risk_summary(
    returns
)

portfolio_returns = returns.dot(
    weights_array
)

portfolio_cumulative = (
    1 + portfolio_returns
).cumprod()

portfolio_rolling_max = (
    portfolio_cumulative.cummax()
)

portfolio_drawdown = (
    portfolio_cumulative -
    portfolio_rolling_max
) / portfolio_rolling_max

portfolio_annual_return = (
    portfolio_returns.mean() * 252
)

portfolio_annual_volatility = (
    portfolio_returns.std() * np.sqrt(252)
)

risk_free_rate = 0.04

portfolio_max_drawdown = (
    portfolio_drawdown.min()
)

portfolio_sharpe_ratio = (
    (portfolio_annual_return - risk_free_rate)
    / portfolio_annual_volatility
)

portfolio_rolling_volatility = (
    portfolio_returns.rolling(30).std() * np.sqrt(252)
)

latest_volatility = (
    portfolio_rolling_volatility.iloc[-1]
)

risk_outlook = get_risk_outlook(
    portfolio_annual_return,
    portfolio_annual_volatility,
    portfolio_max_drawdown,
    portfolio_sharpe_ratio,
    latest_volatility
)


# KPI SECTION
st.subheader("🚦 Portfolio Snapshot")

st.caption(
    "High-level portfolio metrics based on historical market behavior."
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Annual Return",
    f"{portfolio_annual_return:.2%}"
)

kpi2.metric(
    "Risk Level",
    f"{portfolio_annual_volatility:.2%}"
)

kpi3.metric(
    "Worst Decline",
    f"{portfolio_max_drawdown:.2%}"
)

kpi4.metric(
    "Sharpe Ratio",
    f"{portfolio_sharpe_ratio:.2f}"
)

risk_score = min(
    int(portfolio_annual_volatility * 25) + 1,
    10
)

kpi5.metric(
    "Risk Score",
    f"{risk_score}/10"
)

st.markdown("---")


# OUTLOOK
st.subheader("🧭 AI Risk Outlook")

st.caption(
    "AI-generated assessment based on portfolio return, risk, and market conditions."
)

outlook_col1, outlook_col2 = st.columns(2)

outlook_col1.metric(
    "Portfolio Outlook",
    risk_outlook
)

if latest_volatility < 0.10:
    regime = "Low Risk"

elif latest_volatility < 0.20:
    regime = "Moderate Risk"

else:
    regime = "High Risk"

outlook_col2.metric(
    "Market Environment",
    regime
)

st.markdown("---")


# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Portfolio",
        "⚠️ Risk",
        "📈 Benchmark",
        "🧠 AI Analysis",
        "📄 Data",
        "🔮 Future Scenarios"
    ]
)


# PORTFOLIO TAB
with tab1:

    st.subheader("📊 Where Your Money Is Invested")

    st.caption(
        "Shows how your portfolio is allocated between investments."
    )

    pie_fig = px.pie(
        values=weights_array,
        names=selected_assets,
        hole=0.4
    )

    pie_fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📈 How Your Portfolio Grew Over Time")

    st.caption(
        "Tracks historical portfolio performance based on your selected investments and weights."
    )

    growth_fig = px.line(
        portfolio_cumulative,
        title="Portfolio Growth"
    )

    growth_fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Growth"
    )

    st.plotly_chart(
        growth_fig,
        use_container_width=True
    )


# RISK TAB
with tab2:

    st.subheader("⚠️ Investment Risk Breakdown")

    st.caption(
        "Compares return and risk characteristics across portfolio investments."
    )

    st.dataframe(risk_summary)

    st.markdown("---")

    st.subheader("📉 Biggest Drops From Previous Highs")

    st.caption(
        "Shows how far investments fell from their previous peaks during market declines."
    )

    drawdown_fig = px.line(
        drawdowns,
        title="Biggest Drops From Previous Highs"
    )

    drawdown_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        drawdown_fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🔥 How Risk Changed Over Time")

    st.caption(
        "Tracks how investment risk increased or decreased throughout different market periods."
    )

    rolling_volatility = (
        returns.rolling(30).std() * np.sqrt(252)
    )

    vol_fig = px.line(
        rolling_volatility,
        title="How Risk Changed Over Time"
    )

    vol_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        vol_fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🔗 How Investments Move Together")

    st.caption(
        "This shows whether investments tend to move in the same direction. "
        "Higher values mean they behave more similarly."
    )

    correlation_matrix = returns.corr()

    heatmap_fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    heatmap_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )


# BENCHMARK TAB
with tab3:

    st.subheader("📈 Portfolio vs Market Benchmark")

    st.caption(
        "Compares your portfolio performance against a selected market benchmark."
    )

    benchmark_prices = download_price_data(
        [benchmark_ticker],
        start_date,
        end_date
    )

    benchmark_fig = go.Figure()

    benchmark_fig.add_trace(
        go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative,
            mode="lines",
            name="Portfolio"
        )
    )

    if not benchmark_prices.empty:

        benchmark_returns = (
            benchmark_prices.pct_change().dropna()
        )

        benchmark_cumulative = (
            1 + benchmark_returns
        ).cumprod()

        benchmark_fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.iloc[:, 0],
                mode="lines",
                name=benchmark_ticker
            )
        )

    benchmark_fig.update_layout(
        title="Portfolio vs Benchmark",
        template="plotly_white"
    )

    st.plotly_chart(
        benchmark_fig,
        use_container_width=True
    )


# AI TAB
with tab4:

    st.subheader("🧠 AI Portfolio Analysis")

    st.caption(
        "AI-generated portfolio interpretation using market data, risk analysis, and SEC filing context."
    )

    if st.button(
        "Generate AI Analysis"
    ):

        with st.spinner(
            "Downloading SEC filings, retrieving context, and generating AI analysis..."
        ):

            for ticker in selected_assets:

                try:

                    download_latest_10k(ticker)

                    latest_file = find_latest_filing_file()

                    filing_text = extract_text_from_filing(
                        latest_file
                    )

                    risk_text = extract_risk_related_text(
                        filing_text
                    )

                    store_filing_text(
                        ticker,
                        risk_text
                    )

                except Exception as error:

                    st.warning(
                        f"Could not process SEC filing for {ticker}: {error}"
                    )

            query = f"""
            Analyze risk characteristics for:
            {selected_assets}

            Benchmark:
            {benchmark_ticker}

            Risk outlook:
            {risk_outlook}
            """

            retrieved_context = retrieve_context(
                query
            )

            summary_text = (
                risk_summary.round(4).to_string()
            )

            ai_analysis = generate_ai_analysis(
                summary_text,
                "",
                retrieved_context,
                regime,
                latest_volatility
            )

            st.info(
                "SEC filing context was retrieved and used to support the AI analysis. "
                "The raw filing excerpts are hidden to keep the report clean and readable."
            )

            st.subheader("AI Risk Analysis")

            st.write(ai_analysis)


# DATA TAB
with tab5:

    st.subheader("📄 Recent Market Data")

    st.caption(
        "Latest downloaded market pricing data from Yahoo Finance."
    )

    st.dataframe(
        prices.tail()
    )

    st.markdown("---")

    st.subheader("📈 Fair Growth Comparison")

    st.caption(
        "Compares investment growth by starting all investments at the same value."
    )

    normalized_prices = (
        prices / prices.iloc[0]
    )

    norm_fig = px.line(
        normalized_prices,
        title="Fair Growth Comparison"
    )

    norm_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        norm_fig,
        use_container_width=True
    )


# FUTURE SCENARIOS TAB
with tab6:

    st.subheader("🔮 Possible Future Portfolio Paths")

    st.caption(
        "This simulation runs many possible future portfolio paths "
        "based on historical return and volatility behavior."
    )

    simulation_df = run_monte_carlo_simulation(
        portfolio_returns,
        initial_value=10000,
        days=252,
        simulations=1000
    )

    monte_fig = go.Figure()

    for i in range(min(100, simulation_df.shape[1])):

        monte_fig.add_trace(
            go.Scatter(
                y=simulation_df.iloc[:, i],
                mode="lines",
                opacity=0.10,
                line=dict(width=1),
                showlegend=False
            )
        )

    monte_fig.update_layout(
        title="Possible Future Portfolio Paths",
        template="plotly_white"
    )

    st.plotly_chart(
        monte_fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📊 Possible Future Portfolio Outcomes")

    st.caption(
        "These scenarios estimate a range of plausible future portfolio values "
        "based on historical market behavior."
    )

    ending_values = simulation_df.iloc[-1]

    scenario_summary = pd.DataFrame({
        "Scenario": [
            "Conservative Scenario",
            "Expected Scenario",
            "Optimistic Scenario"
        ],
        "Portfolio Value": [
            ending_values.quantile(0.05),
            ending_values.mean(),
            ending_values.quantile(0.95)
        ]
    })

    scenario_summary.index = range(
        1,
        len(scenario_summary) + 1
    )

    st.dataframe(
        scenario_summary
    )

    st.warning(
        "These simulations are hypothetical and not guarantees or financial advice."
    )


# FOOTER
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        AI Financial Risk Analyzer • Built with Python, Streamlit, Plotly,
        Yahoo Finance, SEC EDGAR, quantitative risk analytics, RAG, and AI
        <br><br>
        This application is for educational and informational purposes only.
        It does not constitute financial, investment, or trading advice.
        Past performance does not guarantee future results.
    </div>
    """,
    unsafe_allow_html=True
)