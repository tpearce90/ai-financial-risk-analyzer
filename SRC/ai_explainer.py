from openai import OpenAI
from dotenv import load_dotenv

import os

try:
    import streamlit as st
except ImportError:
    st = None


load_dotenv()


def get_openai_api_key():

    if st is not None:
        try:
            if "OPENAI_API_KEY" in st.secrets:
                return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

    return os.getenv("OPENAI_API_KEY")


client = OpenAI(
    api_key=get_openai_api_key()
)


def generate_ai_analysis(
    summary_text,
    portfolio_text,
    retrieved_context,
    regime,
    latest_volatility
):

    prompt = f"""
    You are a professional financial risk analyst.

    Analyze the following portfolio risk metrics and SEC filing context.

    Current Market Regime:
    {regime}

    Current 30-Day Annualized Portfolio Volatility:
    {latest_volatility:.2%}

    Risk Summary:
    {summary_text}

    Portfolio Summary:
    {portfolio_text}

    Retrieved SEC Filing Context:
    {retrieved_context}

    Explain:
    - which assets appear riskiest
    - which assets appear most defensive
    - diversification behavior
    - overall portfolio stability
    - volatility observations
    - drawdown observations
    - macroeconomic risks
    - interest-rate sensitivity
    - how the current market regime affects portfolio risk
    - portfolio strengths and weaknesses
    """

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    return response.output_text