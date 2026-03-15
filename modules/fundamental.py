"""
modules/fundamental.py — Fundamental Analysis (v4 — Better Validation)
======================================================================
[DQC] v4: Cross-validate PE with computed PE (price/EPS).
Cap all ratios at reasonable limits. Flag anomalies clearly.
"""

import yfinance as yf
import pandas as pd


def validate_percentage(value, max_reasonable=100.0):
    """
    [DQC] Validate percentage values from yfinance.
    If value > 1.0, it's likely already in percent format.
    Cap at max_reasonable to flag data anomalies.
    """
    if not isinstance(value, (int, float)):
        return "N/A"
    if abs(value) > 1.0:
        result = round(value, 2)
    else:
        result = round(value * 100, 2)
    if abs(result) > max_reasonable:
        return "N/A (Anomali)"
    return result


def validate_ratio(value, max_reasonable=500.0, min_reasonable=-500.0):
    """
    [DQC v4] Validate non-percentage ratios.
    Extreme values (PE > 200, PBV > 100) are likely data errors.
    """
    if not isinstance(value, (int, float)):
        return "N/A"
    if value > max_reasonable or value < min_reasonable:
        return "N/A (Anomali)"
    return round(value, 2)


def get_stock_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Nama Perusahaan": info.get("longName", "N/A"),
        "Sektor": info.get("sector", "N/A"),
        "Industri": info.get("industry", "N/A"),
        "Market Cap": info.get("marketCap", 0),
        "Mata Uang": info.get("currency", "IDR"),
        "Harga Terakhir": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "52W High": info.get("fiftyTwoWeekHigh", 0),
        "52W Low": info.get("fiftyTwoWeekLow", 0),
        "Website": info.get("website", "N/A"),
    }


def get_financial_ratios(ticker: str) -> dict:
    """
    [DQC v4] Enhanced ratio validation:
    - Cross-validate PE with price/EPS calculation
    - Cap all ratios at reasonable limits
    - Flag anomalies instead of showing misleading numbers
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    def safe_get(key, default="N/A"):
        val = info.get(key)
        return default if val is None else val

    # Get raw values
    raw_pe = safe_get("trailingPE")
    raw_fwd_pe = safe_get("forwardPE")
    raw_eps = safe_get("trailingEps")
    raw_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
    raw_pbv = safe_get("priceToBook")
    raw_de = safe_get("debtToEquity")

    # Cross-validate PE: if yfinance PE looks wrong, compute from price/EPS
    computed_pe = "N/A"
    if isinstance(raw_eps, (int, float)) and raw_eps > 0 and raw_price > 0:
        computed_pe = round(raw_price / raw_eps, 2)

    # Use computed PE if yfinance PE is anomalous (>200 or negative when EPS positive)
    final_pe = raw_pe
    if isinstance(raw_pe, (int, float)):
        if raw_pe > 200 or raw_pe < 0:
            if isinstance(computed_pe, (int, float)):
                final_pe = computed_pe  # Use computed instead
            else:
                final_pe = "N/A (Anomali)"
        else:
            final_pe = round(raw_pe, 2)
    
    ratios = {
        "PBV (Price/Book)": validate_ratio(raw_pbv, max_reasonable=100),
        "PE Ratio (TTM)": final_pe if isinstance(final_pe, str) else round(final_pe, 2),
        "PE Ratio (Forward)": validate_ratio(raw_fwd_pe, max_reasonable=200),
        "Debt to Equity": validate_ratio(raw_de, max_reasonable=1000),
        "EPS (TTM)": safe_get("trailingEps"),
        "Book Value": safe_get("bookValue"),
    }

    # Percentage ratios with validation
    pct_ratios = {
        "ROE (%)": (safe_get("returnOnEquity"), 200),
        "ROA (%)": (safe_get("returnOnAssets"), 100),
        "Dividend Yield (%)": (safe_get("dividendYield"), 50),
        "Profit Margin (%)": (safe_get("profitMargins"), 200),
    }
    for key, (val, cap) in pct_ratios.items():
        if val == "N/A":
            ratios[key] = "N/A"
        else:
            ratios[key] = validate_percentage(val, cap)

    # Round remaining floats
    for key, val in ratios.items():
        if isinstance(val, float):
            ratios[key] = round(val, 2)

    return ratios


def get_financials_summary(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    summary_data = []
    if income_stmt is not None and not income_stmt.empty:
        for col in income_stmt.columns:
            year = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
            row = {"Tahun": year}
            for rev_key in ["Total Revenue", "Operating Revenue"]:
                if rev_key in income_stmt.index:
                    row["Revenue"] = income_stmt.loc[rev_key, col]; break
            else: row["Revenue"] = None
            for ni_key in ["Net Income", "Net Income Common Stockholders"]:
                if ni_key in income_stmt.index:
                    row["Net Income"] = income_stmt.loc[ni_key, col]; break
            else: row["Net Income"] = None
            if balance_sheet is not None and not balance_sheet.empty and col in balance_sheet.columns:
                for dk in ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"]:
                    if dk in balance_sheet.index:
                        row["Total Debt"] = balance_sheet.loc[dk, col]; break
                else: row["Total Debt"] = None
            else: row["Total Debt"] = None
            summary_data.append(row)
    return pd.DataFrame(summary_data)


def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if hist.empty: return pd.DataFrame()
    return hist.sort_index(ascending=True)


def format_large_number(num) -> str:
    if num is None or pd.isna(num): return "N/A"
    num = float(num)
    if abs(num) >= 1e12: return f"Rp {num/1e12:,.2f} T"
    elif abs(num) >= 1e9: return f"Rp {num/1e9:,.2f} M"
    elif abs(num) >= 1e6: return f"Rp {num/1e6:,.2f} Jt"
    else: return f"Rp {num:,.0f}"
