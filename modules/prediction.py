"""
modules/prediction.py — Price Prediction v5 (Complete)
=======================================================
v5: Fixed stop loss (neutral=long), added entry scenarios per strategy,
best entry calculation with multiple scenarios.
"""

import numpy as np
import pandas as pd
from datetime import timedelta


def predict_price_linear(df: pd.DataFrame, days: int = 30) -> dict:
    if df.empty or "Close" not in df.columns or len(df) < 20:
        return {"predicted_prices": pd.DataFrame(), "trend_line": pd.Series(dtype=float),
                "slope_per_day": 0, "r_squared": 0, "direction": "N/A",
                "predicted_price_end": None, "confidence": "N/A"}

    close = df["Close"].dropna().values
    x = np.arange(len(close))
    
    # v10: Weighted Linear Regression (Recent data has more weight)
    weights = np.ones(len(close))
    if len(close) > 14:
        weights[-14:] *= 3.0
    if len(close) > 30:
        weights[-30:-14] *= 2.0
        
    coeffs = np.polyfit(x, close, 1, w=weights)
    slope, intercept = coeffs[0], coeffs[1]

    y_pred = slope * x + intercept
    ss_res = np.sum((close - y_pred) ** 2)
    ss_tot = np.sum((close - np.mean(close)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    trend_line = pd.Series(y_pred, index=df.index[-len(close):])
    future_x = np.arange(len(close), len(close) + days)
    future_prices = slope * future_x + intercept

    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
    if len(future_dates) < days:
        extra = pd.date_range(start=future_dates[-1] + timedelta(days=1), periods=days - len(future_dates))
        future_dates = future_dates.append(extra)
    future_dates = future_dates[:days]

    rmse = np.sqrt(ss_res / len(close)) if len(close) > 0 else 0
    uncertainty_multiplier = np.linspace(1, 2, days)

    pred_df = pd.DataFrame({
        "Date": future_dates, "Predicted_Price": future_prices,
        "Upper_Bound": future_prices + (rmse * 2 * uncertainty_multiplier), 
        "Lower_Bound": future_prices - (rmse * 2 * uncertainty_multiplier),
    })

    direction = "📈 Uptrend" if slope > 0 else ("📉 Downtrend" if slope < 0 else "➡️ Sideways")
    change_pct = round((future_prices[-1] - close[-1]) / close[-1] * 100, 2)
    
    # Volatility penalty
    is_highly_volatile = False
    if "HV_20" in df.columns and len(df) > 0:
        if df["HV_20"].iloc[-1] > 40:
            is_highly_volatile = True
            
    if is_highly_volatile:
        confidence = "🟡 Sedang (Volatilitas Tinggi)" if r_squared > 0.7 else "🔴 Rendah (Volatilitas Tinggi)"
    else:
        confidence = "🟢 Tinggi" if r_squared > 0.7 else ("🟡 Sedang" if r_squared > 0.4 else "🔴 Rendah")

    return {
        "predicted_prices": pred_df, "trend_line": trend_line,
        "slope_per_day": round(slope, 2), "intercept": intercept, "r_squared": round(r_squared, 4),
        "direction": direction, "predicted_price_end": future_prices[-1],
        "change_pct": change_pct, "current_price": close[-1],
        "confidence": confidence
    }


def predict_price_polynomial(df: pd.DataFrame, days: int = 30) -> dict:
    """v11: 2nd degree polynomial regression to detect acceleration/deceleration."""
    if df.empty or len(df) < 30:
        return {"curve_type": "N/A", "acceleration": 0}
        
    close = df["Close"].dropna().values
    x = np.arange(len(close))
    
    coeffs = np.polyfit(x, close, 2)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    
    if a > 0.01: 
        curve_type = "🚀 Akselerasi (Menguat)"
    elif a < -0.01: 
        curve_type = "🥀 Deselerasi (Melemah)"
    else:
        curve_type = "➡️ Stabil (Linear)"
        
    return {"curve_type": curve_type, "acceleration": a}

def predict_monte_carlo(df: pd.DataFrame, days: int = 30, simulations: int = 1000) -> dict:
    """
    Monte Carlo Simulation untuk memprediksi probabilitas harga di masa depan.
    Berdasarkan Geometric Brownian Motion (GBM).
    """
    if df.empty or "Close" not in df.columns or len(df) < 50:
        return {"dates": [], "median": [], "best_case": [], "worst_case": [], "last_price": 0, "final_median": 0, "final_best": 0, "final_worst": 0}

    close = df["Close"].dropna()
    
    # Calculate daily returns
    returns = close.pct_change().dropna()
    
    # Calculate drift and volatility
    mu = returns.mean()
    sigma = returns.std()
    
    # Drift for GBM math: mu - (sigma^2 / 2)
    drift = mu - (0.5 * sigma**2)
    
    last_price = float(close.iloc[-1])
    
    # Generate random shocks for N simulations and T days
    # Z = standard normal distribution
    Z = np.random.normal(0, 1, (days, simulations))
    
    # Calculate daily simulated returns
    daily_returns = np.exp(drift + sigma * Z)
    
    # Create price paths starting from 1 (we'll multiply by last_price)
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price * daily_returns[0]
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    # Extract percentiles across all simulations for each day
    percentile_10 = np.percentile(price_paths, 10, axis=1) # Worst Case (10th percentile = 90% confidence it won't be worse)
    percentile_50 = np.percentile(price_paths, 50, axis=1) # Median
    percentile_90 = np.percentile(price_paths, 90, axis=1) # Best Case (90th percentile)
    
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
    if len(future_dates) < days:
        extra = pd.date_range(start=future_dates[-1] + timedelta(days=1), periods=days - len(future_dates))
        future_dates = future_dates.append(extra)
    future_dates = future_dates[:days]
    
    return {
        "dates": future_dates.tolist(),
        "median": percentile_50.tolist(),
        "best_case": percentile_90.tolist(),
        "worst_case": percentile_10.tolist(),
        "last_price": last_price,
        "days": days,
        "simulations": simulations,
        "final_median": round(percentile_50[-1], 2),
        "final_best": round(percentile_90[-1], 2),
        "final_worst": round(percentile_10[-1], 2),
    }



def predict_price_ma_projection(df: pd.DataFrame, ma_period: int = 20, days: int = 30) -> dict:
    if df.empty or "Close" not in df.columns or len(df) < ma_period + 10:
        return {"predicted_prices": pd.DataFrame(), "ma_direction": "N/A",
                "ma_slope_daily": 0, "predicted_price_end": None}
    import pandas_ta as ta
    ma = ta.sma(df["Close"], length=ma_period).dropna()
    if len(ma) < 10:
        return {"predicted_prices": pd.DataFrame(), "ma_direction": "N/A",
                "ma_slope_daily": 0, "predicted_price_end": None}
    recent_ma = ma.tail(10).values
    ma_slope = np.mean(np.diff(recent_ma))
    last_ma = ma.iloc[-1]
    future_prices = [last_ma + ma_slope * (i + 1) for i in range(days)]
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
    if len(future_dates) < days:
        extra = pd.date_range(start=future_dates[-1] + timedelta(days=1), periods=days - len(future_dates))
        future_dates = future_dates.append(extra)
    future_dates = future_dates[:days]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_prices})
    direction = "📈 MA Uptrend" if ma_slope > 0 else ("📉 MA Downtrend" if ma_slope < 0 else "➡️ MA Flat")
    return {"predicted_prices": pred_df, "ma_direction": direction,
            "ma_slope_daily": round(ma_slope, 2),
            "predicted_price_end": round(future_prices[-1], 2) if future_prices else None}


def get_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    if df.empty or len(df) < window:
        return {"support_1": None, "support_2": None, "resistance_1": None,
                "resistance_2": None, "pivot_point": None}
    recent = df.tail(window)
    high, low, close = recent["High"].max(), recent["Low"].min(), recent["Close"].iloc[-1]
    pivot = (high + low + close) / 3
    r1, r2 = 2 * pivot - low, pivot + (high - low)
    s1, s2 = 2 * pivot - high, pivot - (high - low)
    return {"support_1": round(s1, 2), "support_2": round(s2, 2),
            "resistance_1": round(r1, 2), "resistance_2": round(r2, 2),
            "pivot_point": round(pivot, 2)}


def get_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Calculate Fibonacci Retracement levels from recent swing high/low."""
    if df.empty or len(df) < lookback:
        return {"fib_0": None, "fib_236": None, "fib_382": None, "fib_500": None, "fib_618": None, "fib_100": None}
    
    recent = df.tail(lookback)
    swing_high = recent["High"].max()
    swing_low = recent["Low"].min()
    diff = swing_high - swing_low
    
    return {
        "fib_0": round(swing_high, 2),          # 0% (Top)
        "fib_236": round(swing_high - diff * 0.236, 2),
        "fib_382": round(swing_high - diff * 0.382, 2),
        "fib_500": round(swing_high - diff * 0.500, 2),
        "fib_618": round(swing_high - diff * 0.618, 2),
        "fib_100": round(swing_low, 2),          # 100% (Bottom)
    }


def get_unified_confidence(linear_result: dict, monte_carlo_result: dict) -> dict:
    """
    Blend Linear R² with Monte Carlo spread to produce a single confidence %.
    - High R² + Tight MC spread = High confidence
    - Low R² + Wide MC spread = Low confidence
    """
    r2 = linear_result.get("r_squared", 0)
    
    mc_median = monte_carlo_result.get("final_median", 0)
    mc_best = monte_carlo_result.get("final_best", 0)
    mc_worst = monte_carlo_result.get("final_worst", 0)
    
    # MC spread ratio: how tight is the Monte Carlo cone?
    # Smaller spread = more predictable = higher confidence
    if mc_median and mc_median > 0:
        spread = (mc_best - mc_worst) / mc_median
        mc_confidence = max(0, 1 - spread) # 1 = tight, 0 = very wide
    else:
        mc_confidence = 0
    
    # Blend: 40% R², 60% MC tightness
    score = (r2 * 0.4 + mc_confidence * 0.6) * 100
    score = max(0, min(100, round(score, 1)))
    
    if score >= 70:
        label = "🟢 Tinggi"
    elif score >= 40:
        label = "🟡 Sedang"
    else:
        label = "🔴 Rendah"
    
    return {"score": score, "label": label, "r2": round(r2, 4), "mc_spread": round(mc_confidence, 4)}

# ============================================================
# v5: FIXED TRADING TARGETS + ENTRY SCENARIOS
# ============================================================

def get_trading_targets(df: pd.DataFrame, signals: dict, adv_signals: dict) -> dict:
    """
    [v5 FIX] Stop loss logic:
    - ONLY "Strong Bearish" = short position (stop above, target below)
    - Everything else (Bullish, Neutral) = long position (stop below, target above)
    """
    if df.empty or "Close" not in df.columns or len(df) < 20:
        return _empty_targets()

    price = df["Close"].iloc[-1]
    high = df["High"].iloc[-1]
    low = df["Low"].iloc[-1]

    atr_val = adv_signals.get("atr", {}).get("value", None)
    if atr_val is None or atr_val == 0:
        atr_val = max(high - low, price * 0.02)

    # === DAY TRADE (1-5 hari) ===
    day_bias = _compute_bias(signals, adv_signals, "short")
    is_short = day_bias["is_short"]  # Only true for Strong Bearish

    if is_short:
        day_target = price - (atr_val * 1.5)
        day_stop = price + (atr_val * 1.0)
    else:
        day_target = price + (atr_val * 1.5)
        day_stop = price - (atr_val * 1.0)

    day_rr = abs(day_target - price) / abs(price - day_stop) if abs(price - day_stop) > 0 else 0
    day_sr = get_support_resistance(df, window=5)

    # Best entry for day trade
    day_best_entry = price - (atr_val * 0.3) if not is_short else price + (atr_val * 0.3)

    day_trade = {
        "timeframe": "1-5 Hari", "bias": day_bias["label"], "bias_color": day_bias["color"],
        "is_short": is_short,
        "entry": round(price, 0), "best_entry": round(day_best_entry, 0),
        "target": round(day_target, 0), "stop_loss": round(day_stop, 0),
        "risk_reward": round(day_rr, 2),
        "support": day_sr.get("support_1"), "resistance": day_sr.get("resistance_1"),
        "method": "ATR × 1.5 target, ATR × 1.0 stop",
    }

    # === SWING TRADE (5-20 hari) ===
    swing_bias = _compute_bias(signals, adv_signals, "mid")
    is_swing_short = swing_bias["is_short"]

    recent_20 = df.tail(20)
    swing_high = recent_20["High"].max()
    swing_low = recent_20["Low"].min()
    fib_range = swing_high - swing_low
    fib_236 = swing_high - fib_range * 0.236
    fib_382 = swing_high - fib_range * 0.382
    fib_500 = swing_high - fib_range * 0.500
    fib_618 = swing_high - fib_range * 0.618

    if is_swing_short:
        swing_target = fib_618 if price > fib_618 else swing_low
        swing_stop = min(swing_high, price + atr_val * 2)
        swing_best = fib_382 if price < fib_382 else price + (atr_val * 0.3)
    else:
        swing_target = fib_236 if price < fib_236 else swing_high
        swing_stop = max(fib_618, price - atr_val * 2)
        swing_best = fib_500 if price > fib_500 else price - (atr_val * 0.5)

    swing_rr = abs(swing_target - price) / abs(price - swing_stop) if abs(price - swing_stop) > 0 else 0

    swing_trade = {
        "timeframe": "5-20 Hari", "bias": swing_bias["label"], "bias_color": swing_bias["color"],
        "is_short": is_swing_short,
        "entry": round(price, 0), "best_entry": round(swing_best, 0),
        "target": round(swing_target, 0), "stop_loss": round(swing_stop, 0),
        "risk_reward": round(swing_rr, 2),
        "fib_236": round(fib_236, 0), "fib_382": round(fib_382, 0),
        "fib_500": round(fib_500, 0), "fib_618": round(fib_618, 0),
        "method": "Fibonacci + MA Crossover",
    }

    # === VALUE INVESTING (30-90 hari) ===
    val_bias = _compute_bias(signals, adv_signals, "long")
    # Value investing is ALWAYS long — never short
    linear = predict_price_linear(df, days=60)
    lin_target = linear.get("predicted_price_end", price) or price

    val_target = max(lin_target, price * 1.05) if val_bias["score"] >= 0.45 else price * 0.95
    val_stop = price * 0.90  # Always below for long position (max 10% loss)

    # Best entry = near support levels
    sr_long = get_support_resistance(df, window=60)
    s1_long = sr_long.get("support_1", price * 0.95) or price * 0.95
    val_best = max(s1_long, price * 0.95)  # Buy near support

    val_rr = abs(val_target - price) / abs(price - val_stop) if abs(price - val_stop) > 0 else 0

    value_invest = {
        "timeframe": "30-90 Hari", "bias": val_bias["label"], "bias_color": val_bias["color"],
        "is_short": False,  # Value investing is NEVER short
        "entry": round(price, 0), "best_entry": round(val_best, 0),
        "target": round(val_target, 0), "stop_loss": round(val_stop, 0),
        "risk_reward": round(val_rr, 2),
        "r_squared": linear.get("r_squared", 0),
        "confidence": linear.get("confidence", "N/A"),
        "method": "Linear Regression + Support Level",
    }

    return {
        "day_trade": day_trade,
        "swing_trade": swing_trade,
        "value_invest": value_invest,
    }


def get_entry_scenarios(df: pd.DataFrame, signals: dict, adv_signals: dict) -> dict:
    """
    [v5 NEW] Generate detailed entry scenarios for each strategy.
    Each scenario: condition, entry price, target, stop, rationale.
    """
    if df.empty or "Close" not in df.columns or len(df) < 20:
        return {"day_trade": [], "swing_trade": [], "value_invest": []}

    price = df["Close"].iloc[-1]
    atr_val = adv_signals.get("atr", {}).get("value", price * 0.02) or price * 0.02
    rsi = signals.get("rsi_value", 50) or 50

    sr_5 = get_support_resistance(df, window=5)
    sr_20 = get_support_resistance(df, window=20)
    sr_60 = get_support_resistance(df, window=60)

    recent_20 = df.tail(20)
    swing_high = recent_20["High"].max()
    swing_low = recent_20["Low"].min()
    fib_range = swing_high - swing_low
    fib_382 = swing_high - fib_range * 0.382
    fib_500 = swing_high - fib_range * 0.500
    fib_618 = swing_high - fib_range * 0.618

    # === DAY TRADE SCENARIOS ===
    day_scenarios = [
        {
            "name": "📗 Breakout Buy",
            "condition": f"Harga tembus di atas Rp {sr_5.get('resistance_1', price * 1.02) or price * 1.02:,.0f} dengan volume tinggi",
            "entry": round(sr_5.get("resistance_1", price * 1.02) or price * 1.02, 0),
            "target": round((sr_5.get("resistance_1", price * 1.02) or price * 1.02) + atr_val * 1.5, 0),
            "stop": round((sr_5.get("resistance_1", price * 1.02) or price * 1.02) - atr_val * 0.5, 0),
            "rationale": "Entry saat harga break resistance dengan konfirmasi volume",
            "probability": "Sedang" if rsi < 65 else "Rendah (RSI tinggi)",
        },
        {
            "name": "📗 Bounce Support",
            "condition": f"Harga menyentuh support Rp {sr_5.get('support_1', price * 0.98) or price * 0.98:,.0f} dan memantul (candle reversal)",
            "entry": round(sr_5.get("support_1", price * 0.98) or price * 0.98, 0),
            "target": round(price + atr_val, 0),
            "stop": round((sr_5.get("support_2", price * 0.96) or price * 0.96), 0),
            "rationale": "Buy on support bounce — risk kecil karena dekat support",
            "probability": "Tinggi" if rsi < 40 else "Sedang",
        },
        {
            "name": "📕 Short Breakdown",
            "condition": f"Harga tembus di bawah support Rp {sr_5.get('support_1', price * 0.98) or price * 0.98:,.0f}",
            "entry": round(sr_5.get("support_1", price * 0.98) or price * 0.98, 0),
            "target": round((sr_5.get("support_1", price * 0.98) or price * 0.98) - atr_val * 1.5, 0),
            "stop": round(price + atr_val * 0.5, 0),
            "rationale": "Short saat breakdown support dengan volume konfirmasi",
            "probability": "Sedang" if rsi > 50 else "Rendah",
        },
    ]

    # === SWING TRADE SCENARIOS ===
    swing_scenarios = [
        {
            "name": "📗 Fibonacci Bounce (61.8%)",
            "condition": f"Harga pullback ke area Fib 61.8% (Rp {fib_618:,.0f}) dan menunjukkan candle reversal",
            "entry": round(fib_618, 0),
            "target": round(swing_high, 0),
            "stop": round(swing_low - atr_val, 0),
            "rationale": "Entry di golden ratio — level bounce probabilitas tinggi",
            "probability": "Tinggi",
        },
        {
            "name": "📗 MA20 Retest",
            "condition": f"Harga pullback ke MA20 (±Rp {signals.get('ma_value', price) or price:,.0f}) saat uptrend",
            "entry": round(signals.get("ma_value", price) or price, 0),
            "target": round(swing_high * 1.02, 0),
            "stop": round((signals.get("ma_value", price) or price) - atr_val * 1.5, 0),
            "rationale": "MA20 sebagai dynamic support dalam uptrend",
            "probability": "Tinggi" if signals.get("ma_color") == "green" else "Rendah",
        },
        {
            "name": "📗 RSI Oversold Reversal",
            "condition": f"RSI turun di bawah 30 (sekarang: {rsi:.0f}) lalu naik kembali di atas 30",
            "entry": round(fib_500, 0),
            "target": round(fib_382, 0),
            "stop": round(fib_618 - atr_val, 0),
            "rationale": "Entry saat RSI keluar dari oversold — momentum reversal",
            "probability": "Tinggi" if rsi < 35 else ("Sedang" if rsi < 45 else "Belum terpenuhi"),
        },
    ]

    # === VALUE INVESTING SCENARIOS ===
    s1_60 = sr_60.get("support_1", price * 0.90) or price * 0.90
    s2_60 = sr_60.get("support_2", price * 0.85) or price * 0.85

    value_scenarios = [
        {
            "name": "📗 Accumulate Near Support",
            "condition": f"Harga mendekati support jangka panjang Rp {s1_60:,.0f}",
            "entry": round(s1_60, 0),
            "target": round(price * 1.20, 0),
            "stop": round(price * 0.85, 0),
            "rationale": "Beli bertahap (DCA) saat harga di area support kuat",
            "probability": "Menunggu" if price > s1_60 * 1.05 else "Siap entry",
        },
        {
            "name": "📗 Graham Number Entry",
            "condition": "PBV < 1.5 dan PE < 15 (sesuai kriteria Benjamin Graham)",
            "entry": round(price * 0.95, 0),
            "target": round(price * 1.30, 0),
            "stop": round(price * 0.80, 0),
            "rationale": "Entry jika valuasi memenuhi kriteria Graham — margin of safety",
            "probability": "Cek tab Fundamental",
        },
        {
            "name": "📗 DCA (Dollar Cost Average)",
            "condition": f"Beli Rp yang sama setiap minggu/bulan dari Rp {round(price * 0.90, 0):,.0f} sampai Rp {round(price * 1.05, 0):,.0f}",
            "entry": round(price, 0),
            "target": round(price * 1.25, 0),
            "stop": round(price * 0.75, 0),
            "rationale": "Rata-rata harga beli — mengurangi risiko timing",
            "probability": "Selalu siap",
        },
    ]

    return {
        "day_trade": day_scenarios,
        "swing_trade": swing_scenarios,
        "value_invest": value_scenarios,
    }


def _compute_bias(signals: dict, adv_signals: dict, horizon: str) -> dict:
    """
    [v5 FIX] Compute bias. Key fix: is_short is ONLY true for Strong Bearish.
    Neutral = long (stop below). This prevents stop > entry for neutral.
    """
    bullish_count = 0
    bearish_count = 0
    total = 0

    rsi = signals.get("rsi_value")
    if rsi is not None:
        total += 1
        if rsi < 30: bullish_count += 1
        elif rsi > 70: bearish_count += 1
        elif rsi < 50: bearish_count += 0.5
        else: bullish_count += 0.5

    ma_color = signals.get("ma_color", "gray")
    if ma_color != "gray":
        total += 1
        if ma_color == "green": bullish_count += 1
        else: bearish_count += 1

    macd_color = adv_signals.get("macd", {}).get("color", "gray")
    if macd_color != "gray":
        total += 1
        if macd_color == "green": bullish_count += 1
        else: bearish_count += 1

    composite = adv_signals.get("composite_score", 50)
    total += 1
    if composite > 55: bullish_count += 1
    elif composite < 45: bearish_count += 1

    if horizon in ("mid", "long"):
        gc_color = adv_signals.get("golden_cross", {}).get("color", "gray")
        if gc_color != "gray":
            total += 1.5
            if gc_color == "green": bullish_count += 1.5
            else: bearish_count += 1.5

    if horizon == "short":
        stoch_color = adv_signals.get("stochastic", {}).get("color", "gray")
        if stoch_color != "gray":
            total += 1
            if stoch_color == "green": bullish_count += 1
            else: bearish_count += 1

    score = bullish_count / total if total > 0 else 0.5

    # CRITICAL FIX: is_short = ONLY Strong Bearish (< 0.25)
    # Everything else (including Neutral) = long position
    if score >= 0.7:
        return {"label": "🟢 Strong Bullish", "color": "green", "is_short": False, "score": score}
    elif score >= 0.55:
        return {"label": "🟢 Bullish", "color": "lightgreen", "is_short": False, "score": score}
    elif score >= 0.45:
        return {"label": "🟡 Neutral", "color": "orange", "is_short": False, "score": score}
    elif score >= 0.25:
        return {"label": "🔴 Bearish", "color": "red", "is_short": False, "score": score}
    else:
        return {"label": "🔴 Strong Bearish", "color": "red", "is_short": True, "score": score}


def _empty_targets():
    empty = {"timeframe": "N/A", "bias": "N/A", "bias_color": "gray", "is_short": False,
             "entry": 0, "best_entry": 0, "target": 0, "stop_loss": 0, "risk_reward": 0, "method": "N/A"}
    return {"day_trade": empty, "swing_trade": empty, "value_invest": empty}


def get_confirmed_prediction(linear: dict, signals: dict, adv_signals: dict) -> dict:
    pred_slope = linear.get("slope_per_day", 0)
    pred_bullish = pred_slope > 0
    agreements = 0
    disagreements = 0
    checks = []

    macd = adv_signals.get("macd", {})
    if macd.get("color") not in ("gray", None):
        macd_bull = macd.get("color") == "green"
        agrees = macd_bull == pred_bullish
        checks.append(("MACD", agrees, "Bullish" if macd_bull else "Bearish"))
        if agrees: agreements += 1
        else: disagreements += 1

    rsi = signals.get("rsi_value")
    if rsi is not None:
        rsi_supports = (pred_bullish and rsi < 60) or (not pred_bullish and rsi > 40)
        checks.append(("RSI", rsi_supports, f"{rsi:.0f}"))
        if rsi_supports: agreements += 1
        else: disagreements += 1

    ma_c = signals.get("ma_color", "gray")
    if ma_c != "gray":
        ma_bull = ma_c == "green"
        agrees = ma_bull == pred_bullish
        checks.append(("MA Trend", agrees, "Up" if ma_bull else "Down"))
        if agrees: agreements += 1
        else: disagreements += 1

    composite = adv_signals.get("composite_score", 50)
    comp_supports = (pred_bullish and composite > 50) or (not pred_bullish and composite < 50)
    checks.append(("Composite", comp_supports, f"{composite:.0f}/100"))
    if comp_supports: agreements += 1
    else: disagreements += 1

    bb_pos = adv_signals.get("bollinger", {}).get("position")
    if bb_pos is not None:
        bb_supports = (pred_bullish and bb_pos < 60) or (not pred_bullish and bb_pos > 40)
        checks.append(("Bollinger", bb_supports, f"{bb_pos:.0f}%"))
        if bb_supports: agreements += 1
        else: disagreements += 1

    total_checks = agreements + disagreements
    ratio = agreements / total_checks if total_checks > 0 else 0
    direction = "Bullish 📈" if pred_bullish else "Bearish 📉"
    
    # v10: Professional Confluence / Divergence Check
    technical_score = adv_signals.get("composite_score", 50)
    ema_s = adv_signals.get("ema_cross", {}).get("signal", "")
    vpa_s = adv_signals.get("vpa", {}).get("signal", "")
    
    # CASE 1: DIVERGENCE (Price Trend Up, but Momentum is Bearish)
    # Trigger if (Technical Score < 50) AND (Fast Momentum is Bearish or VPA is Distribution)
    is_bearish_momentum = ("Bearish" in ema_s) or ("Distribusi" in vpa_s)
    is_bullish_momentum = ("Bullish" in ema_s) or ("Akumulasi" in vpa_s)

    if pred_bullish and (technical_score < 50 or is_bearish_momentum):
        final_direction = "⚠️ DIVERGENCE (Caution)"
        if "Bearish" in ema_s:
            confirmation_level = "Tren Naik vs Momentum Down (EMA 5 < 20)"
        elif "Distribusi" in vpa_s:
            confirmation_level = "Tren Naik vs Distribusi Volume"
        else:
            confirmation_level = "Momentum Melemah vs Tren Bullish"
        color = "orange"
        
    # CASE 2: REVERSAL POTENTIAL (Price Trend Down, but Momentum is Bullish)
    elif not pred_bullish and (technical_score > 55 or is_bullish_momentum):
        final_direction = "🔄 REVERSAL (Neutral)"
        confirmation_level = "Tren Turun vs Momentum Rebound"
        color = "cyan"
        
    # CASE 3: STANDARD CONFLUENCE
    else:
        if ratio >= 0.7:
            final_direction = direction
            confirmation_level = "Konfirmasi Kuat (Confluence DP)"
            color = "green"
        elif ratio >= 0.4:
            final_direction = direction
            confirmation_level = "Konfirmasi Moderat"
            color = "orange"
        else:
            final_direction = "⚪ SIDETRACK / NEUTRAL"
            confirmation_level = "Tidak Ada Konsensus (No Confluence)"
            color = "gray"

    return {
        "confirmation_level": confirmation_level, 
        "confirmation_color": color,
        "final_direction": final_direction,
        "agreements": agreements, 
        "total_checks": total_checks,
        "ratio": round(ratio * 100, 0),
        "prediction_direction": direction,
        "checks": checks
    }


def predict_price_ma_projection(df: pd.DataFrame, window: int = 20, days: int = 30) -> dict:
    """v5: Project future price based on Moving Average slope."""
    if df.empty or len(df) < window:
        return {"predicted_prices": pd.DataFrame()}
    
    ma = df["Close"].rolling(window).mean()
    if ma.isnull().all():
        return {"predicted_prices": pd.DataFrame()}
        
    last_ma = ma.iloc[-1]
    # Estimate slope from last 5 periods of MA
    ma_diff = ma.diff()
    avg_slope = ma_diff.tail(5).mean() if len(ma_diff) >= 5 else 0
    if pd.isna(avg_slope): avg_slope = 0
    
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
    if len(future_dates) < days:
        extra = pd.date_range(start=future_dates[-1] + timedelta(days=1), periods=days - len(future_dates))
        future_dates = future_dates.append(extra)
    future_dates = future_dates[:days]
    
    future_ma = [last_ma + (avg_slope * i) for i in range(1, days + 1)]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_ma})
    return {"predicted_prices": pred_df}


def get_prediction_summary(df: pd.DataFrame, days: int = 30) -> dict:
    linear = predict_price_linear(df, days)
    poly = predict_price_polynomial(df, days)
    ma_proj = predict_price_ma_projection(df, 20, days)
    return {
        "linear": linear,
        "poly": poly,
        "ma_projection": ma_proj,
        "support_resistance": get_support_resistance(df),
        "prediction_days": days
    }
