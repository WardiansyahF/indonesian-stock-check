"""
modules/technical.py — Modul 3: Technical Analysis (Enhanced)
===============================================================
[DQC Perspective] Modul ini bertindak sebagai "Signal Detection Layer".
Menggunakan pandas_ta untuk menghitung indikator teknikal dasar DAN lanjutan,
lalu menginterpretasikan hasilnya menjadi sinyal yang actionable.

Indikator Dasar: RSI, MA20
Indikator Lanjutan: MACD, Bollinger Bands, Stochastic, MA50, MA200, ATR

Prinsip DQC yang diterapkan:
1. Data completeness: Cek apakah data cukup panjang untuk kalkulasi
2. Derived accuracy: Validasi output indikator
3. Clear labeling: Setiap sinyal memiliki label yang unambiguous
4. Scoring system: Composite score 0-100 dari semua indikator
"""

import pandas as pd
import pandas_ta as ta
from config import TECHNICAL_RSI_PERIOD, TECHNICAL_MA_PERIOD


# ============================================================
# BASIC INDICATORS (original)
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung indikator teknikal dasar: RSI dan Moving Average.
    """
    if df.empty or "Close" not in df.columns:
        return df

    min_required = max(TECHNICAL_RSI_PERIOD, TECHNICAL_MA_PERIOD) + 5
    if len(df) < min_required:
        return df

    df[f"RSI_{TECHNICAL_RSI_PERIOD}"] = ta.rsi(df["Close"], length=TECHNICAL_RSI_PERIOD)
    df[f"MA_{TECHNICAL_MA_PERIOD}"] = ta.ema(df["Close"], length=TECHNICAL_MA_PERIOD)  # Changed from SMA to EMA for reduced lag

    return df


# ============================================================
# ADVANCED INDICATORS (new)
# ============================================================

def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung indikator teknikal lanjutan:
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2)
    - Stochastic Oscillator (14, 3, 3)
    - MA50, MA200 (Golden/Death Cross)
    - ATR (14) — Average True Range (volatilitas)

    [DQC] Setiap indikator dihitung secara independen.
    Jika data tidak cukup untuk satu indikator, yang lain tetap dihitung.
    """
    if df.empty or "Close" not in df.columns:
        return df

    close = df["Close"]
    high = df.get("High", close)
    low = df.get("Low", close)

    # --- MACD (12, 26, 9) ---
    if len(df) >= 35:
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            for col in macd.columns:
                df[col] = macd[col]

    # --- Bollinger Bands (20, 2) ---
    if len(df) >= 25:
        bbands = ta.bbands(close, length=20, std=2)
        if bbands is not None and not bbands.empty:
            for col in bbands.columns:
                df[col] = bbands[col]

    # --- Stochastic (14, 3, 3) ---
    if len(df) >= 20:
        stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            for col in stoch.columns:
                df[col] = stoch[col]

    # --- EMA50 & EMA200 ---
    if len(df) >= 55:
        df["EMA_50"] = ta.ema(close, length=50)
    if len(df) >= 205:
        df["EMA_200"] = ta.ema(close, length=200)

    # --- ATR (14) & Historical Volatility (20) ---
    if len(df) >= 20:
        df["ATR_14"] = ta.atr(high, low, close, length=14)
        import numpy as np
        df["HV_20"] = np.log(close / close.shift(1)).rolling(20).std() * np.sqrt(252) * 100

    # --- ADX (14) ---
    if len(df) >= 28:
        adx = ta.adx(high, low, close, length=14)
        if adx is not None and not adx.empty:
            for col in adx.columns:
                df[col] = adx[col]

    # --- OBV ---
    if "Volume" in df.columns and len(df) >= 2:
        df["OBV"] = ta.obv(close, df["Volume"])

    # --- EMA 5 & 20 for Fast Momentum ---
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=20, append=True)

    # --- Candlestick Patterns ---
    if len(df) >= 3:
        # We manually calculate simple patterns for the last 2 days
        pass # The logic is handled in get_advanced_signals directly for simplicity

    return df


def get_technical_signals(df: pd.DataFrame) -> dict:
    """
    Interpretasi sinyal teknikal dasar (RSI + MA20) + advanced signals.
    """
    rsi_col = f"RSI_{TECHNICAL_RSI_PERIOD}"
    ma_col = f"MA_{TECHNICAL_MA_PERIOD}"

    result = {
        "rsi_value": None,
        "rsi_signal": "N/A",
        "rsi_color": "gray",
        "ma_value": None,
        "ma_signal": "N/A",
        "ma_color": "gray",
        "price_current": None,
        "overall_signal": "N/A",
        "overall_color": "gray",
    }

    if df.empty:
        return result

    latest = df.iloc[-1]
    result["price_current"] = round(latest.get("Close", 0), 2)

    # --- RSI Signal ---
    if rsi_col in df.columns and pd.notna(latest.get(rsi_col)):
        rsi = round(latest[rsi_col], 2)
        result["rsi_value"] = rsi

        if rsi < 30:
            result["rsi_signal"] = "🟢 Oversold (Potensi Beli)"
            result["rsi_color"] = "green"
        elif rsi > 70:
            result["rsi_signal"] = "🔴 Overbought (Potensi Jual)"
            result["rsi_color"] = "red"
        else:
            result["rsi_signal"] = "🟡 Neutral"
            result["rsi_color"] = "orange"

    # --- MA Signal ---
    if ma_col in df.columns and pd.notna(latest.get(ma_col)):
        ma = round(latest[ma_col], 2)
        result["ma_value"] = ma
        price = result["price_current"]

        if price and price > ma:
            result["ma_signal"] = "📈 Uptrend (Harga > MA20)"
            result["ma_color"] = "green"
        elif price and price < ma:
            result["ma_signal"] = "📉 Downtrend (Harga < MA20)"
            result["ma_color"] = "red"
        else:
            result["ma_signal"] = "➡️ Sideways"
            result["ma_color"] = "orange"

    # --- Overall Signal ---
    rsi_bullish = result["rsi_color"] == "green"
    ma_bullish = result["ma_color"] == "green"
    rsi_bearish = result["rsi_color"] == "red"
    ma_bearish = result["ma_color"] == "red"

    if rsi_bullish and ma_bullish:
        result["overall_signal"] = "✅ STRONG BUY SIGNAL"
        result["overall_color"] = "green"
    elif rsi_bullish or ma_bullish:
        result["overall_signal"] = "🟢 MODERATE BUY SIGNAL"
        result["overall_color"] = "lightgreen"
    elif rsi_bearish and ma_bearish:
        result["overall_signal"] = "🚫 STRONG SELL SIGNAL"
        result["overall_color"] = "red"
    elif rsi_bearish or ma_bearish:
        result["overall_signal"] = "🟡 CAUTION"
        result["overall_color"] = "orange"
    else:
        result["overall_signal"] = "⚪ NEUTRAL — Wait & See"
        result["overall_color"] = "gray"

    return result


def get_advanced_signals(df: pd.DataFrame) -> dict:
    """
    Interpretasi sinyal dari indikator lanjutan.

    [DQC] Setiap indikator menghasilkan sinyal independen.
    Overall score dihitung sebagai weighted average.
    """
    if df.empty:
        return {
            "macd": {"signal": "N/A", "color": "gray", "value": None, "histogram": None},
            "bollinger": {"signal": "N/A", "color": "gray", "position": None},
            "stochastic": {"signal": "N/A", "color": "gray", "k_value": None, "d_value": None},
            "golden_cross": {"signal": "N/A", "color": "gray", "ma50": None, "ma200": None},
            "atr": {"value": None, "volatility": "N/A", "color": "gray"},
            "adx": {"value": None, "trend_strength": "N/A", "color": "gray"},
            "obv": {"value": None, "signal": "N/A", "color": "gray"},
            "candlestick": {"pattern": "N/A", "signal": "N/A", "color": "gray"},
            "ema_cross": {"signal": "N/A", "color": "gray", "ema5": None, "ema20": None},
            "vpa": {"signal": "N/A", "color": "gray"},
            "macd_divergence": {"signal": "⚪ No Divergence", "color": "gray"},
            "composite_score": 50,
            "composite_label": "⚪ NEUTRAL",
            "composite_color": "gray",
        }

    latest = df.iloc[-1]
    price = latest.get("Close", 0)
    scores = []  # Kumpulkan score per indikator (0-100, 50=neutral)

    # --- MACD ---
    macd_result = {"signal": "N/A", "color": "gray", "value": None, "histogram": None}
    macd_cols = [c for c in df.columns if c.startswith("MACD")]
    if len(macd_cols) >= 2:
        macd_line_col = [c for c in macd_cols if "MACD_" in c and "s_" not in c and "h_" not in c]
        macd_signal_col = [c for c in macd_cols if "s_" in c.lower()]
        macd_hist_col = [c for c in macd_cols if "h_" in c.lower()]

        if macd_line_col and macd_signal_col:
            macd_val = latest.get(macd_line_col[0])
            signal_val = latest.get(macd_signal_col[0])
            hist_val = latest.get(macd_hist_col[0]) if macd_hist_col else None

            if pd.notna(macd_val) and pd.notna(signal_val):
                macd_result["value"] = round(macd_val, 2)
                macd_result["histogram"] = round(hist_val, 2) if pd.notna(hist_val) else None

                if macd_val > signal_val:
                    macd_result["signal"] = "📈 Bullish (MACD > Signal)"
                    macd_result["color"] = "green"
                    scores.append(70)
                else:
                    macd_result["signal"] = "📉 Bearish (MACD < Signal)"
                    macd_result["color"] = "red"
                    scores.append(30)

    # --- Bollinger Bands ---
    bb_result = {"signal": "N/A", "color": "gray", "position": None}
    bb_upper = [c for c in df.columns if "BBU_" in c]
    bb_lower = [c for c in df.columns if "BBL_" in c]
    bb_mid = [c for c in df.columns if "BBM_" in c]

    if bb_upper and bb_lower:
        upper = latest.get(bb_upper[0])
        lower = latest.get(bb_lower[0])
        mid = latest.get(bb_mid[0]) if bb_mid else None

        if pd.notna(upper) and pd.notna(lower) and upper != lower:
            position = (price - lower) / (upper - lower) * 100
            bb_result["position"] = round(position, 1)

            if price <= lower:
                bb_result["signal"] = "🟢 Di Bawah Lower Band (Oversold)"
                bb_result["color"] = "green"
                scores.append(80)
            elif price >= upper:
                bb_result["signal"] = "🔴 Di Atas Upper Band (Overbought)"
                bb_result["color"] = "red"
                scores.append(20)
            elif position < 30:
                bb_result["signal"] = "🟡 Dekat Lower Band"
                bb_result["color"] = "orange"
                scores.append(65)
            elif position > 70:
                bb_result["signal"] = "🟡 Dekat Upper Band"
                bb_result["color"] = "orange"
                scores.append(35)
            else:
                bb_result["signal"] = "⚪ Di Tengah Band"
                bb_result["color"] = "gray"
                scores.append(50)

    # --- Stochastic ---
    stoch_result = {"signal": "N/A", "color": "gray", "k_value": None, "d_value": None}
    stoch_k = [c for c in df.columns if "STOCHk_" in c]
    stoch_d = [c for c in df.columns if "STOCHd_" in c]

    if stoch_k and stoch_d:
        k_val = latest.get(stoch_k[0])
        d_val = latest.get(stoch_d[0])

        if pd.notna(k_val) and pd.notna(d_val):
            stoch_result["k_value"] = round(k_val, 2)
            stoch_result["d_value"] = round(d_val, 2)

            if k_val < 20:
                stoch_result["signal"] = "🟢 Oversold (%K < 20)"
                stoch_result["color"] = "green"
                scores.append(80)
            elif k_val > 80:
                stoch_result["signal"] = "🔴 Overbought (%K > 80)"
                stoch_result["color"] = "red"
                scores.append(20)
            elif k_val > d_val:
                stoch_result["signal"] = "📈 Bullish Crossover (%K > %D)"
                stoch_result["color"] = "green"
                scores.append(65)
            else:
                stoch_result["signal"] = "📉 Bearish (%K < %D)"
                stoch_result["color"] = "red"
                scores.append(35)

    # --- Golden / Death Cross (EMA50 vs EMA200) ---
    gc_result = {"signal": "N/A", "color": "gray", "ma50": None, "ma200": None}
    if "EMA_50" in df.columns and pd.notna(latest.get("EMA_50")):
        gc_result["ma50"] = round(latest["EMA_50"], 2)

        if "EMA_200" in df.columns and pd.notna(latest.get("EMA_200")):
            gc_result["ma200"] = round(latest["EMA_200"], 2)

            if gc_result["ma50"] > gc_result["ma200"]:
                gc_result["signal"] = "🌟 Golden Cross (EMA50 > EMA200)"
                gc_result["color"] = "green"
                scores.append(75)
            else:
                gc_result["signal"] = "💀 Death Cross (EMA50 < EMA200)"
                gc_result["color"] = "red"
                scores.append(25)
        else:
            gc_result["signal"] = "⚪ Data EMA200 belum cukup"
    else:
        gc_result["signal"] = "⚪ Data EMA50 belum cukup"

    # --- ATR ---
    atr_result = {"value": None, "volatility": "N/A", "color": "gray"}
    if "ATR_14" in df.columns and pd.notna(latest.get("ATR_14")):
        atr_val = round(latest["ATR_14"], 2)
        atr_result["value"] = atr_val

        atr_pct = (atr_val / price * 100) if price > 0 else 0
        if atr_pct > 5:
            atr_result["volatility"] = f"🔴 Tinggi ({atr_pct:.1f}%)"
            atr_result["color"] = "red"
        elif atr_pct > 2:
            atr_result["volatility"] = f"🟡 Sedang ({atr_pct:.1f}%)"
            atr_result["color"] = "orange"
        else:
            atr_result["volatility"] = f"🟢 Rendah ({atr_pct:.1f}%)"
            atr_result["color"] = "green"

    # --- ADX ---
    adx_result = {"value": None, "trend_strength": "N/A", "color": "gray"}
    adx_col = [c for c in df.columns if c.startswith("ADX_")]
    if adx_col and pd.notna(latest.get(adx_col[0])):
        adx_val = round(latest[adx_col[0]], 2)
        adx_result["value"] = adx_val
        if adx_val > 25:
            adx_result["trend_strength"] = "🔥 Trending Kuat (ADX > 25)"
            adx_result["color"] = "green"
            scores.append(70) # trend is your friend
        elif adx_val < 20:
            adx_result["trend_strength"] = "💤 Ranging / Sideways (ADX < 20)"
            adx_result["color"] = "orange"
            scores.append(40) # penalize sideways for classic trend follow
        else:
            adx_result["trend_strength"] = "🔄 Transisi Tren (20-25)"
            adx_result["color"] = "gray"
            scores.append(50)

    # --- OBV ---
    obv_result = {"value": None, "signal": "N/A", "color": "gray"}
    if "OBV" in df.columns and len(df) >= 20:
        obv_val = latest.get("OBV")
        if pd.notna(obv_val):
            obv_result["value"] = float(obv_val)
            # Simple OBV trend (vs 20 MA of OBV)
            obv_ma20 = df["OBV"].rolling(20).mean().iloc[-1]
            if pd.notna(obv_ma20):
                if obv_val > obv_ma20:
                    obv_result["signal"] = "🟢 Akumulasi (OBV > MA20)"
                    obv_result["color"] = "green"
                    scores.append(65)
                else:
                    obv_result["signal"] = "🔴 Distribusi (OBV < MA20)"
                    obv_result["color"] = "red"
                    scores.append(35)
                    
    # --- Candlestick Patterns ---
    candle_result = {"pattern": "N/A", "signal": "N/A", "color": "gray"}
    if len(df) >= 3:
        today = df.iloc[-1]
        yest = df.iloc[-2]
        
        t_open, t_close, t_high, t_low = today["Open"], today["Close"], today["High"], today["Low"]
        y_open, y_close = yest["Open"], yest["Close"]
        
        # Helper metrics
        body = abs(t_close - t_open)
        upper_shadow = t_high - max(t_open, t_close)
        lower_shadow = min(t_open, t_close) - t_low
        full_range = t_high - t_low
        y_body = abs(y_close - y_open)
        
        is_green = t_close > t_open
        is_red = t_close < t_open
        y_is_green = y_close > y_open
        y_is_red = y_close < y_open
        
        # 1. Doji
        if full_range > 0 and body / full_range < 0.1:
            candle_result["pattern"] = "➕ Doji"
            candle_result["signal"] = "Keraguan/Konsolidasi (Perlu Konfirmasi)"
            candle_result["color"] = "orange"
        # 2. Bullish Engulfing
        elif y_is_red and is_green and t_close > y_open and t_open <= y_close and body > y_body:
            candle_result["pattern"] = "🟩 Bullish Engulfing"
            candle_result["signal"] = "Potensi Reversal Naik Kuat"
            candle_result["color"] = "green"
            scores.append(75)
        # 3. Bearish Engulfing
        elif y_is_green and is_red and t_close < y_open and t_open >= y_close and body > y_body:
            candle_result["pattern"] = "🟥 Bearish Engulfing"
            candle_result["signal"] = "Potensi Reversal Turun Kuat"
            candle_result["color"] = "red"
            scores.append(25)
        # 4. Hammer (Bullish at bottom)
        elif lower_shadow > (2 * body) and upper_shadow < (0.2 * body):
            candle_result["pattern"] = "🔨 Hammer"
            candle_result["signal"] = "Rejeksi Support, Potensi Naik"
            candle_result["color"] = "green"
            scores.append(65)
        # 5. Shooting Star (Bearish at top)
        elif upper_shadow > (2 * body) and lower_shadow < (0.2 * body):
            candle_result["pattern"] = "🌠 Shooting Star"
            candle_result["signal"] = "Rejeksi Puncak, Potensi Turun"
            candle_result["color"] = "red"
            scores.append(35)
        # 6. Marubozu Bullish (tolerance: shadows < 5% of full range, body > 80%)
        elif is_green and full_range > 0 and body > (0.8 * full_range) and upper_shadow < (0.05 * full_range) and lower_shadow < (0.05 * full_range):
            candle_result["pattern"] = "🟩 Marubozu Bullish"
            candle_result["signal"] = "Dominasi Buyer Penuh"
            candle_result["color"] = "green"
            scores.append(80)
        # 7. Marubozu Bearish (tolerance: shadows < 5% of full range, body > 80%)
        elif is_red and full_range > 0 and body > (0.8 * full_range) and upper_shadow < (0.05 * full_range) and lower_shadow < (0.05 * full_range):
            candle_result["pattern"] = "🟥 Marubozu Bearish"
            candle_result["signal"] = "Dominasi Seller Penuh"
            candle_result["color"] = "red"
            scores.append(20)

    # --- v11: 3-Candle Master Patterns ---
    if len(df) >= 4:
        # t=today, y=yesterday, yy=2 days ago
        t, y, yy = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        
        # 8. Morning Star (Bullish Reversal 3-Candle)
        # 1st large red, 2nd small (star), 3rd green (closing > center of 1st)
        yy_body = abs(yy["Close"] - yy["Open"])
        yy_is_red = yy["Close"] < yy["Open"]
        y_body = abs(y["Close"] - y["Open"])
        t_is_green = t["Close"] > t["Open"]
        t_body = abs(t["Close"] - t["Open"])
        
        if yy_is_red and yy_body > (df["Close"].rolling(20).std().iloc[-1] * 1) and \
           y_body < (yy_body * 0.3) and t_is_green and t["Close"] > (yy["Open"] + yy["Close"])/2:
            candle_result["pattern"] = "⭐ Morning Star"
            candle_result["signal"] = "REVERSAL NAIK (Strong 3-Candle)"
            candle_result["color"] = "green"
            scores.append(90)
        # 9. Evening Star (Bearish Reversal 3-Candle)
        elif (not yy_is_red) and yy_body > (df["Close"].rolling(20).std().iloc[-1] * 1) and \
             y_body < (yy_body * 0.3) and (not t_is_green) and t["Close"] < (yy["Open"] + yy["Close"])/2:
            candle_result["pattern"] = "🌑 Evening Star"
            candle_result["signal"] = "REVERSAL TURUN (Strong 3-Candle)"
            candle_result["color"] = "red"
            scores.append(10)
        # 10. Bullish Harami (2-Candle Inside Bar)
        elif yy_is_red and (not y_is_red) and y["High"] < yy["Open"] and y["Low"] > yy["Close"]:
            candle_result["pattern"] = "🤰 Bullish Harami"
            candle_result["signal"] = "Potensi Reversi/Pause Bearish"
            candle_result["color"] = "green"
            scores.append(65)

    # --- v11: MACD Divergence Detection ---
    macd_div_result = {"signal": "⚪ No Divergence", "color": "gray"}
    if len(df) > 30:
        # Detect peaks in Price and MACD
        price_slice = df["Close"].tail(30).values
        macd_slice = df["MACD_12_26_9"].tail(30).values
        
        # Simple Peak Finding (Higher Highs / Lower Peaks)
        # Check last 2 local highs in price vs MACD
        if price_slice[-1] > max(price_slice[-20:-5]) and macd_slice[-1] < max(macd_slice[-20:-5]):
            macd_div_result["signal"] = "⚠️ BEARISH DIVERGENCE (Price HH, MACD LH)"
            macd_div_result["color"] = "red"
            scores.append(15)
        elif price_slice[-1] < min(price_slice[-20:-5]) and macd_slice[-1] > min(macd_slice[-20:-5]):
            macd_div_result["signal"] = "✅ BULLISH DIVERGENCE (Price LL, MACD HL)"
            macd_div_result["color"] = "green"
            scores.append(85)

    # --- EMA 5/20 Cross (Fast Momentum) ---
    ema_cross_result = {"signal": "⚪ Neutral", "color": "gray", "ema5": None, "ema20": None}
    if "EMA_5" in df.columns and "EMA_20" in df.columns:
        e5 = round(df["EMA_5"].iloc[-1], 2)
        e20 = round(df["EMA_20"].iloc[-1], 2)
        ema_cross_result["ema5"] = e5
        ema_cross_result["ema20"] = e20
        
        if e5 > e20:
            ema_cross_result["signal"] = "🟢 Bullish (EMA 5 > 20)"
            ema_cross_result["color"] = "green"
            scores.append(70)
        else:
            ema_cross_result["signal"] = "🔴 Bearish (EMA 5 < 20)"
            ema_cross_result["color"] = "red"
            scores.append(30)

    # --- Volume Price Analysis (VPA) Lite ---
    vpa_result = {"signal": "⚪ Neutral", "color": "gray"}
    if len(df) >= 2 and "Volume" in df.columns:
        last_price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        last_vol = df["Volume"].iloc[-1]
        
        is_high_vol = last_vol > (1.2 * avg_vol)
        
        if last_price_change > 0: # Price Up
            if is_high_vol:
                vpa_result["signal"] = "✅ Akumulasi (Price Up + High Vol)"
                vpa_result["color"] = "green"
                scores.append(85)
            else:
                vpa_result["signal"] = "🟡 Kenaikan Lemah (Low Vol)"
                vpa_result["color"] = "orange"
                scores.append(55)
        elif last_price_change < 0: # Price Down
            if is_high_vol:
                vpa_result["signal"] = "⚠️ Distribusi (Price Down + High Vol)"
                vpa_result["color"] = "red"
                scores.append(20)
            else:
                vpa_result["signal"] = "⚪ Pelemahan Wajar (Low Vol)"
                vpa_result["color"] = "gray"
                scores.append(45)

    # --- Composite Score ---
    if scores:
        composite = round(sum(scores) / len(scores), 1)
    else:
        composite = 50

    if composite >= 70:
        composite_label = "✅ STRONG BUY"
        composite_color = "green"
    elif composite >= 55:
        composite_label = "🟢 BUY"
        composite_color = "lightgreen"
    elif composite >= 45:
        composite_label = "⚪ NEUTRAL"
        composite_color = "gray"
    elif composite >= 30:
        composite_label = "🟡 SELL"
        composite_color = "orange"
    else:
        composite_label = "🔴 STRONG SELL"
        composite_color = "red"

    return {
        "macd": macd_result,
        "bollinger": bb_result,
        "stochastic": stoch_result,
        "golden_cross": gc_result,
        "atr": atr_result,
        "adx": adx_result,
        "obv": obv_result,
        "candlestick": candle_result,
        "ema_cross": ema_cross_result,
        "vpa": vpa_result,
        "macd_divergence": macd_div_result,
        "composite_score": composite,
        "composite_label": composite_label,
        "composite_color": composite_color,
    }
