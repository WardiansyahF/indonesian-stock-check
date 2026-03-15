"""
modules/screener.py — Screener Module for "Spekulasi" Mode (BPJS & BSJP)
========================================================================
[DQC] This module fetches data in batch for a predefined list of liquid stocks
(e.g., KOMPAS100 or LQ45 subset) to ensure fast screening (approx 5-10s).
It avoids overfitting by using robust, simple momentum & volatility metrics.
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import time

# Daftar saham likuid (LQ45 + mid-cap/small-cap populer untuk spekulasi)
LIQUID_STOCKS = [
    # Top Tier / LQ45 Base
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "UNTR.JK", 
    "ICBP.JK", "INDF.JK", "AMRT.JK", "KLBF.JK", "CPIN.JK", "UNVR.JK", "PGAS.JK", 
    "AKRA.JK", "MEDC.JK", "BRPT.JK", "TPIA.JK", "ADRO.JK", "PTBA.JK", "ITMG.JK", 
    "ANTM.JK", "INCO.JK", "MDKA.JK", "HRUM.JK", "INKP.JK", "TKIM.JK", "SMGR.JK", 
    "INTP.JK", "GOTO.JK", "BUKA.JK", "ARTO.JK", "BRIS.JK", "BFIN.JK", "MAPI.JK", 
    "ACES.JK", "SIDO.JK", "MYOR.JK", "CTRA.JK", "BSDE.JK", "SMRA.JK", "PTPP.JK", 
    "WIKA.JK", "WSKT.JK", "MTEL.JK", "TOWR.JK", "TBIG.JK", "EXCL.JK", "ISAT.JK",
    "LSIP.JK", "SSMS.JK", "AALI.JK", "TAPG.JK",
    # Mid/Small-cap Momentum & Speculative (Non-LQ45)
    "BRMS.JK", "BUMI.JK", "DEWA.JK", "DOID.JK", "ELSA.JK", "ENRG.JK", "BIPI.JK",
    "CUAN.JK", "PANI.JK", "BREN.JK", "AMMN.JK", "WIDI.JK", "RAJA.JK", "HATM.JK",
    "BBKP.JK", "PNLF.JK", "PNBN.JK", "BNGA.JK", "NISP.JK", "BJBR.JK", "BJTM.JK",
    "AGRO.JK", "MAPA.JK", "AUTO.JK", "DRMA.JK", "SMSM.JK", "KAEF.JK", "IRRA.JK",
    "SILO.JK", "HEAL.JK", "MIKA.JK", "WIFI.JK", "MLPT.JK", "DNET.JK", "TCPI.JK",
    "FILM.JK", "SCMA.JK", "MNCN.JK", "BMTR.JK", "LPPF.JK", "RALS.JK", "ERAA.JK",
    "MARK.JK", "WOOD.JK", "CLEO.JK", "ULTJ.JK", "CMRY.JK", "CPRO.JK", "MBAP.JK",
    "SGER.JK", "TOBA.JK", "GEMS.JK", "DSNG.JK", "SIMP.JK", "BWPT.JK"
]

def get_screener_data(period: str = "1mo") -> tuple[pd.DataFrame, list, list]:
    """
    Mengambil data batch dan melakukan scan untuk strategi BPJS dan BSJP.
    Return: (raw_data, bpjs_list, bsjp_list)
    """
    # Batch download (menggunakan threads yfinance)
    df = yf.download(LIQUID_STOCKS, period=period, group_by='ticker', threads=True, progress=False)
    
    if df.empty:
        return df, [], []

    bpjs_candidates = []
    bsjp_candidates = []

    for ticker in LIQUID_STOCKS:
        try:
            if ticker not in df.columns.levels[0]:
                continue
                
            stock_data = df[ticker].dropna()
            if len(stock_data) < 20:
                continue

            # Kalkulasi Indikator Dasar untuk Spekulasi
            stock_data = stock_data.copy()
            # EMA lebih responsif dari SMA untuk scalping
            stock_data.ta.ema(length=5, append=True)
            stock_data.ta.ema(length=20, append=True)
            stock_data.ta.rsi(length=14, append=True)
            stock_data.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            # Volatility (ATR)
            stock_data.ta.atr(length=14, append=True)
            
            # VWAP approximation (Typical Price * Volume / Volume) - simple version
            typical_price = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
            stock_data['VWAP_approx'] = (typical_price * stock_data['Volume']).rolling(window=5).sum() / stock_data['Volume'].rolling(window=5).sum()

            # Avg Volume
            stock_data['Vol_MA20'] = stock_data['Volume'].rolling(20).mean()

            last_row = stock_data.iloc[-1]
            prev_row = stock_data.iloc[-2]

            price = float(last_row['Close'])
            high = float(last_row['High'])
            low = float(last_row['Low'])
            volume = float(last_row['Volume'])
            vol_ma20 = float(last_row['Vol_MA20'])
            ema5 = float(last_row['EMA_5'])
            ema20 = float(last_row['EMA_20'])
            rsi = float(last_row['RSI_14'])
            vwap = float(last_row['VWAP_approx'])
            macd_hist = float(last_row.get('MACDh_12_26_9', 0))
            atr = float(last_row.get('ATRr_14', 0))

            # Filter likuiditas dasar: Volume > 1 juta & Harga > 50
            if volume < 1_000_000 or price < 50:
                continue

            # ==========================================
            # Strategi BSJP (Beli Sore Jual Pagi)
            # Logika: Harga ditutup kuat di dekat high harian, volume spike, uptrend (EMA5 > EMA20).
            # Mengindikasikan akumulasi bandar sebelum penutupan.
            # ==========================================
            is_uptrend = ema5 > ema20 and price > ema20
            is_vol_spike = volume > (1.2 * vol_ma20)
            
            # Penutupan di 25% area teratas hari ini
            range_today = high - low
            close_near_high = (high - price) <= (0.25 * range_today) if range_today > 0 else False
            
            # Body candle hijau dan solid
            is_green_candle = price > float(last_row['Open'])
            
            if is_uptrend and is_vol_spike and close_near_high and is_green_candle:
                bsjp_score = (rsi / 100) * 0.4 + (volume / vol_ma20) * 0.4 + (1 - (high-price)/range_today) * 0.2
                bsjp_candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "target": round(price + (atr * 0.8), 0),   # Target scalping (0.8 ATR)
                    "stop": round(low, 0),                     # Stop di low hari ini
                    "volume_ratio": round(volume / vol_ma20, 2),
                    "rsi": round(rsi, 1),
                    "score": round(bsjp_score * 100, 1),
                    "reason": "Close kuat dekat High + Volume Spike"
                })

            # ==========================================
            # Strategi BPJS (Beli Pagi Jual Sore / Momentum)
            # Logika: RSI terakselerasi (momentum kuat), MACD Histogram positif menaik,
            # harga di atas VWAP (mendominasi rata-rata transaksi).
            # ==========================================
            rsi_momentum = rsi > 55 and rsi < 80 and rsi > float(prev_row['RSI_14'])
            macd_expanding = macd_hist > 0 and macd_hist > float(prev_row.get('MACDh_12_26_9', 0))
            above_vwap = price > vwap
            
            if rsi_momentum and macd_expanding and above_vwap and is_green_candle:
                bpjs_score = (rsi / 100) * 0.5 + (macd_hist / price * 1000) * 0.5
                bpjs_candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "target": round(price + atr, 0),         # Target 1 ATR
                    "stop": round(vwap * 0.99, 0),           # Stop slightly below VWAP
                    "rsi": round(rsi, 1),
                    "vwap_dist": round(((price - vwap) / vwap) * 100, 2), # Jarak % dari VWAP
                    "score": round(bpjs_score * 100, 1),
                    "reason": "RSI Momentum + MACD Expanding + Di atas VWAP"
                })

        except Exception as e:
            # Skip saham yang error datanya
            continue

    # Sort berdasarkan score tertinggi
    bpjs_sorted = sorted(bpjs_candidates, key=lambda x: x['score'], reverse=True)[:10]
    bsjp_sorted = sorted(bsjp_candidates, key=lambda x: x['score'], reverse=True)[:10]

    return df, bpjs_sorted, bsjp_sorted


# --- FUNDAMENTAL SCREENER (MULTITHREADED) ---

def _fetch_single_fundamental(ticker: str) -> dict:
    """Worker function to fetch fundamental data for a single stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # We need Price, PBV, PE, ROE, Market Cap, Revenue Growth, Earnings Growth
        price = info.get("currentPrice", info.get("regularMarketPrice", 0))
        pbv = info.get("priceToBook", 0)
        pe = info.get("trailingPE", info.get("forwardPE", 0))
        roe = info.get("returnOnEquity", 0)
        market_cap = info.get("marketCap", 0)
        t_debt = info.get("totalDebt", 0)
        t_equity = info.get("totalStockholderEquity", 0)
        der = (t_debt / t_equity) if t_equity else 0
        
        rev_growth = info.get("revenueGrowth", 0)
        earn_growth = info.get("earningsGrowth", 0)
        
        # Handle None values
        pbv = pbv if pbv is not None else 999
        pe = pe if pe is not None else 999
        roe = roe if roe is not None else 0
        market_cap = market_cap if market_cap is not None else 0
        rev_growth = rev_growth if rev_growth is not None else 0
        earn_growth = earn_growth if earn_growth is not None else 0
        
        return {
            "ticker": ticker,
            "company": info.get("shortName", ticker),
            "price": price,
            "pbv": pbv,
            "pe": pe,
            "roe": roe,
            "der": der,
            "market_cap": market_cap,
            "rev_growth": rev_growth,
            "earn_growth": earn_growth
        }
    except Exception:
        return None

def get_fundamental_screener_data() -> tuple[list, list, list]:
    """
    Mengambil data fundamental dari LIQUID_STOCKS menggunakan multi-threading.
    Return: (undervalue_list, growth_list, bluechip_list)
    """
    results = []
    
    # Use ThreadPool to fetch data concurrently (max 15 workers for speed vs rate limits)
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(_fetch_single_fundamental, ticker): ticker for ticker in LIQUID_STOCKS}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res and res["price"] > 0:
                results.append(res)
                
    undervalue = []
    growth = []
    bluechip = []
    
    for r in results:
        # 1. UNDERVALUE: PBV < 1.5, PE < 15, ROE > 10% (0.1)
        if 0 < r["pbv"] < 1.5 and 0 < r["pe"] < 15 and r["roe"] > 0.1:
            score = (1.5 - r["pbv"]) * 10 + (15 - r["pe"]) + (r["roe"] * 100) # Simple scoring
            undervalue.append({**r, "score": score, "reason": "Valuasi Murah & Profitabilitas Terjaga"})
            
        # 2. GROWTH: Rev Growth > 10%, Earnings Growth > 15%, ROE > 15%
        if r["rev_growth"] > 0.1 and r["earn_growth"] > 0.15 and r["roe"] > 0.15:
            score = (r["rev_growth"] * 100) + (r["earn_growth"] * 100)
            growth.append({**r, "score": score, "reason": "Pendapatan & Laba Melonjak"})
            
        # 3. BLUE CHIP: Market Cap > 100 Trillion IDR (100_000_000_000_000), ROE > 10%
        if r["market_cap"] > 100_000_000_000_000 and r["roe"] > 0.1:
            score = r["market_cap"] / 1_000_000_000_000 # Score by Trillions
            bluechip.append({**r, "score": score, "reason": "Raksasa Penggerak IHSG Secara Keseluruhan"})
            
    # Sort descending by score
    undervalue.sort(key=lambda x: x["score"], reverse=True)
    growth.sort(key=lambda x: x["score"], reverse=True)
    bluechip.sort(key=lambda x: x["score"], reverse=True)
    
    return undervalue[:10], growth[:10], bluechip[:10]
