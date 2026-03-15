"""
config.py — Centralized Configuration
======================================
[DQC Perspective] Ini adalah "Single Source of Truth" untuk semua konfigurasi.
Sama seperti dalam audit data, kita tidak mau ada hardcoded values tersebar
di banyak tempat — semua dikumpulkan di satu titik untuk traceability.
"""

import os
from dotenv import load_dotenv

# Load .env file jika ada (untuk development lokal)
load_dotenv()

# ============================================================
# API KEYS
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ============================================================
# DEFAULT SETTINGS
# ============================================================
DEFAULT_TICKER = "BBCA.JK"           # Default saham: Bank Central Asia
DEFAULT_PERIOD = "1y"                 # Default periode data: 1 tahun

# ============================================================
# TECHNICAL INDICATOR SETTINGS
# ============================================================
# Basic
TECHNICAL_RSI_PERIOD = 14            # RSI lookback period (standar industri)
TECHNICAL_MA_PERIOD = 20             # Moving Average period (short-term trend)

# Advanced
MACD_FAST = 12                       # MACD fast EMA
MACD_SLOW = 26                       # MACD slow EMA
MACD_SIGNAL = 9                      # MACD signal line
BOLLINGER_PERIOD = 20                # Bollinger Bands period
BOLLINGER_STD = 2                    # Bollinger Bands std deviation
STOCHASTIC_K = 14                    # Stochastic %K period
STOCHASTIC_D = 3                     # Stochastic %D smoothing
ATR_PERIOD = 14                      # Average True Range period

# Prediction
PREDICTION_DAYS = 30                 # Default hari prediksi ke depan

# ============================================================
# GEMINI SETTINGS
# ============================================================
GEMINI_MODEL = "gemini-2.0-flash"    # Model gratis dengan kemampuan baca PDF
GEMINI_MAX_TOKENS = 2048             # Batasi output agar efisien token
GEMINI_TEMPERATURE = 0.3             # Rendah = lebih faktual, kurang kreatif

# ============================================================
# UI SETTINGS
# ============================================================
APP_TITLE = "📊 Stock Analyzer IHSG"
APP_ICON = "📊"
APP_LAYOUT = "wide"
