"""
app.py — Personal Stock Analyzer IHSG v5.0
============================================
v5: Fixed stop loss, entry scenarios, multi-model Groq, 7 tabs
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    APP_TITLE, APP_ICON, APP_LAYOUT, DEFAULT_TICKER, DEFAULT_PERIOD,
    TECHNICAL_RSI_PERIOD, TECHNICAL_MA_PERIOD, PREDICTION_DAYS,
)
from modules.fundamental import (
    get_stock_info, get_financial_ratios,
    get_financials_summary, get_price_history, format_large_number,
)
from modules.technical import (
    calculate_indicators, calculate_advanced_indicators,
    get_technical_signals, get_advanced_signals,
)
from modules.prediction import (
    get_prediction_summary, get_trading_targets, get_confirmed_prediction,
    get_entry_scenarios, predict_monte_carlo, get_fibonacci_levels, get_unified_confidence
)
from modules.news import get_stock_news, get_external_news_links
from modules.ai_assistant import (
    analyze_pdf_report, quick_analysis, get_groq_model_list, GROQ_MODELS,
)
from modules.screener import get_screener_data, get_fundamental_screener_data

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT, initial_sidebar_state="expanded")

# === CSS ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.mc { background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98)); border: 1px solid rgba(100,116,139,0.25); border-radius: 16px; padding: 22px 26px; margin: 8px 0; min-height: 90px; transition: all 0.3s ease; }
.mc:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(0,0,0,0.3); border-color: rgba(99,102,241,0.4); }
.ml { color: #cbd5e1; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; line-height: 1.4; }
.mv { color: #f8fafc; font-size: 1.6rem; font-weight: 800; line-height: 1.2; }
.mv.grn { color: #4ade80; } .mv.red { color: #f87171; } .mv.org { color: #fbbf24; } .mv.blu { color: #60a5fa; }
.sb { display: inline-block; padding: 8px 20px; border-radius: 50px; font-weight: 700; font-size: 0.85rem; text-align: center; }
.sb.grn { background: linear-gradient(135deg,#065f46,#047857); color:#a7f3d0; border:1px solid #10b981; }
.sb.lgrn { background: linear-gradient(135deg,#14532d,#166534); color:#bbf7d0; border:1px solid #22c55e; }
.sb.red { background: linear-gradient(135deg,#7f1d1d,#991b1b); color:#fecaca; border:1px solid #ef4444; }
.sb.org { background: linear-gradient(135deg,#78350f,#92400e); color:#fde68a; border:1px solid #f59e0b; }
.sb.gry { background: linear-gradient(135deg,#1e293b,#334155); color:#e2e8f0; border:1px solid #64748b; }
.sh { font-size: 1.2rem; font-weight: 700; color: #f1f5f9; padding: 12px 0 8px 0; border-bottom: 3px solid rgba(99,102,241,0.5); margin-bottom: 18px; }
.ib { background: linear-gradient(135deg,rgba(30,58,138,0.25),rgba(30,64,175,0.15)); border:1px solid rgba(96,165,250,0.3); border-radius:14px; padding:18px 22px; margin:14px 0; color:#dbeafe; font-size:0.92rem; line-height:1.7; }
.rt { width:100%; border-collapse:separate; border-spacing:0; border-radius:14px; overflow:hidden; border:1px solid rgba(100,116,139,0.3); }
.rt th { background:rgba(30,41,59,0.95); color:#cbd5e1; padding:12px 18px; font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; text-align:left; }
.rt td { background:rgba(15,23,42,0.7); color:#f1f5f9; padding:12px 18px; font-size:0.95rem; border-top:1px solid rgba(100,116,139,0.15); line-height:1.4; }
.rt tr:nth-child(even) td { background:rgba(30,41,59,0.5); }
.rt tr:hover td { background:rgba(51,65,85,0.7); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%); }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:rgba(15,23,42,0.5); border-radius:12px; padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 16px; font-weight:600; font-size:0.9rem; }
.sc { text-align:center; padding:24px; background:linear-gradient(145deg,rgba(30,41,59,0.95),rgba(15,23,42,0.98)); border-radius:20px; border:1px solid rgba(100,116,139,0.25); }
.sn { font-size:3.2rem; font-weight:800; line-height:1; margin:8px 0; }
.nc { background:linear-gradient(145deg,rgba(30,41,59,0.9),rgba(15,23,42,0.95)); border:1px solid rgba(100,116,139,0.2); border-radius:14px; padding:18px 22px; margin:8px 0; transition:all 0.3s ease; }
.nc:hover { border-color:rgba(96,165,250,0.5); transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,0.3); }
.nt { color:#f1f5f9; font-size:1rem; font-weight:600; line-height:1.4; margin-bottom:6px; }
.nt a { color:#93c5fd; text-decoration:none; } .nt a:hover { color:#60a5fa; text-decoration:underline; }
.nm { color:#94a3b8; font-size:0.82rem; }
.pc { background:linear-gradient(145deg,rgba(30,41,59,0.95),rgba(15,23,42,0.98)); border:1px solid rgba(100,116,139,0.25); border-radius:16px; padding:20px 24px; margin:8px 0; text-align:center; }
.pl { color:#cbd5e1; font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }
.pv { font-size:1.4rem; font-weight:800; line-height:1.3; }
.tc { background:linear-gradient(145deg,rgba(30,41,59,0.95),rgba(15,23,42,0.98)); border:1px solid rgba(100,116,139,0.25); border-radius:16px; padding:18px; margin:8px 0; }
.th { color:#f1f5f9; font-size:1.05rem; font-weight:700; margin-bottom:10px; padding-bottom:6px; border-bottom:2px solid rgba(99,102,241,0.3); }
.tr { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(100,116,139,0.1); }
.tk { color:#94a3b8; font-size:0.85rem; } .tv { color:#f1f5f9; font-size:0.9rem; font-weight:600; }
.el { display:inline-block; padding:8px 18px; background:linear-gradient(135deg,rgba(30,41,59,0.9),rgba(51,65,85,0.9)); border:1px solid rgba(100,116,139,0.3); border-radius:10px; color:#93c5fd !important; text-decoration:none; font-weight:600; font-size:0.85rem; margin:4px; transition:all 0.2s ease; }
.el:hover { background:linear-gradient(135deg,rgba(51,65,85,0.9),rgba(71,85,105,0.9)); border-color:rgba(96,165,250,0.5); }
.es-card { background:linear-gradient(145deg,rgba(30,41,59,0.95),rgba(15,23,42,0.98)); border:1px solid rgba(100,116,139,0.25); border-radius:16px; padding:20px; margin:10px 0; }
.es-name { color:#f1f5f9; font-size:1rem; font-weight:700; margin-bottom:8px; }
.es-cond { color:#94a3b8; font-size:0.85rem; line-height:1.5; margin-bottom:10px; padding:10px; background:rgba(15,23,42,0.5); border-radius:8px; border-left:3px solid rgba(99,102,241,0.5); }
.es-row { display:flex; justify-content:space-between; padding:4px 0; }
.es-k { color:#94a3b8; font-size:0.85rem; } .es-v { color:#f1f5f9; font-size:0.85rem; font-weight:600; }
.es-reason { color:#cbd5e1; font-size:0.82rem; font-style:italic; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    app_mode = st.radio("🔍 Mode Aplikasi", options=["Analisa Lengkap (1 Saham)", "Screener Saham (Otomatis)"], index=0)
    st.markdown("---")
    ticker = st.text_input("🏢 Kode Saham", value=DEFAULT_TICKER, placeholder="contoh: BBCA.JK",
        help="Format: KODE.JK", disabled=(app_mode != "Analisa Lengkap (1 Saham)")).upper().strip()
    if ticker and not ticker.endswith(".JK") and app_mode == "Analisa Lengkap (1 Saham)":
        st.warning("💡 Ticker IHSG harus diakhiri `.JK`")
    period = st.selectbox("📅 Periode Data", options=["1d","1mo","3mo","6mo","1y","2y","5y"], index=3,
        format_func=lambda x: {"1d":"1 Hari (Intraday)","1mo":"1 Bulan","3mo":"3 Bulan","6mo":"6 Bulan","1y":"1 Tahun","2y":"2 Tahun","5y":"5 Tahun"}[x],
        disabled=(app_mode != "Analisa Lengkap (1 Saham)"))
    pred_days = st.slider("🔮 Hari Prediksi", min_value=7, max_value=90, value=PREDICTION_DAYS, step=7,
        disabled=(app_mode != "Analisa Lengkap (1 Saham)"))
    st.markdown("---")
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now().strftime("%H:%M:%S")
        st.rerun()
    st.caption(f"🕐 Last refresh: {st.session_state.get('last_refresh', 'Auto')}")
    st.markdown("---")
    st.markdown("### 🤖 AI Assistant")
    ai_provider = st.selectbox("🧠 Provider", ["Groq", "Gemini"], index=0,
        help="Groq: 30 RPM gratis. Gemini: support PDF.")
    if ai_provider == "Groq":
        groq_model_sel = st.selectbox("📦 Model", get_groq_model_list(), index=0)
        groq_model_id = GROQ_MODELS[groq_model_sel]
    else:
        groq_model_sel = None
        groq_model_id = None
    ai_key = st.text_input(
        f"{'Groq' if ai_provider == 'Groq' else 'Gemini'} API Key",
        type="password", placeholder="Masukkan API Key...",
        help="Groq: console.groq.com/keys | Gemini: aistudio.google.com/app/apikey")
    uploaded_pdf = st.file_uploader("📄 Upload LK (PDF)", type=["pdf"])
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#64748b;font-size:0.75rem;'>Stock Analyzer v5.0<br>DQC + IMK</div>", unsafe_allow_html=True)

# === HEADER ===
st.markdown(f"<h1 style='text-align:center;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;font-size:2.1rem;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;font-size:0.95rem;'>Value Investing Dashboard — IHSG • DQC Logic & IMK Design</p>", unsafe_allow_html=True)

# === DATA FETCHING (Analisa Lengkap) ===
@st.cache_data(ttl=600, show_spinner=False)
def fetch_all(t, p):
    info = get_stock_info(t)
    ratios = get_financial_ratios(t)
    financials = get_financials_summary(t)
    # 1-Day support: use 5m interval
    interval = "5m" if p == "1d" else "1d"
    price_df = get_price_history(t, p, interval=interval)
    news = get_stock_news(t, max_items=5)
    return info, ratios, financials, price_df, news

# ============================================================
# MODE SCREENER (MULTIPLE STOCKS)
# ============================================================
if app_mode == "Screener Saham (Otomatis)":
    st.markdown("<div class='sh'>🚀 Screener Saham (Liquid Stocks)</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib'>Mode ini memindai 50+ saham paling likuid di IHSG secara otomatis berdasarkan <strong>Momentum Teknikal</strong> dan <strong>Filter Fundamental</strong> khusus.</div>", unsafe_allow_html=True)
    
    with st.spinner("⏳ Memindai pasar (butuh 10-15 detik)..."):
        try:
            raw_df, bpjs_list, bsjp_list = get_screener_data(period="1mo")
            undervalue_list, growth_list, bluechip_list = get_fundamental_screener_data()
            
            tb1, tb2, tb3, tb4, tb5 = st.tabs([
                "🌅 BPJS (Pagi-Sore)", 
                "🌃 BSJP (Sore-Pagi)",
                "💎 Undervalue",
                "🚀 Growth",
                "🏛️ Blue Chip"
            ])
            
            with tb1:
                st.markdown("### Top 10 BPJS Candidates")
                st.markdown("<p style='color:#94a3b8;font-size:0.9rem;'>Cari saham dengan RSI menguat, MACD Histogram menaik, dan harga di atas VWAP (Momentum kencang).</p>", unsafe_allow_html=True)
                if not bpjs_list:
                    st.info("Tidak ada kandidat BPJS yang memenuhi kriteria saat ini.")
                else:
                    for s in bpjs_list:
                        st.markdown(f"""<div class='es-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(100,116,139,0.2); padding-bottom:8px; margin-bottom:10px;'>
                                <span style='font-size:1.2rem;font-weight:800;color:#60a5fa;'>{s['ticker']}</span>
                                <span class='sb lgrn'>Score: {s['score']}</span>
                            </div>
                            <div class='es-row'><span class='es-k'>Harga Terakhir</span><span class='es-v'>Rp {s['price']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>🎯 Target Cepat (+1 ATR)</span><span class='es-v' style='color:#4ade80;'>Rp {s['target']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>🛑 Stop Loss (VWAP)</span><span class='es-v' style='color:#f87171;'>Rp {s['stop']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>RSI Momentum</span><span class='es-v'>{s['rsi']}</span></div>
                            <div class='es-reason'>💡 Alasan: {s['reason']}</div>
                        </div>""", unsafe_allow_html=True)

            with tb2:
                st.markdown("### Top 10 BSJP Candidates")
                st.markdown("<p style='color:#94a3b8;font-size:0.9rem;'>Cari saham Uptrend dengan Volume Spike yang ditutup di harga tinggi (Tanda akumulasi bandar).</p>", unsafe_allow_html=True)
                if not bsjp_list:
                    st.info("Tidak ada kandidat BSJP yang memenuhi kriteria saat ini.")
                else:
                    for s in bsjp_list:
                        st.markdown(f"""<div class='es-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(100,116,139,0.2); padding-bottom:8px; margin-bottom:10px;'>
                                <span style='font-size:1.2rem;font-weight:800;color:#a78bfa;'>{s['ticker']}</span>
                                <span class='sb lgrn'>Score: {s['score']}</span>
                            </div>
                            <div class='es-row'><span class='es-k'>Harga Terakhir</span><span class='es-v'>Rp {s['price']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>Target Pendek</span><span class='es-v' style='color:#4ade80;'>Rp {s['target']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>Stop Loss</span><span class='es-v' style='color:#f87171;'>Rp {s['stop']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>Volume vs Avg</span><span class='es-v'>{s['volume_ratio']}x lebih tinggi</span></div>
                            <div class='es-reason'>💡 Alasan: {s['reason']}</div>
                        </div>""", unsafe_allow_html=True)

            with tb3:
                st.markdown("### 💎 Top 10 Undervalue Candidates")
                st.markdown("<p style='color:#94a3b8;font-size:0.9rem;'>Saham salah harga (Mispriced): PBV < 1.5x, PE < 15x, tapi profitabilitas masih sehat (ROE > 10%).</p>", unsafe_allow_html=True)
                if not undervalue_list:
                    st.info("Tidak ada kandidat Undervalue yang ketat saat ini.")
                else:
                    for s in undervalue_list:
                        st.markdown(f"""<div class='es-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(100,116,139,0.2); padding-bottom:8px; margin-bottom:10px;'>
                                <span style='font-size:1.2rem;font-weight:800;color:#38bdf8;'>{s['ticker']}</span>
                                <span class='sb lgrn'>Value Score: {s['score']:.1f}</span>
                            </div>
                            <div class='es-row'><span class='es-k'>Harga Terakhir</span><span class='es-v'>Rp {s['price']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>Price/Book (PBV)</span><span class='es-v' style='color:#4ade80;'>{s['pbv']:.2f}x</span></div>
                            <div class='es-row'><span class='es-k'>PE Ratio</span><span class='es-v' style='color:#4ade80;'>{s['pe']:.1f}x</span></div>
                            <div class='es-row'><span class='es-k'>ROE (Efisiensi)</span><span class='es-v' style='color:#a78bfa;'>{s['roe']*100:.1f}%</span></div>
                            <div class='es-reason'>💡 Alasan: {s['reason']}</div>
                        </div>""", unsafe_allow_html=True)

            with tb4:
                st.markdown("### 🚀 Top 10 Growth Candidates")
                st.markdown("<p style='color:#94a3b8;font-size:0.9rem;'>Saham bertumbuh pesat: Revenue > 10% YoY, Laba Bersih > 15% YoY, dan ROE tinggi.</p>", unsafe_allow_html=True)
                if not growth_list:
                    st.info("Tidak ada kandidat Growth super ketat saat ini.")
                else:
                    for s in growth_list:
                        gr_score = f"{s['score']:.1f}" if s['score'] > 0 else "N/A"
                        st.markdown(f"""<div class='es-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(100,116,139,0.2); padding-bottom:8px; margin-bottom:10px;'>
                                <span style='font-size:1.2rem;font-weight:800;color:#f472b6;'>{s['ticker']}</span>
                                <span class='sb lgrn'>Growth Index: {gr_score}</span>
                            </div>
                            <div class='es-row'><span class='es-k'>Harga Terakhir</span><span class='es-v'>Rp {s['price']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>Pertumbuhan Pendapatan</span><span class='es-v' style='color:#4ade80;'>+{s['rev_growth']*100:.1f}%</span></div>
                            <div class='es-row'><span class='es-k'>Pertumbuhan Laba (EPS)</span><span class='es-v' style='color:#4ade80;'>+{s['earn_growth']*100:.1f}%</span></div>
                            <div class='es-row'><span class='es-k'>ROE (Efisiensi)</span><span class='es-v'>{s['roe']*100:.1f}%</span></div>
                            <div class='es-reason'>💡 Alasan: {s['reason']}</div>
                        </div>""", unsafe_allow_html=True)

            with tb5:
                st.markdown("### 🏛️ Top Blue Chip (Market Leaders)")
                st.markdown("<p style='color:#94a3b8;font-size:0.9rem;'>Koleksi raksasa IHSG: Market Cap > 100T Rupiah dengan mesin profitabilitas (ROE) kuat. Cocok untuk nabung santai.</p>", unsafe_allow_html=True)
                if not bluechip_list:
                    st.info("Market sedang fluktuatif.")
                else:
                    for s in bluechip_list:
                        mcap_tril = s['market_cap'] / 1_000_000_000_000
                        st.markdown(f"""<div class='es-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(100,116,139,0.2); padding-bottom:8px; margin-bottom:10px;'>
                                <span style='font-size:1.2rem;font-weight:800;color:#fbbf24;'>{s['ticker']}</span>
                                <span class='sb org'>Index: {s['score']:.0f}</span>
                            </div>
                            <div class='es-row'><span class='es-k'>Harga Terakhir</span><span class='es-v'>Rp {s['price']:,.0f}</span></div>
                            <div class='es-row'><span class='es-k'>Kapitalisasi Pasar (Jumbo)</span><span class='es-v' style='color:#4ade80;'>Rp {mcap_tril:,.0f} Triliun</span></div>
                            <div class='es-row'><span class='es-k'>Valuasi PE Ratio</span><span class='es-v'>{s['pe']:.1f}x</span></div>
                            <div class='es-row'><span class='es-k'>ROE (Efisiensi)</span><span class='es-v' style='color:#a78bfa;'>{s['roe']*100:.1f}%</span></div>
                            <div class='es-reason'>💡 Alasan: {s['reason']}</div>
                        </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Gagal memindai pasar: {str(e)}")
    
    st.stop() # Hentikan eksekusi UI Analisa Lengkap


# ============================================================
# MODE ANALISA LENGKAP (1 SAHAM)
# ============================================================

if not ticker:
    st.info("👈 Masukkan kode saham di sidebar.")
    st.stop()

with st.spinner(f"⏳ Mengambil data **{ticker}**..."):
    try:
        info, ratios, financials, price_df, news_items = fetch_all(ticker, period)
    except Exception as e:
        st.error(f"❌ Gagal mengambil data: {str(e)}")
        st.stop()

if price_df.empty:
    st.error(f"❌ Tidak ada data untuk `{ticker}`.")
    st.stop()

tech_df = calculate_indicators(price_df.copy())
tech_df = calculate_advanced_indicators(tech_df)
signals = get_technical_signals(tech_df)
adv_signals = get_advanced_signals(tech_df)
prediction = get_prediction_summary(price_df, pred_days)
monte_carlo = predict_monte_carlo(price_df, pred_days, simulations=1000)
trading_targets = get_trading_targets(price_df, signals, adv_signals)
confirmed = get_confirmed_prediction(prediction["linear"], signals, adv_signals)
entry_scenarios = get_entry_scenarios(price_df, signals, adv_signals)
fibonacci = get_fibonacci_levels(price_df)
unified_conf = get_unified_confidence(prediction["linear"], monte_carlo)

# Build technical context string for AI
tech_context_lines = [
    f"Harga Terakhir: Rp {info.get('Harga Terakhir', 0):,.0f}",
    f"Composite Score: {adv_signals.get('composite_score', 50)}/100 ({adv_signals.get('composite_label', 'N/A')})",
    f"RSI (14): {signals.get('rsi_value', 'N/A')} — {signals.get('rsi_signal', 'N/A')}",
    f"MACD: {adv_signals['macd']['value']} — {adv_signals['macd']['signal']}",
    f"MACD Divergence: {adv_signals.get('macd_divergence',{}).get('signal','N/A')}",
    f"ADX (Kekuatan Tren): {adv_signals.get('adx',{}).get('value','N/A')} — {adv_signals.get('adx',{}).get('trend_strength','N/A')}",
    f"OBV (Akumulasi): {adv_signals.get('obv',{}).get('signal','N/A')}",
    f"EMA 5/20 Momentum: {adv_signals.get('ema_cross',{}).get('signal','N/A')}",
    f"VPA (Volume Price): {adv_signals.get('vpa',{}).get('signal','N/A')}",
    f"Bollinger Position: {adv_signals['bollinger']['position']}% — {adv_signals['bollinger']['signal']}",
    f"Candlestick Pattern: {adv_signals.get('candlestick',{}).get('pattern','N/A')} — {adv_signals.get('candlestick',{}).get('signal','N/A')}",
    f"MA50/MA200: {adv_signals['golden_cross']['signal']}",
    f"ATR Volatilitas: {adv_signals['atr']['volatility']}",
    f"",
    f"--- PREDIKSI & CURVE ---",
    f"Trend Momentum (Curve): {prediction['poly'].get('curve_type','N/A')}",
    f"Linear Regression: Arah {prediction['linear'].get('direction','N/A')}, R²={prediction['linear'].get('r_squared',0)}, Prediksi {pred_days}H: Rp {prediction['linear'].get('predicted_price_end',0):,.0f} ({prediction['linear'].get('change_pct',0):+.1f}%)",
    f"Monte Carlo ({pred_days}H): Worst Rp {monte_carlo.get('final_worst',0):,.0f}, Median Rp {monte_carlo.get('final_median',0):,.0f}, Best Rp {monte_carlo.get('final_best',0):,.0f}",
    f"Unified Prediction Confidence: {unified_conf.get('score',0)}% ({unified_conf.get('label','N/A')})",
    f"Confirmed Direction: {confirmed.get('final_direction','N/A')} — {confirmed.get('confirmation_level','N/A')} ({confirmed.get('agreements',0)}/{confirmed.get('total_checks',0)} indikator setuju)",
    f"",
    f"--- SUPPORT / RESISTANCE ---",
    f"Pivot: Rp {prediction['support_resistance'].get('pivot_point',0) or 0:,.0f}",
    f"R1: Rp {prediction['support_resistance'].get('resistance_1',0) or 0:,.0f}, R2: Rp {prediction['support_resistance'].get('resistance_2',0) or 0:,.0f}",
    f"S1: Rp {prediction['support_resistance'].get('support_1',0) or 0:,.0f}, S2: Rp {prediction['support_resistance'].get('support_2',0) or 0:,.0f}",
    f"Fibonacci: 23.6%=Rp {fibonacci.get('fib_236',0) or 0:,.0f}, 38.2%=Rp {fibonacci.get('fib_382',0) or 0:,.0f}, 50%=Rp {fibonacci.get('fib_500',0) or 0:,.0f}, 61.8%=Rp {fibonacci.get('fib_618',0) or 0:,.0f}",
    f"",
    f"--- TARGET TRADING ---",
    f"Day Trade: Bias {trading_targets['day_trade'].get('bias','N/A')}, Entry Rp {trading_targets['day_trade'].get('entry',0):,.0f}, Target Rp {trading_targets['day_trade'].get('target',0):,.0f}, SL Rp {trading_targets['day_trade'].get('stop_loss',0):,.0f}, R:R 1:{trading_targets['day_trade'].get('risk_reward',0)}",
    f"Swing: Bias {trading_targets['swing_trade'].get('bias','N/A')}, Entry Rp {trading_targets['swing_trade'].get('entry',0):,.0f}, Target Rp {trading_targets['swing_trade'].get('target',0):,.0f}, SL Rp {trading_targets['swing_trade'].get('stop_loss',0):,.0f}, R:R 1:{trading_targets['swing_trade'].get('risk_reward',0)}",
    f"Value: Bias {trading_targets['value_invest'].get('bias','N/A')}, Entry Rp {trading_targets['value_invest'].get('entry',0):,.0f}, Target Rp {trading_targets['value_invest'].get('target',0):,.0f}, SL Rp {trading_targets['value_invest'].get('stop_loss',0):,.0f}, R:R 1:{trading_targets['value_invest'].get('risk_reward',0)}",
]
technical_context_str = "\n".join(tech_context_lines)

# === 7 TABS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "📋 Fundamental", "📈 Technical", "🎯 Entry Strategy", "🔮 Prediction", "🤖 AI", "📰 News"
])

# =================== TAB 1: OVERVIEW ===================
with tab1:
    st.markdown(f"<div class='sh'>🏢 {info.get('Nama Perusahaan', ticker)}</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f"<div class='mc'><div class='ml'>Harga Terakhir</div><div class='mv blu'>Rp {info.get('Harga Terakhir',0):,.0f}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='mc'><div class='ml'>Market Cap</div><div class='mv'>{format_large_number(info.get('Market Cap',0))}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='mc'><div class='ml'>52W High</div><div class='mv grn'>Rp {info.get('52W High',0):,.0f}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='mc'><div class='ml'>52W Low</div><div class='mv red'>Rp {info.get('52W Low',0):,.0f}</div></div>", unsafe_allow_html=True)
    d1,d2,d3 = st.columns(3)
    with d1: st.markdown(f"<div class='mc'><div class='ml'>Sektor</div><div class='mv' style='font-size:1.1rem;'>{info.get('Sektor','N/A')}</div></div>", unsafe_allow_html=True)
    with d2: st.markdown(f"<div class='mc'><div class='ml'>Industri</div><div class='mv' style='font-size:1.1rem;'>{info.get('Industri','N/A')}</div></div>", unsafe_allow_html=True)
    with d3:
        sc = adv_signals.get("composite_color","gray"); sv = adv_signals.get("composite_score",50); sl = adv_signals.get("composite_label","N/A")
        st.markdown(f"<div class='mc'><div class='ml'>Composite Score</div><div class='mv {sc}' style='font-size:1.8rem;'>{sv}/100</div><div class='sb {sc}' style='margin-top:6px;'>{sl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>📈 Grafik Harga</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=price_df.index, open=price_df["Open"], high=price_df["High"], low=price_df["Low"], close=price_df["Close"], name="OHLC", increasing_line_color="#4ade80", decreasing_line_color="#f87171"))
    mac = f"MA_{TECHNICAL_MA_PERIOD}"
    if mac in tech_df.columns: fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df[mac], name=f"MA{TECHNICAL_MA_PERIOD}", line=dict(color="#fbbf24", width=2)))
    if "MA_50" in tech_df.columns: fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MA_50"], name="MA50", line=dict(color="#38bdf8", width=1.5, dash="dot")))
    if "MA_200" in tech_df.columns: fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MA_200"], name="MA200", line=dict(color="#f472b6", width=1.5, dash="dash")))
    bbu = [c for c in tech_df.columns if "BBU_" in c]; bbl = [c for c in tech_df.columns if "BBL_" in c]
    if bbu and bbl:
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df[bbu[0]], name="BB Upper", line=dict(color="rgba(167,139,250,0.4)", width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df[bbl[0]], name="BB Lower", line=dict(color="rgba(167,139,250,0.4)", width=1), fill="tonexty", fillcolor="rgba(167,139,250,0.05)", showlegend=False))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)", height=500, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1), font=dict(family="Inter",color="#94a3b8"), yaxis=dict(title="Harga (Rp)",gridcolor="rgba(100,116,139,0.15)"), xaxis=dict(gridcolor="rgba(100,116,139,0.15)"))
    st.plotly_chart(fig, use_container_width=True)
    colors = ["#4ade80" if c>=o else "#f87171" for c,o in zip(price_df["Close"],price_df["Open"])]
    fv = go.Figure(go.Bar(x=price_df.index, y=price_df["Volume"], marker_color=colors, opacity=0.7))
    fv.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)", height=170, margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(title="Volume",gridcolor="rgba(100,116,139,0.15)"), xaxis=dict(gridcolor="rgba(100,116,139,0.15)"), font=dict(family="Inter",color="#94a3b8"), showlegend=False)
    st.plotly_chart(fv, use_container_width=True)

# =================== TAB 2: FUNDAMENTAL ===================
with tab2:
    st.markdown("<div class='sh'>📋 Rasio Keuangan Utama</div>", unsafe_allow_html=True)
    st.markdown("""<div class='ib'><strong>🎓 Panduan Value Investing (Benjamin Graham):</strong><br>• PBV < 1.5 → <strong>Murah</strong> • PE < 15 → Harga wajar • D/E < 50 → Hutang sehat • ROE > 15% → Efisien</div>""", unsafe_allow_html=True)
    gc = {
        "PBV (Price/Book)": lambda v: ("🟢","Murah") if isinstance(v,(int,float)) and v<1.5 else (("🔴","Mahal") if isinstance(v,(int,float)) and v>3 else ("🟡","Wajar")),
        "PE Ratio (TTM)": lambda v: ("🟢","Murah") if isinstance(v,(int,float)) and v<15 else (("🔴","Mahal") if isinstance(v,(int,float)) and v>25 else ("🟡","Wajar")),
        "Debt to Equity": lambda v: ("🟢","Sehat") if isinstance(v,(int,float)) and v<50 else (("🔴","Tinggi") if isinstance(v,(int,float)) and v>100 else ("🟡","Moderat")),
        "ROE (%)": lambda v: ("🟢","Bagus") if isinstance(v,(int,float)) and v>15 else (("🔴","Buruk") if isinstance(v,(int,float)) and v<5 else ("🟡","Cukup")),
        "ROA (%)": lambda v: ("🟢","Bagus") if isinstance(v,(int,float)) and v>5 else (("🟡","Cukup") if isinstance(v,(int,float)) else ("⚪","N/A")),
        "Dividend Yield (%)": lambda v: ("🟢","Bagus") if isinstance(v,(int,float)) and v>3 else (("🟡","Rendah") if isinstance(v,(int,float)) else ("⚪","N/A")),
    }
    rows = ""
    for key, val in ratios.items():
        si, st_txt = "⚪", ""
        if key in gc and val != "N/A" and not isinstance(val, str): si, st_txt = gc[key](val)
        dv = f"{val}%" if "(%)" in key and isinstance(val,(int,float)) else val
        rows += f"<tr><td>{key}</td><td><strong>{dv}</strong></td><td>{si} {st_txt}</td></tr>"
    st.markdown(f"<table class='rt'><thead><tr><th>Metrik</th><th>Nilai</th><th>Penilaian</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>📊 Laporan Keuangan (Annual)</div>", unsafe_allow_html=True)
    if financials is not None and not financials.empty:
        ddf = financials.copy()
        for cn in ["Revenue","Net Income","Total Debt"]:
            if cn in ddf.columns: ddf[cn] = ddf[cn].apply(format_large_number)
        st.dataframe(ddf, use_container_width=True, hide_index=True)

# =================== TAB 3: TECHNICAL ===================
with tab3:
    st.markdown("<div class='sh'>📈 Analisis Teknikal</div>", unsafe_allow_html=True)
    scc = adv_signals["composite_color"]; sv = adv_signals["composite_score"]; sl = adv_signals["composite_label"]
    stc = {"green":"#4ade80","lightgreen":"#86efac","red":"#f87171","orange":"#fbbf24","gray":"#e2e8f0"}.get(scc,"#e2e8f0")
    sc1, sc2 = st.columns([1,2])
    with sc1:
        st.markdown(f"<div class='sc'><div class='ml'>COMPOSITE SCORE</div><div class='sn' style='color:{stc};'>{sv}</div><div style='color:#94a3b8;font-size:0.82rem;'>dari 100</div><div class='sb {scc}' style='margin-top:10px;'>{sl}</div></div>", unsafe_allow_html=True)
    with sc2:
        sd = [
            ("RSI (14)", signals.get("rsi_value","N/A"), signals.get("rsi_signal","N/A")),
            ("MA20 Trend", signals.get("ma_value","N/A"), signals.get("ma_signal","N/A")),
            ("EMA 5/20 Momentum", f"EMA5:{adv_signals['ema_cross']['ema5']} vs EMA20:{adv_signals['ema_cross']['ema20']}", adv_signals["ema_cross"]["signal"]),
            ("VPA (Vol-Price Analysis)", "Price vs Volume MA20", adv_signals["vpa"]["signal"]),
            ("MACD Divergence", "Momentum vs Price", adv_signals["macd_divergence"]["signal"]),
            ("Candlestick (Master)", adv_signals.get("candlestick",{}).get("pattern","N/A"), adv_signals.get("candlestick",{}).get("signal","N/A")),
            ("ADX (Trend Strength)", adv_signals.get("adx",{}).get("value","N/A"), adv_signals.get("adx",{}).get("trend_strength","N/A")),
            ("OBV (Accumulation)", "Signal", adv_signals.get("obv",{}).get("signal","N/A")),
            ("MACD", adv_signals["macd"]["value"], adv_signals["macd"]["signal"]),
            ("Bollinger", f'{adv_signals["bollinger"]["position"]}%' if adv_signals["bollinger"]["position"] else "N/A", adv_signals["bollinger"]["signal"]),
            ("ATR Volatility", adv_signals["atr"]["value"], adv_signals["atr"]["volatility"]),
        ]
        sr = "".join([f"<tr><td><strong>{n}</strong></td><td>{v if v else 'N/A'}</td><td>{s}</td></tr>" for n,v,s in sd])
        st.markdown(f"<table class='rt'><thead><tr><th>Indikator</th><th>Nilai</th><th>Sinyal</th></tr></thead><tbody>{sr}</tbody></table>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TRADING TARGETS
    st.markdown("<div class='sh'>🎯 Target Trading (Multi-Timeframe)</div>", unsafe_allow_html=True)
    t1,t2,t3 = st.columns(3)
    for col, (label, icon, tgt) in zip([t1,t2,t3], [("Day Trade","⚡",trading_targets["day_trade"]),("Swing Trade","🔄",trading_targets["swing_trade"]),("Value Investing","💎",trading_targets["value_invest"])]):
        with col:
            bc = tgt.get("bias_color","gray")
            tcm = {"green":"#4ade80","lightgreen":"#86efac","red":"#f87171","orange":"#fbbf24","gray":"#e2e8f0"}
            tclr = tcm.get(bc, "#e2e8f0")
            is_s = tgt.get("is_short", False)
            tgt_c = "#f87171" if is_s else "#4ade80"
            stp_c = "#4ade80" if is_s else "#f87171"
            st.markdown(f"""<div class='tc' style='border-color:{tclr}40;'><div class='th' style='border-bottom-color:{tclr}80;'>{icon} {label} ({tgt.get('timeframe','N/A')})</div><div class='tr'><span class='tk'>Bias</span><span class='tv' style='color:{tclr};'>{tgt.get('bias','N/A')}</span></div><div class='tr'><span class='tk'>Entry</span><span class='tv' style='color:#60a5fa;'>Rp {tgt.get('entry',0):,.0f}</span></div><div class='tr'><span class='tk'>⭐ Best Entry</span><span class='tv' style='color:#a78bfa;'>Rp {tgt.get('best_entry',0):,.0f}</span></div><div class='tr'><span class='tk'>🎯 Target{'(Short)' if is_s else ''}</span><span class='tv' style='color:{tgt_c};'>Rp {tgt.get('target',0):,.0f}</span></div><div class='tr'><span class='tk'>🛑 Stop Loss</span><span class='tv' style='color:{stp_c};'>Rp {tgt.get('stop_loss',0):,.0f}</span></div><div class='tr'><span class='tk'>R:R</span><span class='tv'>1 : {tgt.get('risk_reward',0)}</span></div><div style='color:#64748b;font-size:0.78rem;margin-top:6px;'>{tgt.get('method','')}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    st.markdown("<div class='sh'>📉 Harga, MA & RSI</div>", unsafe_allow_html=True)
    ft = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.08,row_heights=[0.7,0.3],subplot_titles=("Harga & Indikator",f"RSI ({TECHNICAL_RSI_PERIOD})"))
    ft.add_trace(go.Scatter(x=tech_df.index,y=tech_df["Close"],name="Close",line=dict(color="#60a5fa",width=2)),row=1,col=1)
    if mac in tech_df.columns: ft.add_trace(go.Scatter(x=tech_df.index,y=tech_df[mac],name=f"MA{TECHNICAL_MA_PERIOD}",line=dict(color="#fbbf24",width=2,dash="dot")),row=1,col=1)
    if "MA_50" in tech_df.columns: ft.add_trace(go.Scatter(x=tech_df.index,y=tech_df["MA_50"],name="MA50",line=dict(color="#38bdf8",width=1.5)),row=1,col=1)
    rc = f"RSI_{TECHNICAL_RSI_PERIOD}"
    if rc in tech_df.columns:
        ft.add_trace(go.Scatter(x=tech_df.index,y=tech_df[rc],name="RSI",line=dict(color="#a78bfa",width=2),fill="tozeroy",fillcolor="rgba(167,139,250,0.1)"),row=2,col=1)
        ft.add_hline(y=70,line_dash="dash",line_color="rgba(248,113,113,0.5)",annotation_text="Overbought",row=2,col=1)
        ft.add_hline(y=30,line_dash="dash",line_color="rgba(74,222,128,0.5)",annotation_text="Oversold",row=2,col=1)
    ft.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.5)",height=550,margin=dict(l=10,r=10,t=40,b=10),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),font=dict(family="Inter",color="#94a3b8"))
    ft.update_yaxes(gridcolor="rgba(100,116,139,0.15)"); ft.update_xaxes(gridcolor="rgba(100,116,139,0.15)")
    st.plotly_chart(ft, use_container_width=True)

    macd_cols = [c for c in tech_df.columns if "MACD" in c]
    if len(macd_cols) >= 2:
        st.markdown("<div class='sh'>📊 MACD</div>", unsafe_allow_html=True)
        fm = go.Figure()
        ml = [c for c in macd_cols if "MACD_" in c and "s_" not in c.lower() and "h_" not in c.lower()]
        ms = [c for c in macd_cols if "s_" in c.lower()]; mh = [c for c in macd_cols if "h_" in c.lower()]
        if ml: fm.add_trace(go.Scatter(x=tech_df.index,y=tech_df[ml[0]],name="MACD",line=dict(color="#60a5fa",width=2)))
        if ms: fm.add_trace(go.Scatter(x=tech_df.index,y=tech_df[ms[0]],name="Signal",line=dict(color="#f87171",width=2,dash="dot")))
        if mh:
            hc = ["#4ade80" if v>=0 else "#f87171" for v in tech_df[mh[0]].fillna(0)]
            fm.add_trace(go.Bar(x=tech_df.index,y=tech_df[mh[0]],name="Histogram",marker_color=hc,opacity=0.6))
        fm.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.5)",height=280,margin=dict(l=10,r=10,t=10,b=10),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),font=dict(family="Inter",color="#94a3b8"),yaxis=dict(gridcolor="rgba(100,116,139,0.15)"),xaxis=dict(gridcolor="rgba(100,116,139,0.15)"))
        st.plotly_chart(fm, use_container_width=True)

# =================== TAB 4: ENTRY STRATEGY ===================
with tab4:
    st.markdown("<div class='sh'>🎯 Skenario Entry (Per Strategi)</div>", unsafe_allow_html=True)
    st.markdown("""<div class='ib' style='border-color:rgba(167,139,250,0.3);background:linear-gradient(135deg,rgba(88,28,135,0.15),rgba(126,34,206,0.1));'>💡 <strong>Petunjuk:</strong> Setiap skenario punya kondisi yang harus terpenuhi sebelum entry. Tunggu konfirmasi — jangan entry tanpa sinyal.</div>""", unsafe_allow_html=True)

    for style_key, style_name, style_icon in [("day_trade","Day Trade","⚡"),("swing_trade","Swing Trade","🔄"),("value_invest","Value Investing","💎")]:
        st.markdown(f"<div class='sh'>{style_icon} {style_name}</div>", unsafe_allow_html=True)
        scenarios = entry_scenarios.get(style_key, [])
        for sc in scenarios:
            prob_c = "#4ade80" if "Tinggi" in str(sc.get("probability","")) or "Siap" in str(sc.get("probability","")) or "Selalu" in str(sc.get("probability","")) else ("#fbbf24" if "Sedang" in str(sc.get("probability","")) else "#f87171")
            st.markdown(f"""<div class='es-card'><div class='es-name'>{sc.get('name','')}</div><div class='es-cond'>📋 <strong>Kondisi:</strong> {sc.get('condition','')}</div><div class='es-row'><span class='es-k'>Entry</span><span class='es-v' style='color:#60a5fa;'>Rp {sc.get('entry',0):,.0f}</span></div><div class='es-row'><span class='es-k'>🎯 Target</span><span class='es-v' style='color:#4ade80;'>Rp {sc.get('target',0):,.0f}</span></div><div class='es-row'><span class='es-k'>🛑 Stop Loss</span><span class='es-v' style='color:#f87171;'>Rp {sc.get('stop',0):,.0f}</span></div><div class='es-row'><span class='es-k'>📊 Probabilitas</span><span class='es-v' style='color:{prob_c};'>{sc.get('probability','N/A')}</span></div><div class='es-reason'>💬 {sc.get('rationale','')}</div></div>""", unsafe_allow_html=True)

# =================== TAB 5: PREDICTION ===================
with tab5:
    st.markdown("<div class='sh'>🔮 Prediksi Harga (Dikonfirmasi Teknikal)</div>", unsafe_allow_html=True)
    st.markdown("""<div class='ib' style='border-color:rgba(251,191,36,0.3);background:linear-gradient(135deg,rgba(120,53,15,0.2),rgba(146,64,14,0.1));'>⚠️ <strong>Disclaimer:</strong> Prediksi statistik + konfirmasi teknikal. BUKAN jaminan.</div>""", unsafe_allow_html=True)

    lin = prediction["linear"]; ma_proj = prediction["ma_projection"]; sr_data = prediction["support_resistance"]
    poly = prediction["poly"]
    cf = confirmed
    fd = cf.get('final_direction', cf.get('prediction_direction','N/A'))
    pd_raw = cf.get('prediction_direction','N/A')
    override_note = f"<div style='color:#fbbf24;font-size:0.82rem;margin-top:6px;'>⚠️ Linear: {pd_raw} — di-override teknikal</div>" if fd != pd_raw else ""
    st.markdown(f"<div class='mc' style='text-align:center;'><div class='ml'>Arah Final: {fd}</div><div class='sb {cf.get('confirmation_color','gry')}' style='font-size:1.05rem;margin:10px 0;'>{cf.get('confirmation_level','N/A')}</div><div style='color:#94a3b8;font-size:0.85rem;'>{cf.get('agreements',0)}/{cf.get('total_checks',0)} indikator setuju ({cf.get('ratio',0)}%)</div>{override_note}</div>", unsafe_allow_html=True)

    # Check details
    chk = cf.get("checks", [])
    if chk:
        cr = "".join([f"<tr><td>{c[0]}</td><td>{'✅' if c[1] else '❌'}</td><td>{c[2]}</td></tr>" for c in chk])
        st.markdown(f"<table class='rt'><thead><tr><th>Indikator</th><th>Status</th><th>Detail</th></tr></thead><tbody>{cr}</tbody></table>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    with p1: st.markdown(f"<div class='pc'><div class='pl'>Harga Saat Ini</div><div class='pv blu'>Rp {lin.get('current_price',0):,.0f}</div></div>", unsafe_allow_html=True)
    with p2:
        pe = lin.get("predicted_price_end",0) or 0; chg = lin.get("change_pct",0)
        clr = "grn" if chg>0 else ("red" if chg<0 else "org")
        st.markdown(f"<div class='pc'><div class='pl'>Prediksi {pred_days}H</div><div class='pv {clr}'>Rp {pe:,.0f}</div><div style='color:{'#4ade80' if chg>0 else '#f87171'};font-size:0.95rem;font-weight:700;margin-top:4px;'>{'+' if chg>0 else ''}{chg}%</div></div>", unsafe_allow_html=True)
    with p3: st.markdown(f"<div class='pc'><div class='pl'>Tren</div><div style='font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-top:10px;'>{lin.get('direction','N/A')}</div></div>", unsafe_allow_html=True)
    with p4:
        uc_clr = "#4ade80" if unified_conf["score"] >= 70 else ("#fbbf24" if unified_conf["score"] >= 40 else "#f87171")
        st.markdown(f"<div class='pc'><div class='pl'>Unified Confidence</div><div style='font-size:1.6rem;font-weight:700;color:{uc_clr};margin-top:10px;'>{unified_conf['score']}%</div><div style='color:#94a3b8;font-size:0.82rem;margin-top:4px;'>{unified_conf['label']} (R²={unified_conf['r2']})</div></div>", unsafe_allow_html=True)

    sr1,sr2,sr3,sr4 = st.columns(4)
    with sr1: st.markdown(f"<div class='pc'><div class='pl'>R2</div><div class='pv red'>Rp {sr_data.get('resistance_2',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with sr2: st.markdown(f"<div class='pc'><div class='pl'>R1</div><div class='pv org'>Rp {sr_data.get('resistance_1',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with sr3: st.markdown(f"<div class='pc'><div class='pl'>S1</div><div class='pv org'>Rp {sr_data.get('support_1',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with sr4: st.markdown(f"<div class='pc'><div class='pl'>S2</div><div class='pv grn'>Rp {sr_data.get('support_2',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)

    # Fibonacci Retracement
    st.markdown("<div class='sh'>📐 Fibonacci Retracement</div>", unsafe_allow_html=True)
    fb1, fb2, fb3, fb4, fb5, fb6 = st.columns(6)
    with fb1: st.markdown(f"<div class='pc'><div class='pl'>0% (High)</div><div class='pv red'>Rp {fibonacci.get('fib_0',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with fb2: st.markdown(f"<div class='pc'><div class='pl'>23.6%</div><div class='pv org'>Rp {fibonacci.get('fib_236',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with fb3: st.markdown(f"<div class='pc'><div class='pl'>38.2%</div><div class='pv org'>Rp {fibonacci.get('fib_382',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with fb4: st.markdown(f"<div class='pc'><div class='pl'>50%</div><div class='pv blu'>Rp {fibonacci.get('fib_500',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with fb5: st.markdown(f"<div class='pc'><div class='pl'>61.8%</div><div class='pv grn'>Rp {fibonacci.get('fib_618',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)
    with fb6: st.markdown(f"<div class='pc'><div class='pl'>100% (Low)</div><div class='pv grn'>Rp {fibonacci.get('fib_100',0) or 0:,.0f}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>📈 Grafik Prediksi</div>", unsafe_allow_html=True)
    fp = go.Figure()
    fp.add_trace(go.Scatter(x=price_df.index,y=price_df["Close"],name="Historis",line=dict(color="#60a5fa",width=2)))
    if lin.get("trend_line") is not None and not lin["trend_line"].empty:
        fp.add_trace(go.Scatter(x=lin["trend_line"].index,y=lin["trend_line"].values,name="Trend",line=dict(color="#fbbf24",width=2,dash="dash")))
    prd = lin.get("predicted_prices")
    if prd is not None and not prd.empty:
        fp.add_trace(go.Scatter(x=prd["Date"],y=prd["Predicted_Price"],name="Linear",line=dict(color="#4ade80",width=2,dash="dot")))
        if "Upper_Bound" in prd.columns:
            fp.add_trace(go.Scatter(x=prd["Date"],y=prd["Upper_Bound"],line=dict(color="rgba(74,222,128,0.3)",width=1),showlegend=False))
            fp.add_trace(go.Scatter(x=prd["Date"],y=prd["Lower_Bound"],line=dict(color="rgba(74,222,128,0.3)",width=1),fill="tonexty",fillcolor="rgba(74,222,128,0.05)",showlegend=False))
    mpd = ma_proj.get("predicted_prices")
    if mpd is not None and not mpd.empty:
        fp.add_trace(go.Scatter(x=mpd["Date"],y=mpd["Predicted_Price"],name="MA Proj",line=dict(color="#a78bfa",width=2,dash="dashdot")))
    
    # v11: Poly curve visualization (approximate extrapolation)
    if poly.get("acceleration") != 0:
        poly_label = "Trend Curve"
        fp.add_trace(go.Scatter(x=prd["Date"], y=prd["Predicted_Price"] * (1 + poly["acceleration"]*np.arange(len(prd))/100), name=poly_label, line=dict(color="#f472b6", width=2, dash="dash")))
    if sr_data.get("resistance_1"): fp.add_hline(y=sr_data["resistance_1"],line_dash="dash",line_color="rgba(248,113,113,0.4)",annotation_text=f"R1: {sr_data['resistance_1']:,.0f}")
    if sr_data.get("support_1"): fp.add_hline(y=sr_data["support_1"],line_dash="dash",line_color="rgba(74,222,128,0.4)",annotation_text=f"S1: {sr_data['support_1']:,.0f}")
    fp.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.5)",height=480,margin=dict(l=10,r=10,t=30,b=10),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),font=dict(family="Inter",color="#94a3b8"),yaxis=dict(title="Harga (Rp)",gridcolor="rgba(100,116,139,0.15)"),xaxis=dict(gridcolor="rgba(100,116,139,0.15)"))
    st.plotly_chart(fp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>🎲 Monte Carlo Simulation (Probabilistik)</div>", unsafe_allow_html=True)
    st.markdown("""<div class='ib' style='border-color:rgba(167,139,250,0.3);background:linear-gradient(135deg,rgba(88,28,135,0.15),rgba(126,34,206,0.1));'>💡 <strong>Memutar 1.000 simulasi acak (Random Walk)</strong> berdasarkan historis volatilitas saham ini. Percentile 90th = Best Case, 50th = Median/Wajar, 10th = Worst Case.</div>""", unsafe_allow_html=True)
    
    if monte_carlo and monte_carlo.get("dates"):
        mc_dates = monte_carlo["dates"]
        fmc = go.Figure()
        fmc.add_trace(go.Scatter(x=price_df.index[-60:], y=price_df["Close"].tail(60), name="Historis (60 Hari)", line=dict(color="#60a5fa", width=2)))
        
        # Best Case
        fmc.add_trace(go.Scatter(x=mc_dates, y=monte_carlo["best_case"], name="90th Percentile (Bullish)", line=dict(color="rgba(74,222,128,0.8)", width=1, dash="dot")))
        # Worst Case
        fmc.add_trace(go.Scatter(x=mc_dates, y=monte_carlo["worst_case"], name="10th Percentile (Bearish)", line=dict(color="rgba(248,113,113,0.8)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(74,222,128,0.05)"))
        # Median
        fmc.add_trace(go.Scatter(x=mc_dates, y=monte_carlo["median"], name="50th Percentile (Wajar)", line=dict(color="#a78bfa", width=2)))
        
        fmc.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.5)",height=480,margin=dict(l=10,r=10,t=30,b=10),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),font=dict(family="Inter",color="#94a3b8"),yaxis=dict(title="Harga (Rp)",gridcolor="rgba(100,116,139,0.15)"),xaxis=dict(gridcolor="rgba(100,116,139,0.15)"))
        st.plotly_chart(fmc, use_container_width=True)
        
        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.markdown(f"<div class='pc'><div class='pl'>Skema Bearish (10%)</div><div class='pv red'>Rp {monte_carlo['final_worst']:,.0f}</div></div>", unsafe_allow_html=True)
        with mc2: st.markdown(f"<div class='pc'><div class='pl'>Skema Wajar (50%)</div><div class='pv purp' style='color:#a78bfa;'>Rp {monte_carlo['final_median']:,.0f}</div></div>", unsafe_allow_html=True)
        with mc3: st.markdown(f"<div class='pc'><div class='pl'>Skema Bullish (90%)</div><div class='pv grn'>Rp {monte_carlo['final_best']:,.0f}</div></div>", unsafe_allow_html=True)

# =================== TAB 6: AI ===================
with tab6:
    prov_name = f"Groq ({groq_model_sel})" if ai_provider == "Groq" else "Gemini"
    prov_key = "groq" if ai_provider == "Groq" else "gemini"
    st.markdown(f"<div class='sh'>🤖 AI Analysis — {prov_name}</div>", unsafe_allow_html=True)
    if not ai_key:
        hu = "https://console.groq.com/keys" if prov_key == "groq" else "https://aistudio.google.com/app/apikey"
        st.markdown(f"<div class='ib'><strong>🔑 Cara:</strong><br>1. API Key GRATIS di <a href='{hu}' target='_blank' style='color:#60a5fa;'>{prov_name.split('(')[0].strip()}</a><br>2. Masukkan di sidebar → Klik analisis<br><br>{'<strong>Groq Free:</strong> 30 RPM, 14.400 req/hari' if prov_key == 'groq' else '<strong>Gemini:</strong> 15 RPM'}</div>", unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Analysis")
    if st.button("🚀 Jalankan Quick Analysis", type="primary", use_container_width=True):
        if not ai_key: st.warning("⚠️ Masukkan API Key.")
        else:
            with st.spinner(f"🧠 {prov_name} menganalisis (Fundamental + Teknikal + Prediksi)..."):
                result = quick_analysis(ticker, ratios, api_key=ai_key, provider=prov_key,
                    model_name=groq_model_id if groq_model_id else "llama-3.3-70b-versatile",
                    technical_context=technical_context_str)
            st.markdown(result)
    st.markdown("---")
    st.markdown("### 📄 Analisis PDF")
    if prov_key == "groq":
        st.info("💡 PDF hanya didukung Gemini. Pilih 'Gemini' di sidebar.")
    if uploaded_pdf:
        st.success(f"✅ {uploaded_pdf.name} ({uploaded_pdf.size/1024:.1f} KB)")
        if st.button("🔬 Analisis PDF", type="primary", use_container_width=True):
            if not ai_key: st.warning("⚠️ Masukkan API Key.")
            else:
                with st.spinner("🧠 Membaca PDF..."): result = analyze_pdf_report(uploaded_pdf.read(), ticker, api_key=ai_key, provider=prov_key)
                st.markdown(result)
    elif prov_key != "groq": st.info("👈 Upload PDF di sidebar.")

# =================== TAB 7: NEWS ===================
with tab7:
    st.markdown("<div class='sh'>📰 Berita Terbaru</div>", unsafe_allow_html=True)
    if news_items:
        for item in news_items[:5]:
            if item.get("news_type") == "ERROR":
                st.warning(item["title"]); continue
            sb = "RSS" if item.get("news_type") == "RSS" else "Yahoo"
            st.markdown(f"<div class='nc'><div class='nt'><a href=\"{item['link']}\" target=\"_blank\">{item['title']}</a></div><div class='nm'>📰 {item['publisher']} &nbsp;•&nbsp; 🕐 {item['published_time']} &nbsp;•&nbsp; <span style='background:rgba(96,165,250,0.2);padding:2px 6px;border-radius:4px;font-size:0.72rem;'>{sb}</span></div></div>", unsafe_allow_html=True)
    else:
        st.info("📭 Tidak ada berita.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>🔗 Portal Berita</div>", unsafe_allow_html=True)
    ext = get_external_news_links(ticker)
    st.markdown("<div style='margin:10px 0;'>" + " ".join([f"<a href='{l['url']}' target='_blank' class='el'>{l['name']}</a>" for l in ext]) + "</div>", unsafe_allow_html=True)
