"""
modules/ai_assistant.py — AI Analysis v5 (Multi-Model)
======================================================
Groq models: llama-3.3-70b, llama-3.1-8b, gemma2-9b, mixtral-8x7b
Gemini: gemini-2.0-flash (for PDF)
"""

from config import GEMINI_MODEL, GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE


GROQ_MODELS = {
    "Llama 3.3 70B (Best)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "Gemma 2 9B": "gemma2-9b-it",
    "Mixtral 8×7B": "mixtral-8x7b-32768",
}


def get_groq_model_list() -> list:
    return list(GROQ_MODELS.keys())


def _get_analysis_prompt(ticker: str, context: str = "", technical_context: str = "", mode: str = "quick") -> str:
    if mode == "pdf":
        return f"""
Kamu adalah analis keuangan profesional pasar saham Indonesia (IHSG).
Analisis Laporan Keuangan emiten: **{ticker}**

Berikan:
## 📊 Ringkasan Eksekutif
2-3 kalimat kondisi keuangan.

## ✅ 3 Alasan Membeli (Bullish Case)
1. **[Alasan 1]**: Penjelasan berbasis data
2. **[Alasan 2]**: Penjelasan berbasis data
3. **[Alasan 3]**: Penjelasan berbasis data

## ⚠️ 3 Risiko Utama (Bearish Case)
1. **[Risiko 1]**: Penjelasan berbasis data
2. **[Risiko 2]**: Penjelasan berbasis data
3. **[Risiko 3]**: Penjelasan berbasis data

## 📈 Metrik Kunci
3-5 angka kunci beserta trennya.

Analisis HANYA dari data PDF. Bahasa Indonesia. Perspektif Value Investing.
"""
    else:
        tech_block = f"\n--- DATA TEKNIKAL & PREDIKSI ---\n{technical_context}" if technical_context else ""
        return f"""
Kamu adalah analis saham Indonesia (IHSG) berpengalaman. Kamu menguasai Value Investing (Graham), Technical Analysis, dan Quantitative Methods.

Analisis LENGKAP untuk saham **{ticker}**:

--- DATA FUNDAMENTAL ---
{context}
{tech_block}

Berdasarkan SEMUA data di atas (Fundamental + Teknikal + Prediksi), berikan analisis berikut:

## 🎯 Verdict (1-2 kalimat)
Apakah saham ini layak beli/hold/jual SAAT INI? Sebutkan alasan utama.

## ✅ 3 Alasan Positif (dari data fundamental DAN teknikal)
1. ...
2. ...
3. ...

## ⚠️ 3 Risiko / Sinyal Negatif
1. ...
2. ...
3. ...

## 📊 Analisis Teknikal Singkat
Ringkas kondisi teknikal: apakah tren kuat/lemah, ada divergence, pola candlestick apa yang terbentuk, dan apa implikasinya.

## 🔮 Proyeksi & Rekomendasi
- **Day Trade**: Apakah cocok untuk scalping hari ini? Level entry/exit?
- **Swing (1-2 minggu)**: Apakah ada setup swing yang bagus?
- **Value Investing (1-3 bulan)**: Berdasarkan Monte Carlo & fundamental, apakah harga sekarang masih murah?

## ⚖️ Confidence Level
Seberapa yakin kamu dengan analisis ini (Rendah/Sedang/Tinggi)? Jelaskan kenapa.

Bahasa Indonesia. Berdasarkan DATA yang diberikan saja, jangan mengarang.
"""


def _analyze_with_groq(prompt: str, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=GEMINI_MAX_TOKENS,
            temperature=GEMINI_TEMPERATURE,
        )

        if response and response.choices:
            return response.choices[0].message.content
        return "⚠️ Groq tidak mengembalikan respons."

    except ImportError:
        return "❌ Library `groq` belum terinstall. Jalankan: `pip install groq`"
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower():
            return "⚠️ **Rate limit Groq tercapai.** Tunggu sebentar."
        return f"❌ Error Groq: {error_msg}"


def _analyze_with_gemini(prompt: str, api_key: str, pdf_bytes: bytes = None) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        content = [prompt, {"mime_type": "application/pdf", "data": pdf_bytes}] if pdf_bytes else prompt

        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=GEMINI_MAX_TOKENS,
                temperature=GEMINI_TEMPERATURE,
            ),
        )

        if response and response.text:
            return response.text
        return "⚠️ Gemini tidak mengembalikan respons."
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "⚠️ **Quota Gemini habis.** Gunakan Groq."
        return f"❌ Error Gemini: {error_msg}"


def analyze_pdf_report(pdf_bytes: bytes, ticker: str, api_key: str = None, provider: str = "gemini") -> str:
    if not api_key:
        return "❌ **API Key belum dikonfigurasi.**"
    if len(pdf_bytes) > 20 * 1024 * 1024:
        return f"❌ File terlalu besar ({len(pdf_bytes)/1024/1024:.1f} MB). Max 20 MB."
    if len(pdf_bytes) < 100:
        return "❌ File PDF kosong/corrupt."

    prompt = _get_analysis_prompt(ticker, mode="pdf")

    if provider == "groq":
        return "⚠️ **Groq tidak mendukung PDF.** Gunakan Gemini."
    return _analyze_with_gemini(prompt, api_key, pdf_bytes)


def quick_analysis(ticker: str, financial_data: dict, api_key: str = None,
                   provider: str = "groq", model_name: str = "llama-3.3-70b-versatile",
                   technical_context: str = "") -> str:
    if not api_key:
        return "❌ **API Key belum dikonfigurasi.**"

    data_text = "\n".join([f"- {k}: {v}" for k, v in financial_data.items()])
    prompt = _get_analysis_prompt(ticker, context=data_text, technical_context=technical_context, mode="quick")

    if provider == "groq":
        return _analyze_with_groq(prompt, api_key, model_name)
    return _analyze_with_gemini(prompt, api_key)
