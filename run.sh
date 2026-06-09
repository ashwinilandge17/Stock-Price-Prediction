#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Stock Price Predictor — Quick Launch Script
# ─────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        📈  Stock Price Predictor — ML Dashboard         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌  Python3 not found. Please install Python 3.9+."
    exit 1
fi

# Install dependencies
echo "📦  Installing dependencies …"
pip install -q -r requirements.txt

echo ""
echo "✅  Dependencies ready."
echo "🚀  Launching Streamlit on http://localhost:8501"
echo ""

streamlit run Home.py
