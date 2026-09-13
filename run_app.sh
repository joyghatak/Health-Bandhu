#!/usr/bin/env bash
# Start HealthBandhu V (Linux or macOS)
cd "$(dirname "$0")"
python -m streamlit run App.py "$@"
