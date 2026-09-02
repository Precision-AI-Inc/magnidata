#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d .venv ]; then
  python3 -m venv .venv
  echo "Created .venv"
fi

source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f .env ]; then
  echo "No .env found — running with defaults (copy .env.example to override DATALAKE_ROOT, THUMB_ROOT_*, DB_PATH, or API_PORT)"
fi

echo "Starting API on port ${API_PORT:-5050} …"
python app.py
