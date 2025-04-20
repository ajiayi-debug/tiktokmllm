JSON_PATH = "data/Gemini_top8.json"     # ← put your file here

import json
from pathlib import Path
import sys

file = Path(JSON_PATH)

if not file.is_file():
    sys.exit(f"❌ File not found: {file}")

try:
    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    sys.exit(f"❌ Invalid JSON: {e}")

if not isinstance(data, list):
    sys.exit("❌ JSON root is not a list.")

print(f"Items in list: {len(data)}")