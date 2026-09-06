"""Minimal backend adapter. Replace ``payload`` with data from your sources."""

import json
from pathlib import Path

from news_hybrid import decide

payload = json.loads(Path("request.json").read_text(encoding="utf-8"))
result = decide(payload, state_path="state/news-hybrid.sqlite3")
print(json.dumps(result, ensure_ascii=False, indent=2))
